import datasets
import numpy as np
import torch
import datasets
from misc.utils import *
from model.video_crowd_flux import DAANet
from model.points_from_den import get_ROI_and_MatchInfo

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
from misc.gt_generate import *



parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='SENSE',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TASK', type=str, default='FT',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--OUTPUT_DIR', type=str, default='./test_demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TEST_INTERVALS', type=int, default=15,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SKIP_FLAG', type=bool, default=True,
    help='if you need to caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SAVE_FREQ', type=int, default=200,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='0',
    help='Directory where to write output frames (If None, no output)')

parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)


parser.add_argument('--TRAIN_SIZE', type=int, nargs='+', default=[768,1024])
parser.add_argument('--FEATURE_SCALE', type=float, default=1/4.)


parser.add_argument('--DEN_FACTOR', type=float, default=200.)
parser.add_argument('--MEAN_STD', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
parser.add_argument('--ROI_RADIUS', type=float, default=4.)
parser.add_argument('--GAUSSIAN_SIGMA', type=float, default=4)
parser.add_argument('--CONF_BLOCK_SIZE', type=int, default=16)

parser.add_argument('--BACKBONE', type=str, default='vgg')



parser.add_argument(
    '--MODEL_PATH', type=str, default='',
    help='pretrained weight path')





opt = parser.parse_args()


opt.VAL_INTERVALS = opt.TEST_INTERVALS

opt.MODE = 'test'


def test(cfg, cfg_data):
    print("model_path: ",cfg.MODEL_PATH)

    with torch.no_grad():
        net = DAANet(cfg, cfg_data)
        with open(osp.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
            lines = f.readlines()
        scene_label = {}
        for line in lines:
            line = line.rstrip().split(' ')
            scene_label.update({line[0]: [int(i) for i in line[1:]] })

        # test_loader, restore_transform = datasets.loading_testset(cfg.DATASET, test_interval=cfg.TEST_INTERVALS,mode='test')
        test_loader, restore_transform = datasets.loading_testset(cfg, mode=cfg.MODE)
        device = torch.device("cuda:"+str(torch.cuda.current_device()))
        

        state_dict = torch.load(cfg.MODEL_PATH,map_location=device)
        
        net.load_state_dict(state_dict, strict=True)

        net.cuda()
        net.eval()

        generate_gt = GenerateGT(cfg)
        get_roi_and_matchinfo = get_ROI_and_MatchInfo( cfg.TRAIN_SIZE, cfg.ROI_RADIUS, feature_scale=cfg.FEATURE_SCALE)


        scenes_pred_dict = {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                        'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
        scenes_gt_dict =  {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                        'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}


        if cfg.SKIP_FLAG:
            intervals = 1
        else:
            intervals = cfg.TEST_INTERVALS

        for scene_id, sub_valset in enumerate(test_loader, 0):
            # if scene_id>2:
            #     break
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + cfg.TEST_INTERVALS
            print(video_time)

            scene_name = ''
            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            img_pair_idx = 0
            for vi, data in enumerate(gen_tqdm, 0):
                img, target = data
                # import pdb
                # pdb.set_trace()
                img,target = img[0],target[0]
                scene_name = target[0]['scene_name']
                img = torch.stack(img, 0).cuda()
                b, c, h, w = img.shape
                if h % 64 != 0:
                    pad_h = 64 - h % 64
                else:
                    pad_h = 0
                if w % 64 != 0:
                    pad_w = 64 - w % 64
                else:
                    pad_w = 0
                pad_dims = (0, pad_w, 0, pad_h)
                img = F.pad(img, pad_dims, "constant")
                img_pair_num = img.shape[0]//2

                if vi % cfg.TEST_INTERVALS == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'

                if frame_signal == 'match' or not cfg.SKIP_FLAG:

                    den_scales, pred_map, _, out_den, in_den, _, _, _, _, _ = net(img)

                    pre_inflow, pre_outflow = \
                        in_den.sum().detach().cpu(), out_den.sum().detach().cpu()
                   
                    target_ratio = pred_map.shape[2]/img.shape[2]

                    for b in range(len(target)):
                        target[b]["points"] = target[b]["points"] * target_ratio
                        target[b]["sigma"] = target[b]["sigma"] * target_ratio
                        
                        for key,data in target[b].items():
                            if torch.is_tensor(data):
                                target[b][key]=data.cuda()
                    #    -----------gt generate metric computation------------------
                        
                    gt_den_scales = generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales))
                    gt_den = gt_den_scales[0]
                    
                    assert pred_map.size() == gt_den.size()

                    gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda()

                    gt_in_cnt = torch.zeros(img_pair_num).detach()
                    gt_out_cnt = torch.zeros(img_pair_num).detach()

                    assert pred_map.size() == gt_den.size()

                    for pair_idx in range(img_pair_num):
                        count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                        
                        if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                            match_gt, _ = get_roi_and_matchinfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                            gt_io_map, gt_in_cnt, gt_out_cnt \
                                = generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
                                
                                # = generate_gt.get_pair_seg_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)

                        #    -----------Counting performance------------------
                    gt_count, pred_cnt = gt_den[0].sum().item(),  pred_map[0].sum().item()

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)

                    #===================================================================
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])


                    pred_dict['inflow'].append(pre_inflow)
                    pred_dict['outflow'].append(pre_outflow)
                    gt_dict['inflow'].append(torch.tensor(gt_in_cnt))
                    gt_dict['outflow'].append(torch.tensor(gt_out_cnt))



                
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, 1)

                    print(f'den_gt: {gt_count} den_pre: {pred_cnt} mae: {s_mae}')
                    print(f'gt_crowd_flow:{gt_crowdflow_cnt.cpu().numpy()}, gt_inflow: {gt_in_cnt.cpu().numpy()}')
                    print(f'pre_crowd_flow:{np.round(pre_crowdflow_cnt.cpu().numpy(),2)},  pre_inflow: {np.round(pre_inflow.cpu().numpy(),2)}')


                    img_pair_idx+=1
        
                 
            scenes_pred_dict['all'].append(pred_dict)
            scenes_gt_dict['all'].append(gt_dict)

            scene_l = scene_label[scene_name]
            if scene_l[0] == 0: scenes_pred_dict['in'].append(pred_dict);  scenes_gt_dict['in'].append(gt_dict)
            if scene_l[0] == 1: scenes_pred_dict['out'].append(pred_dict);  scenes_gt_dict['out'].append(gt_dict)
            if scene_l[1] == 0: scenes_pred_dict['day'].append(pred_dict);  scenes_gt_dict['day'].append(gt_dict)
            if scene_l[1] == 1: scenes_pred_dict['night'].append(pred_dict);  scenes_gt_dict['night'].append(gt_dict)
            if scene_l[2] == 0: scenes_pred_dict['scenic0'].append(pred_dict);  scenes_gt_dict['scenic0'].append(gt_dict)
            if scene_l[2] == 1: scenes_pred_dict['scenic1'].append(pred_dict);  scenes_gt_dict['scenic1'].append(gt_dict)
            if scene_l[2] == 2: scenes_pred_dict['scenic2'].append(pred_dict);  scenes_gt_dict['scenic2'].append(gt_dict)
            if scene_l[2] == 3: scenes_pred_dict['scenic3'].append(pred_dict);  scenes_gt_dict['scenic3'].append(gt_dict)
            if scene_l[2] == 4: scenes_pred_dict['scenic4'].append(pred_dict);  scenes_gt_dict['scenic4'].append(gt_dict)
            if scene_l[2] == 5: scenes_pred_dict['scenic5'].append(pred_dict);  scenes_gt_dict['scenic5'].append(gt_dict)
            if scene_l[3] == 0: scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
            if scene_l[3] == 1: scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
            if scene_l[3] == 2: scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
            if scene_l[3] == 3: scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
            if scene_l[3] == 4: scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict)
        
        dir = cfg.MODEL_PATH.replace('exp', cfg.OUTPUT_DIR).replace(cfg.MODEL_PATH.split('/')[-1],'' )
        if not os.path.isdir(dir):
            os.makedirs(dir)

        log_file = os.path.join(dir, 'log.txt')
        with open(log_file, 'a') as f:
            try:
                f.write(f'iter: {os.path.basename(cfg.MODEL_PATH).split("_")[1]}    iter: {os.path.basename(cfg.MODEL_PATH).split("_")[3]}\n\n')
            except:
                f.write('iter: N/A')
            f.write(f'model_path: {cfg.MODEL_PATH}\n\n')
            f.write(f'test_intervals: {cfg.TEST_INTERVALS}\n\n')

            
        

            for key in scenes_pred_dict.keys():
                s_pred_dict = scenes_pred_dict[key]
                s_gt_dict = scenes_gt_dict[key]
                MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, intervals)
                
                mae = sing_cnt_errors['mae'].avg
                mse = np.sqrt(sing_cnt_errors['mse'].avg)

                print('='*20, key, '='*20)
                
                print('DEN_MAE: %.2f, DEN_MSE: %.2f, MAE: %.2f, MSE: %.2f  WRAE: %.2f MIAE: %.2f MOAE: %.2f' % (mae, mse, MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
                f.write(f"{'='*20} {key} {'='*20}\n")
                f.write(f"DEN_MAE {mae}, DEN_MSE {mse}, MAE {MAE.data}, MSE {MSE.data}, WRAE {WRAE.data}, MIAE {MIAE.data}, MOAE {MOAE.data}\n")
                if key == 'all':
                    save_cnt_result = cnt_result
                    np.save(os.path.join(dir,'SENSE_cnt.py'),save_cnt_result.numpy())

                    print('Pre vs GT:', cnt_result)
                    # f.write(f'Pre vs GT: {cnt_result}\n')

            f.write(f"{'-'*250}\n\n")
            



def save_visImg( kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals,
                save_path, id0=None, id1=None, scene_id='',restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])

    text = [
        'VIC',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts1))
    ]
    small_text = [
        'Match Threshold: {:.2f}'.format(0.1),
        'Image Pair: {:06}:{:06}'.format(vi - intervals, vi)
    ]

    out, out_by_point = make_matching_plot_fast(
        last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=small_text, restore_transform=restore_transform,
        id0=id0, id1=id1)
    if save_path is not None:
        # print('==> Will write outputs to {}'.format(save_path))
        os.makedirs(save_path,mode =0o777, exist_ok=True)

        stem = '{}_{}_{}_matches'.format(scene_id, vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)


def generate_cycle_mask(height, width, back_color, fore_color):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    # ellipse mask
    cir_idx = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask = np.zeros((2 * height + 1, 2 * width + 1, 3)).astype(np.uint8)
    mask[cir_idx == 0] = back_color
    mask[cir_idx == 1] = fore_color
    return mask


def save_inflow_outflow_density(img, scores, pre_points, target, match_gt, save_path, scene_id, vi, intervals):
    scores = scores.cpu().numpy()
    _, __, img_h, img_w = img.size()
    gt_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    gt_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)

    RoyalBlue1 = np.array([255, 118, 72])  # np.array([205,82,180])
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    gt_inflow[:, :, 0:3] = RoyalBlue1
    gt_outflow[:, :, 0:3] = RoyalBlue1
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1
    # matched_mask = np.zeros(scores.shape)
    # matched_mask[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]] = 1
    # matched_mask[match_gt['un_a'], -1] = 1
    # matched_mask[-1, match_gt['un_b']] = 1
    kernel = 8
    wide = 2 * kernel + 1

    pre_outflow_p = pre_points[0][scores[:-1, -1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_outflow_p, target[0], match_gt['un_a'].cpu().numpy())

    # import pdb
    # pdb.set_trace()
    for row_id, pos in enumerate(pre_outflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if row_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if row_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[0]['points'][match_gt['un_a']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)

        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]

    # ================================pre_inflow========================
    pre_inflow_p = pre_points[1][scores[-1, :-1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_inflow_p, target[1], match_gt['un_b'].cpu().numpy())

    for column_id, pos in enumerate(pre_inflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if column_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if column_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[1]['points'][match_gt['un_b']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)

        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    # pred_inflow_map = cv2.applyColorMap((255 * pre_inflow / (pre_inflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    for row_id in match_gt['un_a'].cpu().numpy():
        # import pdb
        # pdb.set_trace()
        w, h = target[0]['points'][row_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]

        print(w, h)
    # gt_outflow_map = cv2.applyColorMap((255 * gt_outflow / (gt_outflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    for column_id in match_gt['un_b'].cpu().numpy():
        w, h = target[1]['points'][column_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                              max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    # gt_inflow_map = cv2.applyColorMap((255 * gt_inflow / (gt_inflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    os.makedirs(save_path, mode=0o777, exist_ok=True)
    stem = '{}_{}_{}_matches_outflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[:-1, -1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_outflow)

    stem = '{}_{}_{}_matches_inflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[-1, :-1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_inflow)

    stem = '{}_{}_{}_matches_outflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_a'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_outflow)

    stem = '{}_{}_{}_matches_inflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_b'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_inflow)
    # import pdb
    # pdb.set_trace()
if __name__=='__main__':
    import os
    import numpy as np
    import torch
    # from config import cfg
    from importlib import import_module
    


    # ------------prepare enviroment------------
    seed = opt.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(opt,cfg_data)

