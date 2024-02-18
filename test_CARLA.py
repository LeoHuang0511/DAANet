import datasets
# from  config import cfg
import numpy as np
import torch
import datasets
from misc.utils import *
# from model.VIC import Video_Individual_Counter
# from model.video_crowd_count import video_crowd_count
from model.video_people_flux import DutyMOFANet
from model.points_from_den import get_ROI_and_MatchInfo

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
from misc.gt_generate import *


# from model.MatchTool.compute_metric import associate_pred2gt_point_vis

parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='CARLA',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TASK', type=str, default='FT',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--OUTPUT_DIR', type=str, default='./test_demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--TEST_INTERVALS', type=int, default=62,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SAVE_FREQ', type=int, default=20,
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


# parser.add_argument(
#     '--model_path', type=str, default='./exp/SENSE/03-22_17-33_SENSE_VGG16_FPN_5e-05/ep_15_iter_115000_mae_2.211_mse_3.677_seq_MAE_6.439_WRAE_9.506_MIAE_1.447_MOAE_1.474.pth',
#     help='pretrained weight path')



opt = parser.parse_args()

opt.VAL_INTERVALS = opt.TEST_INTERVALS


opt.MODE = 'test'



def test(cfg, cfg_data):
    print("model_path: ",cfg.MODEL_PATH)
        
    with torch.no_grad():
        net = DutyMOFANet(cfg, cfg_data)


        test_loader, restore_transform = datasets.loading_testset(cfg, mode='test')

        device = torch.device("cuda:"+str(torch.cuda.current_device()))


        state_dict = torch.load(cfg.MODEL_PATH,map_location=device)
        net.load_state_dict(state_dict, strict=True)

        net.cuda()
        net.eval()
        # sing_cnt_errors = {'cnt_mae': AverageMeter(), 'cnt_mse': AverageMeter(), 'flow_mae':AverageMeter(), 'flow_mae_inv':AverageMeter(),\
        #                        'flow_mse':AverageMeter(), 'flow_mse_inv':AverageMeter(), 'flow_mape':AverageMeter(), 'flow_mape_inv':AverageMeter(),\
        #                         "MIAE":AverageMeter()}
        scenes_pred_dict = []
        gt_flow_cnt = [232,204,278,82,349]
        scene_names = ['11','12','13','14','15']
        generate_gt = GenerateGT(cfg)
        get_roi_and_matchinfo = get_ROI_and_MatchInfo( cfg.TRAIN_SIZE, cfg.ROI_RADIUS, feature_scale=cfg.FEATURE_SCALE)

        
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}

        scenes_gt_dict =  []

        intervals = 1
        
        for scene_id, sub_valset in enumerate(test_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + cfg.TEST_INTERVALS
            print(video_time)
            scene_name = scene_names[scene_id]


            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': [], 'total_flow': gt_flow_cnt}
            img_pair_idx = 0
            for vi, data in enumerate(gen_tqdm, 0):
                img, target = data

                img,target = img[0], target[0]
                # scene_name = target[0]['scene_name']
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

                if frame_signal == 'skip':
                        
                        continue

                else:

                    den_scales, pred_map, mask, out_den, in_den, den_prob, io_prob, confidence, f_flow, b_flow, feature1, feature2, attn_1, attn_2 = net(img)

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

                    gt_io_map = torch.zeros(img_pair_num, 4, den_scales[0].size(2), den_scales[0].size(3)).cuda()

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
                    gt_mask = (gt_io_map>0).float()

                        #    -----------Counting performance------------------
                    gt_count, pred_cnt = gt_den[0].sum().item(),  pred_map[0].sum().item()

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)

                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])

                    pred_dict['inflow'].append(pre_inflow)
                    pred_dict['outflow'].append(pre_outflow)
                    gt_dict['inflow'].append(torch.tensor(gt_in_cnt).clone().detach())
                    gt_dict['outflow'].append(torch.tensor(gt_out_cnt).clone().detach())

                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, 1)

                    print(f'den_gt: {gt_count} den_pre: {pred_cnt} mae: {s_mae}')
                    print(f'gt_crowd_flow:{gt_crowdflow_cnt.cpu().numpy()}, gt_inflow: {gt_in_cnt.cpu().numpy()}')
                    print(f'pre_crowd_flow:{np.round(pre_crowdflow_cnt.cpu().numpy(),2)},  pre_inflow: {np.round(pre_inflow.cpu().numpy(),2)}')

                    img_pair_idx+=1
                    if img_pair_idx % cfg.SAVE_FREQ == 0:


                        save_results_mask(cfg, None, None, scene_name, (vi, vi+cfg.TEST_INTERVALS), restore_transform, 0, 
                                img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),\
                                pred_map[0].detach().cpu().numpy(), pred_map[1].detach().cpu().numpy(),out_den[0].detach().cpu().numpy(), in_den[0].detach().cpu().numpy(), gt_io_map[0].unsqueeze(0).detach().cpu().numpy(),\
                                (confidence[0,:,:,:]).unsqueeze(0).detach().cpu().numpy(),(confidence[1,:,:,:]).unsqueeze(0).detach().cpu().numpy(),\
                                f_flow , b_flow, [attn_1,attn_1,attn_1], [attn_2,attn_2,attn_2], den_scales, gt_den_scales, \
                                [mask,mask,mask], [gt_mask,gt_mask,gt_mask], [den_prob,den_prob,den_prob], [io_prob,io_prob,io_prob])
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)

        
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, intervals)
        # print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
        mae = sing_cnt_errors['mae'].avg
        mse = np.sqrt(sing_cnt_errors['mse'].avg)
        print('DEN_MAE: %.2f, DEN_MSE: %.2f, MAE: %.2f, MSE: %.2f  WRAE: %.2f MIAE: %.2f MOAE: %.2f' % (mae, mse, MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))


        print('Pre vs GT:', cnt_result)
        
        final_result = {'DEN_MAE':mae, 'DEN_MSE':mse, 'seq_MAE':MAE, 'seq_MSE':MSE, 'WRAE':WRAE, 'MIAE':MIAE, 'MOAE':MOAE}
        
        print(final_result)
        save_test_logger(cfg, cfg.OUTPUT_DIR, cnt_result, final_result)
        


            
 


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
    test(opt, cfg_data)

