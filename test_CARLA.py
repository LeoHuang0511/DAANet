import datasets
# from  config import cfg
import numpy as np
import torch
import datasets
from misc.utils import *
# from model.VIC import Video_Individual_Counter

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
from evaluation.adjustive_patch_indices import *
from evaluation.compute_region_flow import *
from evaluation import metrics
from misc.layer import Gaussianlayer

# from model.MatchTool.compute_metric import associate_pred2gt_point_vis

parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='CARLA',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--task', type=str, default='FineTune',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--output_dir', type=str, default='./test_demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--test_intervals', type=int, default=75,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--save_freq', type=int, default=2,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='0',
    help='Directory where to write output frames (If None, no output)')



parser.add_argument('--ADJ_SCALES', type=int, nargs='+', default=[1])
parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)


parser.add_argument('--GRID_SIZE', type=int, default=8)

parser.add_argument('--DEN_FACTOR', type=float, default=200.)
parser.add_argument('--MEAN_STD', type=tuple, default=([0.3467, 0.5197, 0.4980], [0.2125, 0.0232, 0.0410]))

parser.add_argument(
    '--model_path', type=str, default='',
    help='pretrained weight path')


# parser.add_argument(
#     '--model_path', type=str, default='./exp/SENSE/03-22_17-33_SENSE_VGG16_FPN_5e-05/ep_15_iter_115000_mae_2.211_mse_3.677_seq_MAE_6.439_WRAE_9.506_MIAE_1.447_MOAE_1.474.pth',
#     help='pretrained weight path')



opt = parser.parse_args()

opt.VAL_INTERVALS = opt.test_intervals


def test(cfg, cfg_data):
        
    with torch.no_grad():
        net = ARNet(opt, load_weights=True)
    #     with open(osp.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
    #         lines = f.readlines()
    #     scene_label = {}
    #     for line in lines:
    #         line = line.rstrip().split(' ')
    #         scene_label.update({line[0]: [int(i) for i in line[1:]] })


        test_loader, restore_transform = datasets.loading_testset(opt, mode='test')

        device = torch.device("cuda:" + opt.GPU_ID)

        Gaussian = Gaussianlayer(sigma=[4/cfg.GRID_SIZE]).to(device)


        state_dict = torch.load(opt.model_path,map_location=device)
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        net.to(device)
        # sing_cnt_errors = {'cnt_mae': AverageMeter(), 'cnt_mse': AverageMeter(), 'flow_mae':AverageMeter(), 'flow_mae_inv':AverageMeter(),\
        #                        'flow_mse':AverageMeter(), 'flow_mse_inv':AverageMeter(), 'flow_mape':AverageMeter(), 'flow_mape_inv':AverageMeter(),\
        #                         "MIAE":AverageMeter()}
        sing_cnt_errors = {'cnt_mae': AverageMeter(), 'cnt_mse': AverageMeter(), 'MIAE':AverageMeter()}

        scenes_pred_dict = []
        scenes_gt_dict =  []

        intervals = 1
        
        for scene_id, sub_valset in enumerate(test_loader, 0):
            # if scene_id>2:
            #     break
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + opt.test_intervals
            print(video_time)

            pred_dict = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': []}
            for vi, data in enumerate(gen_tqdm, 0):
                img, target = data
                # import pdb
                # pdb.set_trace()
                img,target = img[0], target[0]
                scene_name = target[0]['scene_name']
                img = torch.stack(img, 0).to(device)
                if cfg.task == 'ShiftPretrain':

                        break
                elif cfg.task == 'FineTune':    
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

                    if vi % opt.test_intervals == 0 or vi == len(sub_valset) - 1:
                        frame_signal = 'match'
                    else:
                        frame_signal = 'skip'

                    if frame_signal == 'skip':
                            
                            img_pair_num = img.shape[0]//2
                            gt_pass_frame_ids = get_regions_gt_cnt(img_pair_num, target, regions, previous_ids=gt_pass_frame_ids)


                    else:
                        pred_prev2c_flow_den, gt_prev2c_flow_den, pred_c2prev_flow_den, gt_c2prev_flow_den \
                                = net(img,target)
                        
                        dot_map = torch.zeros((img.shape[0],1,pred_c2prev_flow_den.shape[2],pred_c2prev_flow_den.shape[3])).to(device)


                        for i, data in enumerate(target):
                            points = (data['points']/cfg.GRID_SIZE).long()
                            dot_map[i, 0, points[:, 1], points[:, 0]] = 1
                        gt_den = Gaussian(dot_map).squeeze()

                        c_gt_den = gt_den[1::2]
                        prev_gt_den = gt_den[0::2]

                        # save_inflow_outflow_density(img, matched_results['scores'], matched_results['pre_points'],
                        #                             matched_results['target'], matched_results['match_gt'],
                        #                             osp.join(opt.output_dir, scene_name), scene_name, vi, opt.test_intervals)

                        #    -----------Counting performance------------------
                        # gt_count = torch.sum(gt_prev2c_flow_den).item()
                        # pred_cnt = torch.sum(pred_prev2c_flow_den).item()
                        gt_count = torch.sum(c_gt_den).item()
                        pred_cnt = torch.sum(pred_prev2c_flow_den[:,-1]).item()
                        

                        cnt_mae = abs(gt_count - pred_cnt)
                        cnt_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['cnt_mae'].update(cnt_mae)
                        sing_cnt_errors['cnt_mse'].update(cnt_mse)
                        # ============================================================

                        #===================================================================
                        # sing_cnt_errors['flow_mae'].update(metrics.mae(pred_prev2c_flow_den, gt_prev2c_flow_den))
                        # sing_cnt_errors['flow_mae_inv'].update(metrics.mae(pred_c2prev_flow_den, gt_c2prev_flow_den))

                        # sing_cnt_errors['flow_mse'].update(metrics.mse(pred_prev2c_flow_den, gt_prev2c_flow_den))
                        # sing_cnt_errors['flow_mse_inv'].update(metrics.mse(pred_c2prev_flow_den, gt_c2prev_flow_den))

                        # sing_cnt_errors['flow_mape'].update(metrics.pixel_mape(pred_prev2c_flow_den, gt_prev2c_flow_den))
                        # sing_cnt_errors['flow_mape_inv'].update(metrics.pixel_mape(pred_c2prev_flow_den, gt_c2prev_flow_den))

                        if vi == 0:
                            regions = get_region_verteces(cfg.ADJ_SCALES,(img.shape[2],img.shape[3]))
                            grids = find_passing_grids(regions, (img.shape[2],img.shape[3]), cfg.GRID_SIZE)

                            last_gt_crowdflow_cnt = torch.zeros((len(regions),1)).to(device)
                            last_pre_crowdflow_cnt = torch.zeros((len(regions),1)).to(device)

                            # gt_dict['first_frame'] = len(target[0]['person_id'])
                            # pred_inflow_cnt, pred_first_frame, gt_pass_frame_ids = compute_inflow(pred_prev2c_flow_den, pred_c2prev_flow_den, target, grids, regions, previous_ids=None)
                            pred_inflow_cnt, _, gt_pass_frame_ids = compute_inflow(pred_prev2c_flow_den[:,:-1], pred_c2prev_flow_den[:,:-1], target, grids, regions, previous_ids=None)
                            pred_first_frame = torch.sum(pred_c2prev_flow_den[:,-1]).unsqueeze(dim=0).unsqueeze(dim=0)
                            
                            pred_dict['first_frame'] = pred_first_frame.clone()
                            pred_dict['inflow'].append(pred_inflow_cnt)
                        else:
                            pred_inflow_cnt, _, gt_pass_frame_ids = compute_inflow(pred_prev2c_flow_den[:,:-1],pred_c2prev_flow_den[:,:-1], target, grids, regions, previous_ids=gt_pass_frame_ids)
                            pred_dict['inflow'].append(pred_inflow_cnt)



                        if frame_signal == 'match':
                            pre_crowdflow_cnt, gt_crowdflow_cnt =compute_metrics_single_scene(pred_dict, gt_pass_frame_ids)# cfg.VAL_INTERVALS)
                            gt_in_cnt = gt_crowdflow_cnt - last_gt_crowdflow_cnt
                            pre_inf_cnt = pre_crowdflow_cnt - last_pre_crowdflow_cnt
                            last_gt_crowdflow_cnt = gt_crowdflow_cnt.clone()
                            last_pre_crowdflow_cnt = pre_crowdflow_cnt.clone()
                            sing_cnt_errors['MIAE'].update(torch.abs(gt_in_cnt-pre_inf_cnt))


                            print(f'den_gt: {gt_count} den_pre: {pred_cnt} mae: {cnt_mae}')
                            print(f'gt_crowd_flow:{gt_crowdflow_cnt.squeeze().cpu().numpy()}, gt_inflow: {gt_in_cnt.squeeze().cpu().numpy()}')
                            print(f'pre_crowd_flow:{np.round(pre_crowdflow_cnt.squeeze().cpu().numpy(),2)},  pre_inflow: {np.round(pre_inf_cnt.squeeze().cpu().numpy(),2)}')
            
                if vi%(cfg.test_intervals*cfg.save_freq )== 0:
                    save_test_results(cfg, (target[0]['frame'],target[1]['frame']), opt.output_dir, scene_name, restore_transform,\
                                    img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0), pred_c2prev_flow_den[0].detach().cpu().numpy() , \
                                    torch.cat([gt_c2prev_flow_den[0],prev_gt_den[0][None,:]],0)[-2:].detach().cpu().numpy(), pred_prev2c_flow_den[0].detach().cpu().numpy(), torch.cat([gt_prev2c_flow_den[0],c_gt_den[0][None,:]],0)[-2:].detach().cpu().numpy())

    #                    
    # +
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_pass_frame_ids)

    #         scene_l = scene_label[scene_name]
    #         if scene_l[0] == 0: scenes_pred_dict['in'].append(pred_dict);  scenes_gt_dict['in'].append(gt_dict)
    #         if scene_l[0] == 1: scenes_pred_dict['out'].append(pred_dict);  scenes_gt_dict['out'].append(gt_dict)
    #         if scene_l[1] == 0: scenes_pred_dict['day'].append(pred_dict);  scenes_gt_dict['day'].append(gt_dict)
    #         if scene_l[1] == 1: scenes_pred_dict['night'].append(pred_dict);  scenes_gt_dict['night'].append(gt_dict)
    #         if scene_l[2] == 0: scenes_pred_dict['scenic0'].append(pred_dict);  scenes_gt_dict['scenic0'].append(gt_dict)
    #         if scene_l[2] == 1: scenes_pred_dict['scenic1'].append(pred_dict);  scenes_gt_dict['scenic1'].append(gt_dict)
    #         if scene_l[2] == 2: scenes_pred_dict['scenic2'].append(pred_dict);  scenes_gt_dict['scenic2'].append(gt_dict)
    #         if scene_l[2] == 3: scenes_pred_dict['scenic3'].append(pred_dict);  scenes_gt_dict['scenic3'].append(gt_dict)
    #         if scene_l[2] == 4: scenes_pred_dict['scenic4'].append(pred_dict);  scenes_gt_dict['scenic4'].append(gt_dict)
    #         if scene_l[2] == 5: scenes_pred_dict['scenic5'].append(pred_dict);  scenes_gt_dict['scenic5'].append(gt_dict)
    #         if scene_l[3] == 0: scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
    #         if scene_l[3] == 1: scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
    #         if scene_l[3] == 2: scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
    #         if scene_l[3] == 3: scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
    #         if scene_l[3] == 4: scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict)
        
        MAEs, RMSEs,WRAEs, cnt_result =compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict)#cfg.VAL_INTERVALS)
        # print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
        MAE = torch.mean(MAEs).item()
        RMSE = torch.mean(RMSEs).sqrt().item()
        WRAE = torch.mean(WRAEs).item()
        MIAE = sing_cnt_errors['MIAE'].avg.item()
        print(f'MAE: {MAE}, MSE: {RMSE}  WRAE: {WRAE}  MIAE: {MIAE}')


        print('Pre vs GT:', cnt_result)
        cnt_mae = np.round(sing_cnt_errors['cnt_mae'].avg, 3)
        cnt_rmse = np.round(np.sqrt(sing_cnt_errors['cnt_mse'].avg),3)
        # flow_mae = np.round(sing_cnt_errors['flow_mae'].avg.item(),3)
        # flow_mae_inv = np.round(sing_cnt_errors['flow_mae_inv'].avg.item(),3)
        # flow_rmse = np.round((sing_cnt_errors['flow_mse'].avg).sqrt().item(),3)
        # flow_rmse_inv = np.round((sing_cnt_errors['flow_mse_inv'].avg).sqrt().item(),3)
        # flow_mape = np.round(sing_cnt_errors['flow_mape'].avg.item(),3)
        # flow_mape_inv = np.round(sing_cnt_errors['flow_mape_inv'].avg.item(),3)

        # final_result = {'cnt_mae':cnt_mae, 'cnt_rmse':cnt_rmse, 'flow_mae':flow_mae, 'flow_mae_inv':flow_mae_inv,\
        #                 'flow_rmse':flow_rmse, 'flow_rmse_inv':flow_rmse_inv, 'flow_mape':flow_mape, 'flow_mape_inv':flow_mape_inv, 'seq_MAE':MAE, 'seq_MSE':RMSE, 'WRAE':WRAE, 'MIAE':MIAE}
        final_result = {'cnt_mae':cnt_mae, 'cnt_rmse':cnt_rmse, 'seq_MAE':MAE, 'seq_MSE':RMSE, 'WRAE':WRAE, 'MIAE':MIAE}
        
        print(final_result)
        save_test_logger(opt, opt.output_dir, cnt_result, MAEs,RMSEs,WRAEs, final_result)
        


            
 


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

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(opt, cfg_data)

