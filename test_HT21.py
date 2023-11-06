import datasets
# from  config import cfg
import numpy as np
import torch
import datasets
from misc.utils import *
# from model.VIC import Video_Individual_Counter
# from model.video_crowd_count import video_crowd_count
from model.SMDCA import SMDCANet

from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp


parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='HT21',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--task', type=str, default='FT',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--output_dir', type=str, default='../test_demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--test_intervals', type=int, default=60,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--skip_flag', type=bool, default=True,
    help='To caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--save_freq', type=int, default=2,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='0',
    help='Directory where to write output frames (If None, no output)')



parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)
parser.add_argument('--TemporalScale', type=int, default=2)



parser.add_argument('--TRAIN_SIZE', type=int, nargs='+', default=[768,1024])
parser.add_argument('--feature_scale', type=float, default=1/4.)


parser.add_argument('--DEN_FACTOR', type=float, default=200.)
parser.add_argument('--MEAN_STD', type=tuple, default=([0.3467, 0.5197, 0.4980], [0.2125, 0.0232, 0.0410]))

parser.add_argument(
    '--model_path', type=str, default='',
    help='pretrained weight path')


# parser.add_argument(
#     '--model_path', type=str, default='./exp/SENSE/03-22_17-33_SENSE_VGG16_FPN_5e-05/ep_15_iter_115000_mae_2.211_mse_3.677_seq_MAE_6.439_WRAE_9.506_MIAE_1.447_MOAE_1.474.pth',
#     help='pretrained weight path')



opt = parser.parse_args()
# opt.output_dir = opt.output_dir+'_'+opt.DATASET


opt.VAL_INTERVALS = opt.test_intervals

opt.TRAIN_BATCH_SIZE = opt.VAL_BATCH_SIZE

opt.mode = 'test'


def test(cfg, cfg_data):

    print("model_path: ",cfg.model_path)
        
    with torch.no_grad():
        # net = video_crowd_count(cfg, cfg_data)
        net = SMDCANet(cfg, cfg_data)

        test_loader, restore_transform = datasets.loading_testset(cfg, mode=cfg.mode)
        device = torch.device("cuda:"+str(torch.cuda.current_device()))

        state_dict = torch.load(cfg.model_path,map_location=device)
        # try:
        net.load_state_dict(state_dict, strict=True)
        # except:
        #     # net.load_state_dict(state_dict, strict=True)
        #     model_dict = net.state_dict()
        #     # load_dict = []
            
        #     state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        #     # list = {k:v for k,v in state_dict.items() if k in load_dict}

        #     model_dict.update(state_dict)
        #     net.load_state_dict(model_dict, strict=True)
        net.cuda()
        net.eval()
        scenes_pred_dict = []
        gt_flow_cnt = [133,737,734,1040,321]
        scene_names = ['HT21-11','HT21-12','HT21-13','HT21-14','HT21-15']

        if cfg.skip_flag:
            intervals = 1
        else:
            intervals = cfg.test_intervals
        
        for scene_id, sub_valset in enumerate(test_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + cfg.test_intervals
            print(video_time)
            scene_name = scene_names[scene_id]

            pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            img_pair_idx = 0
            for vi, data in enumerate(gen_tqdm, 0):
                    img, _ = data
                    img = img[0]
                    
                    img = torch.stack(img,0).cuda()
                    img_pair_num = img.shape[0]//2
                 
 
                    
                    b, c, h, w = img.shape
                    if h % 64 != 0: pad_h = 64 - h % 64
                    else: pad_h = 0
                    if w % 64 != 0: pad_w = 64 - w % 64
                    else: pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")



                    if vi % cfg.VAL_INTERVALS== 0 or vi ==len(sub_valset)-1:
                        frame_signal = 'match'
                    else: frame_signal = 'skip'

                    if frame_signal == 'skip':
                        
                        continue
                    
                    else:



                        den_scales, masks, confidence, f_flow, b_flow, feature1, feature2, attn_1, attn_2 = net(img)



                        final_den, out_den, in_den, den_probs, io_probs = net.scale_fuse(den_scales, masks, confidence, 'val')

                        




                        pre_inf_cnt, pre_out_cnt = \
                            in_den.sum().detach().cpu(), out_den.sum().detach().cpu()


                        #    -----------gt generate & loss computation------------------
                         
                        pred_cnt = final_den[0].sum().item()


                

                        if vi == 0:
                            pred_dict['first_frame'] = final_den[0].sum().item()


                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                       

                        pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, None, 1, target=False)
                        print(f'den_gt: {None} den_pre: {pred_cnt} mae: {None}')
                        print(f'pre_crowd_flow:{np.round(pre_crowdflow_cnt.squeeze().cpu().numpy(),2)},  pre_inflow: {np.round(pre_inf_cnt.squeeze().cpu().numpy(),2)}')

                        img_pair_idx+=1
            
                        if img_pair_idx % cfg.save_freq == 0:
                             if img_pair_idx % cfg.save_freq == 0:
                                gt_confidence = confidence.clone()
                                gt_den_scales = den_scales.copy()
                                gt_mask_scales = []
                                for i in range(3):
                                    

                                    gt_mask_scales.append(torch.zeros(1,2,masks[i].shape[2], masks[i].shape[3]).long())
                                


                                save_results_mask(cfg, None, None, scene_name, (vi, vi+cfg.test_intervals), restore_transform, 0, 
                                    img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),\
                                    final_den[0].detach().cpu().numpy(), final_den[1].detach().cpu().numpy(),out_den[0].detach().cpu().numpy(), in_den[0].detach().cpu().numpy(), \
                                    (confidence[0,:,:,:]).unsqueeze(0).detach().cpu().numpy(), (gt_confidence[0,:,:,:]).unsqueeze(0).detach().cpu().numpy(),(confidence[img.size(0)//2,:,:,:]).unsqueeze(0).detach().cpu().numpy(),(gt_confidence[img.size(0)//2,:,:,:]).unsqueeze(0).detach().cpu().numpy(),\
                                    f_flow , b_flow, attn_1, attn_2, den_scales, gt_den_scales, masks, gt_mask_scales, den_probs, io_probs)

    #                    
    # +
            scenes_pred_dict.append(pred_dict)


        
        MAE,MSE, WRAE, crowdflow_cnt  = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, intervals, target=False)
        # print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
        print(crowdflow_cnt)


        print('Pre vs GT:', crowdflow_cnt)
       
        final_result = {'seq_MAE':MAE, 'seq_MSE':MSE, 'WRAE':WRAE}
        
        print(final_result)
        save_test_logger(cfg, cfg.output_dir, crowdflow_cnt, final_result)
        


            
 


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

