# +
import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
from model.video_crowd_flux import DAANet
from model.loss import *
from tqdm import tqdm
import torch.nn.functional as F

from misc.gt_generate import *




import os
import random
import numpy as np
import torch
import datasets
from importlib import import_module

import argparse




class Trainer():
    def __init__(self,cfg, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.cfg = cfg

        self.device = torch.device("cuda:"+str(torch.cuda.current_device()))

        self.pwd = pwd
        self.cfg_data = cfg_data

        if cfg.RESUME_PATH != '':
            self.resume = True
        else:
            self.resume = False

        self.net = DAANet(cfg, cfg_data).cuda()


        params = [
            {"params": self.net.Extractor.parameters(), 'lr': cfg.LR_BASE, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.deformable_alignment.parameters(), "lr": cfg.LR_THRE, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.mask_predict_layer.parameters(), "lr": cfg.LR_THRE, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.ASAM.parameters(), "lr": cfg.LR_BASE, 'weight_decay': cfg.WEIGHT_DECAY},
            
        ]
        

        
        self.i_tb = 0
        self.epoch = 1
        

        
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20,'seq_MSE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        
        
        if self.cfg.RESUME_PATH != '':
            gpu_id = self.cfg.GPU_ID
            self.optimizer = optim.Adam(params)
            latest_state = torch.load(self.cfg.RESUME_PATH,map_location=self.device)
            self.net.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            self.cfg = latest_state['cfg']
            self.cfg.GPU_ID = gpu_id
            print("Finish loading resume model")

        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(self.cfg)
        self.optimizer = optim.Adam(params)


        
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.num_iters = self.cfg.MAX_EPOCH * np.int64(len(self.train_loader))
        self.compute_kpi_loss = ComputeKPILoss(self,cfg)

        self.generate_gt = GenerateGT(cfg)
        self.feature_scale = cfg.FEATURE_SCALE
        self.get_ROI_and_MatchInfo = get_ROI_and_MatchInfo( self.cfg.TRAIN_SIZE, self.cfg.ROI_RADIUS, feature_scale=self.feature_scale)


        self.writer, self.log_txt = logger(self.cfg, self.exp_path, self.exp_name, self.pwd, ['exp','test_demo', 'notebooks','.git'], resume=self.resume)

    def forward(self):
        for epoch in range(self.epoch, self.cfg.MAX_EPOCH):
            self.epoch = epoch
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

        
    def train(self): # training for all datasets

        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer,
                                    self.epoch,
                                    self.cfg.LR_BASE,
                                    self.cfg.LR_THRE,
                                    self.cfg.LR_DECAY)

        batch_loss = {'den':AverageMeter(), 'in':AverageMeter(), 'out':AverageMeter(), 'mask':AverageMeter(), 'con':AverageMeter(), 'scale_mask':AverageMeter(), 'scale_den':AverageMeter(), 'scale_io':AverageMeter()}

        loader = self.train_loader

        for i, data in enumerate(loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            img,target = data
            img = torch.stack(img,0).cuda()
            img_pair_num = img.size(0)//2  
            den_scales, final_den, mask, out_den, in_den, attns, f_flow, b_flow, feature1, feature2 = self.net(img)
            

            pre_inf_cnt, pre_out_cnt = in_den.sum(), out_den.sum()
            

            #    -----------gt generate & loss computation------------------
            target_ratio = den_scales[0].shape[2]/img.shape[2]

            for b in range(len(target)):        
                for key,data in target[b].items():
                    if torch.is_tensor(data):
                        target[b][key]=data.cuda()



            gt_den_scales = self.generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales))
            
        



            gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda()

            gt_inflow_cnt = torch.zeros(img_pair_num).cuda()
            gt_outflow_cnt = torch.zeros(img_pair_num).cuda()
            con_loss = torch.Tensor([0]).cuda()
            for pair_idx in range(img_pair_num):
                count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                
                if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                    match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                    gt_io_map, gt_inflow_cnt, gt_outflow_cnt \
                        = self.generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_outflow_cnt, gt_inflow_cnt, target_ratio)
                    
                    # contrastive loss
                    if len(match_gt['a2b'][:, 0]) > 0:

                        con_loss = con_loss + self.compute_kpi_loss.compute_con_loss(pair_idx, 
                                                                            feature1, 
                                                                            feature2, 
                                                                            match_gt, pois, 
                                                                            count_in_pair, 
                                                                            self.feature_scale)
            con_loss /= cfg.TRAIN_BATCH_SIZE
        
        
            gt_mask = (gt_io_map>0).float()
 
            kpi_loss = self.compute_kpi_loss(final_den, den_scales, gt_den_scales, mask, gt_mask,  out_den, in_den, gt_io_map, pre_inf_cnt, pre_out_cnt, gt_inflow_cnt, gt_outflow_cnt,attns)
            all_loss = (kpi_loss + con_loss * cfg.CON_WEIGHT ).sum()



            # back propagate
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
           

            batch_loss['den'].update(self.compute_kpi_loss.cnt_loss.sum().item())
            batch_loss['in'].update(self.compute_kpi_loss.in_loss.sum().item())
            batch_loss['out'].update(self.compute_kpi_loss.out_loss.sum().item())
            batch_loss['mask'].update(self.compute_kpi_loss.mask_loss.sum().item())
            batch_loss['scale_den'].update(self.compute_kpi_loss.cnt_loss_scales.sum().item())
            batch_loss['con'].update(con_loss.item())




            

            if (self.i_tb) % self.cfg.PRINT_FREQ == 0:
          
                self.writer.add_scalar('loss_den_overall',batch_loss['den'].avg, self.i_tb)
                self.writer.add_scalar('loss_den',batch_loss['scale_den'].avg, self.i_tb)
                self.writer.add_scalar('loss_mask', batch_loss['mask'].avg, self.i_tb)
                self.writer.add_scalar('loss_in', batch_loss['in'].avg, self.i_tb)
                self.writer.add_scalar('loss_out', batch_loss['out'].avg, self.i_tb)
                self.writer.add_scalar('loss_con', batch_loss['con'].avg, self.i_tb)
                self.writer.add_scalar('dynamic_weight',np.mean(self.compute_kpi_loss.dynamic_weight), self.i_tb)


                self.writer.add_scalar('base_lr', lr1, self.i_tb)
                self.writer.add_scalar('thre_lr', lr2, self.i_tb)





                self.timer['iter time'].toc(average=False)
               
                print('[ep %d][it %d][loss_den %.4f][loss_den_scale %.4f][loss_mask %.4f][loss_in %.4f][loss_out %.4f][loss_con %.4f][LR_BASE %f][LR_THRE %f][%.2fs]' % \
                        (self.epoch, self.i_tb, batch_loss['den'].avg,batch_loss['scale_den'].avg, batch_loss['mask'].avg, batch_loss['in'].avg,
                        batch_loss['out'].avg,batch_loss['con'].avg, lr1, lr2, self.timer['iter time'].diff))



            if (self.i_tb) % self.cfg.SAVE_VIS_FREQ == 0:
                
              
                save_results_mask(self.cfg, self.exp_path, self.exp_name, None, self.i_tb, self.restore_transform, 0, 
                                    img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),\
                                    final_den[0].detach().cpu().numpy(), final_den[1].detach().cpu().numpy(),out_den[0].detach().cpu().numpy(), in_den[0].detach().cpu().numpy(), gt_io_map[0].unsqueeze(0).detach().cpu().numpy(),\
                                    (attns[0,:,:,:]).unsqueeze(0).detach().cpu().numpy(),(attns[1,:,:,:]).unsqueeze(0).detach().cpu().numpy(),\
                                    f_flow , b_flow,  den_scales, gt_den_scales, 
                                    mask, gt_mask)




            if (self.i_tb % self.cfg.VAL_FREQ == 0) and  (self.i_tb > self.cfg.VAL_START):
                self.timer['val time'].tic()
                self.validate()
                self.net.train()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))
            



    def validate(self):
        with torch.no_grad():
            self.net.eval()
            sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
            scenes_pred_dict = []
            scenes_gt_dict = []
            
            for scene_id, sub_valset in  enumerate(self.val_loader, 0):

                gen_tqdm = tqdm(sub_valset)
                video_time = len(sub_valset)+self.cfg.VAL_INTERVALS
                print(video_time)

                pred_dict = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
                gt_dict  = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}    
                   
                for vi, data in enumerate(gen_tqdm, 0):
                    img, target = data
                    img, target = img[0],target[0]
                    
                    img = torch.stack(img,0).cuda()
                    img_pair_num = img.shape[0]//2
 
                    
                    b, c, h, w = img.shape
                    if h % 64 != 0: pad_h = 64 - h % 64
                    else: pad_h = 0
                    if w % 64 != 0: pad_w = 64 - w % 64
                    else: pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")



                    if vi % self.cfg.VAL_INTERVALS== 0 or vi ==len(sub_valset)-1:
                        frame_signal = 'match'
                    else: frame_signal = 'skip'

                    if frame_signal == 'skip':
                        
                        continue
                    
                    else:

                        den_scales, final_den, _, out_den, in_den, _, _, _, _, _, _, _, _, _ = self.net(img)

                        
                        pre_inf_cnt, pre_out_cnt = \
                            in_den.sum().detach().cpu(), out_den.sum().detach().cpu()
                        


                        target_ratio = final_den.shape[2]/img.shape[2]

                        for b in range(len(target)):
                            target[b]["points"] = target[b]["points"] * target_ratio
                            target[b]["sigma"] = target[b]["sigma"] * target_ratio
                            
                            for key,data in target[b].items():
                                if torch.is_tensor(data):
                                    target[b][key]=data.cuda()

                        #    -----------gt generate & loss computation------------------
                         
                        gt_den_scales = self.generate_gt.get_den(den_scales[0].shape, target, target_ratio, scale_num=len(den_scales))
                        gt_den = gt_den_scales[0]
                        
                        assert final_den.size() == gt_den.size()

                        gt_io_map = torch.zeros(img_pair_num, 2, den_scales[0].size(2), den_scales[0].size(3)).cuda()

                        gt_in_cnt = torch.zeros(img_pair_num).detach()
                        gt_out_cnt = torch.zeros(img_pair_num).detach()


                        for pair_idx in range(img_pair_num):
                            count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                            
                            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                                match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                                gt_io_map, gt_in_cnt, gt_out_cnt \
                                    = self.generate_gt.get_pair_io_map(pair_idx, target, match_gt, gt_io_map, gt_out_cnt, gt_in_cnt, target_ratio)
                                    # = self.generate_gt.get_pair_seg_map(pair_idx, target, match_gt, gt_io_map, gt_outflow_cnt, gt_inflow_cnt, target_ratio)


                        
                        
                        #    -----------Counting performance------------------
                        gt_count, pred_cnt = gt_den[0].sum().item(),  final_den[0].sum().item()
                                              

                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)

                

                        if vi == 0:
                            pred_dict['first_frame'] = final_den[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])


                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                        gt_dict['inflow'].append(torch.tensor(gt_in_cnt))
                        gt_dict['outflow'].append(torch.tensor(gt_out_cnt))

                        pre_crowdflow_cnt, gt_crowdflow_cnt,_,_ =compute_metrics_single_scene(pred_dict, gt_dict,1)# cfg.VAL_INTERVALS)
                        print(f'den_gt: {gt_count} den_pre: {pred_cnt} mae: {s_mae}')
                        print(f'gt_crowd_flow:{gt_crowdflow_cnt.cpu().numpy()}, gt_inflow: {gt_in_cnt.cpu().numpy()}')
                        print(f'pre_crowd_flow:{np.round(pre_crowdflow_cnt.cpu().numpy(),2)},  pre_inflow: {np.round(pre_inf_cnt.cpu().numpy(),2)}')



#                            
                scenes_pred_dict.append(pred_dict)
                scenes_gt_dict.append(gt_dict)
           
            MAE, MSE,WRAE, MIAE, MOAE, cnt_result =compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict, 1)#cfg.VAL_INTERVALS)
            print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pre vs GT:', cnt_result)
            mae = sing_cnt_errors['mae'].avg
            mse = np.sqrt(sing_cnt_errors['mse'].avg)

            self.writer.add_scalar('mae',mae, self.i_tb)
            self.writer.add_scalar('mse',mse, self.i_tb)
            self.writer.add_scalar('seqMAE',MAE, self.i_tb)
            self.writer.add_scalar('seqMSE',MSE, self.i_tb)
            self.writer.add_scalar('WRAE', WRAE, self.i_tb)
            self.writer.add_scalar('MIAE', MIAE, self.i_tb)
            self.writer.add_scalar('MOAE', MOAE, self.i_tb)


            self.train_record = update_model(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'seq_MSE':MSE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE },val=True)

            print_NWPU_summary_det(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'seq_MSE':MSE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
            


    


def compute_metrics_single_scene(pre_dict, gt_dict, intervals, target=True):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
    pre_crowdflow_cnt  = pre_dict['first_frame']

    if  target:
        gt_crowdflow_cnt =  gt_dict['first_frame']
        all_data = zip(pre_dict['inflow'],  pre_dict['outflow'],gt_dict['inflow'], gt_dict['outflow'])
    else:
        all_data = zip(pre_dict['inflow'],  pre_dict['outflow'])

    for idx, data in enumerate(all_data,0):

        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]
        if target:
            inflow_cnt[idx, 1] = data[2]
            outflow_cnt[idx, 1] = data[3]

        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]
            if target:
                gt_crowdflow_cnt += data[2]
    if target:
        return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt
    else:
        return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt


def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals, target=True):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2), 'MIAE':torch.zeros(0), 'MOAE':torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        if target:
            pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict,intervals,target)
        else:
            pre_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict,intervals,target)
        if 'total_flow' in scene_gt_dict[0].keys():
            gt_crowdflow_cnt = scene_gt_dict[0]['total_flow'][i]

        # print(pre_crowdflow_cnt)
        # print(gt_crowdflow_cnt)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

        if target:
            metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0]-inflow_cnt[:,1])])
            metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])

    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100
    if target:
        MIAE = torch.mean(metrics['MIAE'] )
        MOAE = torch.mean(metrics['MOAE'])

        return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']
    return MAE, MSE, WRAE, metrics['MAE']








if __name__=='__main__':
    
    # ------------prepare enviroment------------


    parser = argparse.ArgumentParser()
    parser.add_argument('--EXP_NAME', type=str, default='')
    parser.add_argument('--RESUME_PATH',type=str, default='')


    parser.add_argument('--GPU_ID', type=str, default='0')
    parser.add_argument('--SEED', type=int, default=3035)
    parser.add_argument('--DATASET', type=str, default='SENSE')
    parser.add_argument('--PRINT_FREQ', type=int, default=20)
    parser.add_argument('--SAVE_VIS_FREQ', type=int, default=500)
    parser.add_argument('--BACKBONE', type=str, default='vgg')


    parser.add_argument('--LR_MIN', type=float, default=1e-6)
    parser.add_argument('--LR_BASE', type=float, default=5e-5, help='density branch')
    parser.add_argument('--LR_THRE', type=float, default=1e-4, help='mask branch')
    parser.add_argument('--LR_DECAY', type=float, default=0.95)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-5)
    parser.add_argument('--WARMUP_EPOCH', type=int, default=3, help='number of epochs for warm up step in cosine annealing lr scheduler')
    parser.add_argument('--MAX_EPOCH', type=int, default=20)
    parser.add_argument('--WORKER', type=int, default=4)


    parser.add_argument('--CON_WEIGHT', type=float, default=0.5)
    parser.add_argument('--SCALE_WEIGHT', type=float, nargs='+', default=[2,0.1,0.01])
    parser.add_argument('--CNT_WEIGHT', type=float, default=10)
    parser.add_argument('--MASK_WEIGHT', type=float, default=1)
    parser.add_argument('--IO_WEIGHT', type=float, default=1)




    #_test or val
    parser.add_argument('--VAL_FREQ', type=int, default=2000)
    parser.add_argument('--VAL_START', type=int, default=0)
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)
    parser.add_argument('--VAL_INTERVALS', type=int, default=50)




    #_train
    parser.add_argument('--TRAIN_SIZE', type=int, nargs='+', default=[768,1024])
    parser.add_argument('--TRAIN_FRAME_INTERVALS', type=int, nargs='+', default=[40, 85])
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=2)
    parser.add_argument('--ROI_RADIUS', type=float, default=4.)
    parser.add_argument('--FEATURE_SCALE', type=float, default=1/4.)
    parser.add_argument('--GAUSSIAN_SIGMA', type=float, default=4)
    parser.add_argument('--CONF_BLOCK_SIZE', type=int, default=16)
    parser.add_argument('--CROP_RATE', type=float, nargs='+', default=[0.8,1.2])




    parser.add_argument('--DEN_FACTOR', type=float, default=200.)
    parser.add_argument('--MEAN_STD', type=tuple, default=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    cfg = parser.parse_args()
    

    now = time.strftime("%m-%d_%H-%M", time.localtime())

    if cfg.DATASET == "SENSE":
        cfg.TRAIN_FRAME_INTERVALS = [5,17]
        cfg.VAL_INTERVALS = 10
        # cfg.SAVE_VIS_FREQ = 5000
        # cfg.VAL_FREQ = 2500
    elif cfg.DATASET == "CARLA":
        cfg.CROP_RATE = [0.6, 1.2]
        cfg.LR_BASE = 1e-5
        cfg.VAL_INTERVALS = 62

    cfg.EXP_NAME = now \
    + '_' + cfg.BACKBONE\
    + '_' + cfg.EXP_NAME\
    + '_' + cfg.DATASET \
    + '_' + str(cfg.LR_BASE)\
    + '_' + str(cfg.LR_THRE)

    cfg.EXP_PATH = os.path.join('../exp', cfg.DATASET)  # the path of logs, checkpoints, and current codes
    
    cfg.MODE = 'train'

    




    # cfg.VAL_INTERVALS = (cfg.TRAIN_FRAME_INTERVALS[0]+cfg.TRAIN_FRAME_INTERVALS[1])//2

    if not os.path.exists(cfg.EXP_PATH ):
        os.makedirs(cfg.EXP_PATH )

  
    
    print(cfg)



    
    


    args = parser.parse_args()
    seed = cfg.SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
        
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg, cfg_data, pwd)
    cc_trainer.forward()
