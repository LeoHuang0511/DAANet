# +
import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
# from model.video_crowd_count import video_crowd_count
from model.SMDCA import SMDCANet
from model.loss import *
from tqdm import tqdm
import torch.nn.functional as F
# import matplotlib.cm as cm
# from pathlib import Path
from misc.lr_scheduler import CosineAnnealingWarmupRestarts

# from evaluation import metrics
# from misc.layer import Gaussianlayer
from misc.gt_generate import *




import os
import random
import numpy as np
import torch
import datasets
# from config import cfg
from importlib import import_module

import argparse




# from config import cfg
# from misc.KPI_pool import Task_KPI_Pool
# from thop import profile
class Trainer():
    def __init__(self,cfg, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.cfg = cfg

        # self.device = torch.device("cuda:" + cfg.GPU_ID)
        self.device = torch.device("cuda:"+str(torch.cuda.current_device()))

        self.pwd = pwd
        self.cfg_data = cfg_data
        # self.Gaussian = Gaussianlayer(sigma=[4/cfg.GRID_SIZE]).to(self.device)
        self.resume = cfg.RESUME

        # self.net = video_crowd_count(cfg, cfg_data)
        self.net = SMDCANet(cfg, cfg_data).cuda()


        params = [
            {"params": self.net.Extractor.parameters(), 'lr': cfg.LR_Base, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.deformable_alignment.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.mask_predict_layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
        ]
        
        # self.optimizer = optim.Adam(params)

        
        self.i_tb = 0
        self.epoch = 1
        # self.train_record = {'cnt_mae':1e20, 'cnt_rmse':1e20, 'flow_mae':1e20, 'flow_mae_inv':1e20,\
        #                     'flow_rmse':1e20, 'flow_rmse_inv':1e20, 'flow_mape':1e20, 'flow_mape_inv':1e20, 'seq_MAE':1e20, 'seq_MSE':1e20, 'WRAE':1e20, 'MIAE':1e20}
        if cfg.task == "SP":
            self.train_record = {'best_model_name': '', 'den_mae':1e20, 'den_mse':1e20, 'in_mae':1e20, 'in_mse':1e20,\
                                                'out_mae':1e20, 'out_mse':1e20}

        elif cfg.task == "FT":
            self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        
        
        if self.cfg.RESUME:
            self.optimizer = optim.Adam(params)
            latest_state = torch.load(self.cfg.RESUME_PATH,map_location=self.device)
            self.net.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            # self.lr_scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            self.cfg = latest_state['cfg']
            print("Finish loading resume model")

        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(self.cfg)

        if self.cfg.PRETRAIN:
            state_dict = torch.load(self.cfg.PRETRAIN_PATH,map_location=self.device)
            
            model_dict = self.net.state_dict()
            # load_dict = []
            # load_weights = ["optical_defromable_layer","Extractor.layer1","Extractor.layer2","Extractor.layer3","Extractor.neck","Extractor.neck2f"]
            # for k, v in state_dict.items():
            #     for module in load_weights:
            #         if module in k:
            #             load_dict.append(k)
            #             break

            # print(f"Loading weights of {load_dict}......")

            # pretrain_weight = torch.load(self.cfg.PRETRAIN_PATH, map_location=self.device)
            state_dict = {k:v for k,v in state_dict.items() if k in model_dict}

            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict, strict=True)
            if cfg.FROZEN:
                freeze_weights = ["Extractor.layer1","Extractor.layer2","Extractor.layer3","Extractor.neck","Extractor.neck2f"]

                for name, child in self.net.named_children():
                    for name_1, child_1 in child.named_children():
                        for module in freeze_weights:
                            if module in name + '.' + name_1:
                                for param in child_1.parameters():
                                    param.requires_grad = False
                                print(f"{name_1}'s weights are frozen!")
                for param in params:
                    param["params"] = filter(lambda p:p.requires_grad, param["params"])
                self.optimizer = optim.Adam(params)
            else:
                self.optimizer = optim.Adam(params)

                
            
            self.lr_scheduler_base = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=self.cfg.WARMUP_EPOCH*len(self.train_loader)*2,\
                                    cycle_mult=2,max_lr=self.cfg.LR_Base,min_lr=self.cfg.LR_MIN, warmup_steps=self.cfg.WARMUP_EPOCH*len(self.train_loader),gamma=0.8,group_index=[0])
            
            self.lr_scheduler_thre = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=self.cfg.WARMUP_EPOCH*len(self.train_loader)*2,\
                                    cycle_mult=2,max_lr=self.cfg.LR_Thre,min_lr=self.cfg.LR_MIN, warmup_steps=self.cfg.WARMUP_EPOCH*len(self.train_loader),gamma=0.8,group_index=[1,2])

               
            print("Finish loading pretrained model")
        else:
            self.optimizer = optim.Adam(params)
        
        
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.num_iters = self.cfg.MAX_EPOCH * np.int64(len(self.train_loader))
        # self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'pre_cnt'], 'mask': ['gt_cnt', 'acc_cnt']}, maximum_sample=1000)
        self.compute_kpi_loss = ComputeKPILoss(cfg)

        self.generate_gt = GenerateGT(cfg)
        self.feature_scale = cfg.feature_scale
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
        if not cfg.PRETRAIN:
            lr1, lr2 = adjust_learning_rate(self.optimizer,
                                    self.epoch,
                                    self.cfg.LR_Base,
                                    self.cfg.LR_Thre,
                                    self.cfg.LR_DECAY)

        batch_loss = {'den':AverageMeter(), 'in':AverageMeter(), 'out':AverageMeter(), 'mask':AverageMeter(), 'con':AverageMeter(), 'warp':AverageMeter(), 'scale_mask':AverageMeter(), 'scale_den':AverageMeter()}

        loader = self.train_loader

        for i, data in enumerate(loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            img,target = data

            if self.cfg.task == 'SP':
                flat_target = []
                for tar_pair in target:
                    for tar in tar_pair:
                        flat_target.append(tar)

                target = flat_target
                img = torch.cat(img,0).cuda()

            elif self.cfg.task == 'FT':
                img = torch.stack(img,0).cuda()
            

            
            img_pair_num = img.size(0)//2  
            den_scales, masks, pre_outflow_map, pre_inflow_map, f_flow, b_flow, feature1, feature2, attn_1, attn_2 = self.net(img)
            

            pre_inf_cnt = []
            pre_out_cnt = []
            for scale in range(len(pre_inflow_map)):
                pre_inf_cnt.append(pre_inflow_map[scale].sum(axis=2).sum(axis=2).detach())
                pre_out_cnt.append(pre_outflow_map[scale].sum(axis=2).sum(axis=2).detach())



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
            con_loss = torch.tensor([0]).cuda()
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
            
            gt_mask_scales = self.generate_gt.get_scale_io_masks( gt_io_map, scale_num=len(masks))

            # overall loss

            
            kpi_loss = self.compute_kpi_loss(den_scales, gt_den_scales,masks, gt_mask_scales, pre_inf_cnt, pre_out_cnt, gt_inflow_cnt, gt_outflow_cnt)
            


            # warp_loss = self.net.deformable_alignment.warp_loss
            all_loss = (kpi_loss + con_loss *cfg.con_alpha).sum()
            # all_loss = (kpi_loss + con_loss *cfg.con_alpha + warp_loss * cfg.warp_alpha + scale_mask_loss * cfg.scale_mask_alpha).sum()


            # back propagate
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            if cfg.PRETRAIN:
                lr1 = self.optimizer.param_groups[0]['lr']
                lr2 = self.optimizer.param_groups[1]['lr']
                self.lr_scheduler_base.step(self.i_tb)
                self.lr_scheduler_thre.step(self.i_tb)

            # self.lr_scheduler.step()
            batch_loss['den'].update(self.compute_kpi_loss.cnt_loss_scales.sum().item())
            batch_loss['in'].update(self.compute_kpi_loss.in_loss.sum().item())
            batch_loss['out'].update(self.compute_kpi_loss.out_loss.sum().item())
            batch_loss['mask'].update(self.compute_kpi_loss.mask_loss_scales.sum().item())
            batch_loss['con'].update(con_loss.item())

            # batch_loss['warp'].update(warp_loss.item())

            self.train_record = update_model(self, None, val=False)



            

            if (self.i_tb) % self.cfg.PRINT_FREQ == 0:
          
                self.writer.add_scalar('loss_den',batch_loss['den'].avg, self.i_tb)

                self.writer.add_scalar('loss_mask', batch_loss['mask'].avg, self.i_tb)
                self.writer.add_scalar('loss_in', batch_loss['in'].avg, self.i_tb)
                self.writer.add_scalar('loss_out', batch_loss['out'].avg, self.i_tb)
                self.writer.add_scalar('loss_con', batch_loss['con'].avg, self.i_tb)
                self.writer.add_scalar('base_lr', lr1, self.i_tb)
                self.writer.add_scalar('thre_lr', lr2, self.i_tb)





                self.timer['iter time'].toc(average=False)
               
                print('[ep %d][it %d][loss_den %.4f][loss_mask %.4f][loss_in %.4f][loss_out %.4f][loss_con %.4f][lr_base %f][lr_thre %f][%.2fs]' % \
                        (self.epoch, self.i_tb, batch_loss['den'].avg, batch_loss['mask'].avg,batch_loss['in'].avg,
                        batch_loss['out'].avg,batch_loss['con'].avg, lr1, lr2, self.timer['iter time'].diff))
                



            if (self.i_tb) % self.cfg.SAVE_VIS_FREQ == 0:
                # save_results_mask(self.cfg, self.exp_path, self.exp_name, None, self.i_tb, self.restore_transform, 0, 
                #                   img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0), 
                #                   den[0].detach().cpu().numpy() , gt_den[0].detach().cpu().numpy(), den[1].detach().cpu().numpy(), gt_den[1].detach().cpu().numpy() , \
                #                   (mask[0,:,:,:]).detach().cpu().numpy(), gt_mask[0,0:1,:,:].detach().cpu().numpy(), \
                #                   (mask[img.size(0)//2,:,:,:]).detach().cpu().numpy(), gt_mask[0,1:2,:,:].detach().cpu().numpy(),\
                #                   f_flow , b_flow, attn_1, attn_2, den_scales, gt_den_scales)
                save_results_mask(self.cfg, self.exp_path, self.exp_name, None, self.i_tb, self.restore_transform, 0, 
                                    img[0].clone().unsqueeze(0), img[1].clone().unsqueeze(0),\
                                    f_flow , b_flow, attn_1, attn_2, den_scales, gt_den_scales, masks, gt_mask_scales)


            # if (self.i_tb % self.cfg.VAL_FREQ == 0) and  (self.i_tb > self.cfg.VAL_START):
            #     self.timer['val time'].tic()
            #     if self.cfg.task == "SP":
            #         self.shift_validate()
            #     elif self.cfg.task == "FT":
            #         self.validate()
            #     self.net.train()
            #     self.timer['val time'].toc(average=False)
            #     print('val time: {:.2f}s'.format(self.timer['val time'].diff))
            
            torch.cuda.empty_cache()



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

                        den, _, pre_outflow_map, pre_inflow_map, _, _,_,_, _,_ = self.net(img)
                        pre_inf_cnt, pre_out_cnt = \
                            pre_inflow_map.sum(axis=2).sum(axis=2).detach().cpu(), pre_outflow_map.sum(axis=2).sum(axis=2).detach().cpu()
                        target_ratio = den.shape[2]/img.shape[2]

                        for b in range(len(target)):
                            target[b]["points"] = target[b]["points"] * target_ratio
                            target[b]["sigma"] = target[b]["sigma"] * target_ratio
                            
                            for key,data in target[b].items():
                                if torch.is_tensor(data):
                                    target[b][key]=data.cuda()

                        #    -----------gt generate & loss computation------------------
                         
                        gt_den = self.generate_gt.get_den(den.shape, target,1)
                        
                        gt_mask = torch.zeros(img_pair_num, 2, den.size(2), den.size(3)).cuda()
                        assert den.size() == gt_den.size()
                        gt_in_cnt = torch.zeros(img_pair_num).detach()
                        gt_out_cnt = torch.zeros(img_pair_num).detach()
                        for pair_idx in range(img_pair_num):
                            count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                            
                            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                                match_gt, _ = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                                _, gt_in_cnt, gt_out_cnt \
                                    = self.generate_gt.get_io_mask(pair_idx, target, match_gt, gt_mask, gt_out_cnt, gt_in_cnt, 1)


                        
                        
                        #    -----------Counting performance------------------
                        gt_count, pred_cnt = gt_den[0].sum().item(),  den[0].sum().item()

                        

                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)

                

                        if vi == 0:
                            pred_dict['first_frame'] = den[0].sum().item()
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
                torch.cuda.empty_cache()
           
            MAE, MSE,WRAE, MIAE, MOAE, cnt_result =compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict, 1)#cfg.VAL_INTERVALS)
            # print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
            print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pre vs GT:', cnt_result)
            mae = sing_cnt_errors['mae'].avg
            mse = np.sqrt(sing_cnt_errors['mse'].avg)

            self.writer.add_scalar('seqMAE',MAE, self.i_tb)
            self.writer.add_scalar('seqMSE',MSE, self.i_tb)
            self.writer.add_scalar('WRAE', WRAE, self.i_tb)
            self.writer.add_scalar('MIAE', MIAE, self.i_tb)
            self.writer.add_scalar('MOAE', MOAE, self.i_tb)


            self.train_record = update_model(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE },val=True)

            print_NWPU_summary_det(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
            


    def shift_validate(self):
        with torch.no_grad():
            self.net.eval()
            sing_cnt_errors = {'den_mae': AverageMeter(), 'den_mse': AverageMeter(), 'in_mae':AverageMeter(),'out_mae':AverageMeter(),'in_mse':AverageMeter(),'out_mse':AverageMeter() }
            # scenes_pred_dict = []
            # scenes_gt_dict = []
            gen_tqdm = tqdm(self.val_loader)

    
            for vi, data in enumerate(gen_tqdm, 0):
               
                img,target = data
                flat_target = []
                for tar_pair in target:
                    for tar in tar_pair:
                        flat_target.append(tar)

                target = flat_target
                img = torch.cat(img,0).cuda()
                img_pair_num = img.shape[0]//2
                for i in range(len(target)):
                    for key,data in target[i].items():
                        if torch.is_tensor(data):
                            target[i][key]=data.cuda()
                
                    
                den, _, pre_outflow_map, pre_inflow_map, _, _, _ = self.net(img)

                #    -----------gt generate & loss computation------------------
                gt_den = self.generate_gt.get_den(den.shape, target)
                
                gt_mask = torch.zeros(img_pair_num, 2, den.size(2), den.size(3)).cuda()
                assert den.size() == gt_den.size()
                gt_in_cnt = torch.zeros(img_pair_num).detach()
                gt_out_cnt = torch.zeros(img_pair_num).detach()
                for pair_idx in range(img_pair_num):
                    count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                    
                    if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                        match_gt, _ = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                        gt_mask, _, _ \
                            = self.generate_gt.get_io_mask(pair_idx, target, match_gt, gt_mask, gt_out_cnt, gt_in_cnt)
                        
                gt_outflow_map = gt_mask[:,0:1,:,:]* (gt_den[0::2,:,:,:].detach()) #* (mask[:,0:1,:,:] >= 0.8)
                gt_inflow_map = gt_mask[:,1:2,:,:] * (gt_den[1::2,:,:,:].detach()) #* (mask[:,1:2,:,:] >= 0.8)


                

                #    -----------Density map performance------------------
        
                s_den_mae = torch.sum(torch.abs(gt_den - den)).item()
                s_den_mse = F.mse_loss(den, gt_den, reduction='sum').item()
                sing_cnt_errors['den_mae'].update(s_den_mae)
                sing_cnt_errors['den_mse'].update(s_den_mse)

                #    -----------Flow performance------------------
                
                sing_cnt_errors['in_mae'].update(torch.sum(torch.abs(gt_inflow_map - pre_inflow_map)).item())
                sing_cnt_errors ['in_mse'].update(F.mse_loss(pre_inflow_map, gt_inflow_map,reduction='sum').item())
                sing_cnt_errors['out_mae'].update(torch.sum(torch.abs(gt_outflow_map - pre_outflow_map)).item())
                sing_cnt_errors ['out_mse'].update(F.mse_loss(pre_outflow_map, gt_outflow_map,reduction='sum').item())


            den_mae = sing_cnt_errors['den_mae'].avg
            den_mse = np.sqrt(sing_cnt_errors['den_mse'].avg)
            in_mae = sing_cnt_errors['in_mae'].avg
            in_mse =  np.sqrt(sing_cnt_errors['in_mse'].avg)
            out_mae = sing_cnt_errors['out_mae'].avg
            out_mse =  np.sqrt(sing_cnt_errors['out_mse'].avg)

           

            self.train_record = update_model(self,{'den_mae':den_mae, 'den_mse':den_mse, 'in_mae':in_mae, 'in_mse':in_mse,\
                                                'out_mae':out_mae, 'out_mse':out_mse})

            print_NWPU_summary_det(self,{'den_mae':den_mae, 'den_mse':den_mse, 'in_mae':in_mae, 'in_mse':in_mse,\
                                                'out_mae':out_mae, 'out_mse':out_mse})





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
            gt_crowdflow_cnt = gt_dict
            pre_crowdflow_cnt, inflow_cnt, outflow_cnt = compute_metrics_single_scene(pre_dict, gt_dict,intervals,target)

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

    parser.add_argument('--RESUME', default=False, action='store_true', help="resume previous training")
    parser.add_argument('--RESUME_PATH',type=str, default='')
    parser.add_argument('--PRETRAIN', default=False, action='store_true', help="resume previous training")
    parser.add_argument('--PRETRAIN_PATH',type=str, default='')
    parser.add_argument('--FROZEN', default=False, action='store_true', help="frozen pretrained frontend weights")


    parser.add_argument('--GPU_ID', type=str, default='0')
    parser.add_argument('--SEED', type=int, default=3035)
    parser.add_argument('--DATASET', type=str, default='HT21')
    parser.add_argument('--task', type=str, default='FT')
    parser.add_argument('--PRINT_FREQ', type=int, default=20)
    parser.add_argument('--SAVE_VIS_FREQ', type=int, default=800)




    parser.add_argument('--LR_MIN', type=float, default=1e-6)
    # parser.add_argument('--LR_MAX', type=float, default=1e-4)
    parser.add_argument('--LR_Base', type=float, default=5e-5, help='density branch')
    parser.add_argument('--LR_Thre', type=float, default=1e-4, help='mask branch')
    parser.add_argument('--LR_DECAY', type=float, default=0.95)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-5)
    parser.add_argument('--WARMUP_EPOCH', type=int, default=3, help='number of epochs for warm up step in cosine annealing lr scheduler')
    parser.add_argument('--MAX_EPOCH', type=int, default=20)
    parser.add_argument('--worker', type=int, default=16)


    parser.add_argument('--con_alpha', type=float, default=0.01)
    parser.add_argument('--con_scale', type=int, default=1)
    parser.add_argument('--intra_loss', default=False, action='store_true', help="intra loss")
    parser.add_argument('--intra_loss_alpha', type=float, default=0.1)



    parser.add_argument('--warp_alpha', type=float, default=0)
    parser.add_argument('--scale_mask_alpha', type=float, default=0)





    #_test or val
    parser.add_argument('--VAL_FREQ', type=int, default=1000)
    parser.add_argument('--VAL_START', type=int, default=14000)

    # parser.add_argument('--VAL_INTERVALS', type=int, default=75)
    # parser.add_argument('--ADJ_SCALES', type=int, nargs='+', default=[1])
    parser.add_argument('--VAL_BATCH_SIZE', type=int, default=1)
    parser.add_argument('--mask_threshold', type=float, default=0.5)



    #_train
    parser.add_argument('--TRAIN_SIZE', type=int, nargs='+', default=[768,1024])
    parser.add_argument('--GRID_SIZE', type=int, default=8)
    parser.add_argument('--TRAIN_FRAME_INTERVALS', type=int, nargs='+', default=[40, 85])
    parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--ROI_RADIUS', type=float, default=4.)
    parser.add_argument('--feature_scale', type=float, default=1/4.)
    parser.add_argument('--target_ratio', type=float, default=2)
    parser.add_argument('--gaussian_sigma', type=float, default=4)


    #_shift pretrain
    parser.add_argument('--WIN_OFFSET_RANGE', type=int, nargs='+', default=[100,350])
    parser.add_argument('--IMG_OFFSET_RANGE', type=int, nargs='+', default=[-100,100])


    parser.add_argument('--DEN_FACTOR', type=float, default=200.)
    parser.add_argument('--MEAN_STD', type=tuple, default=([117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]))

    cfg = parser.parse_args()
    

    now = time.strftime("%m-%d_%H-%M", time.localtime())

    cfg.EXP_NAME = now \
    + '_' + cfg.EXP_NAME\
    + '_' + cfg.DATASET \
    + '_' + str(cfg.LR_Base)

    cfg.EXP_PATH = os.path.join('../exp', cfg.DATASET, cfg.task)  # the path of logs, checkpoints, and current codes
    
    cfg.mode = 'train'

    if cfg.DATASET == "SENSE":
        cfg.MEAN_STD = ([117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.])
        cfg.TRAIN_FRAME_INTERVALS = [5,12]

    cfg.VAL_INTERVALS = (cfg.TRAIN_FRAME_INTERVALS[0]+cfg.TRAIN_FRAME_INTERVALS[1])//2

    if not os.path.exists(cfg.EXP_PATH ):
        os.makedirs(cfg.EXP_PATH )

    if cfg.FROZEN:
        assert cfg.PRETRAIN == True
    
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