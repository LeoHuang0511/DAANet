import torch
import torch.nn.functional as F
import torch.nn as nn
from misc.KPI_pool import Task_KPI_Pool
from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
from misc.gt_generate import GenerateGT
import numpy as np
import PIL.Image as Image
import cv2
from misc.utils import *





# +
class ComputeKPILoss(object):
    def __init__(self, trainer, cfg, scale_num=3) -> None:

        self.cfg = cfg
        self.trainer = trainer
        
        self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'pre_cnt'], 'mask': ['gt_cnt', 'acc_cnt']}, maximum_sample=1000)
        
        self.DEN_FACTOR = cfg.DEN_FACTOR
        self.gt_generater = GenerateGT(cfg)
        
        self.den_scale_weight = [2, 0.1,0.01]
            
        self.dynamic_weight = []
        
        


        
        

    def __call__(self, den , den_scales, gt_den_scales, masks, gt_mask, pre_outflow_map, pre_inflow_map, gt_io_map, \
                 pre_inf_cnt, pre_out_cnt, gt_in_cnt, gt_out_cnt, confidence):
        img_pair_num = den_scales[0].shape[0]//2

        assert den.shape == gt_den_scales[0].shape
        self.cnt_loss = F.mse_loss(den*self.DEN_FACTOR, gt_den_scales[0] * self.DEN_FACTOR)
        self.cnt_loss_scales = torch.zeros(len(den_scales)).cuda()
        # self.mask_loss_scales = torch.zeros(len(den_scales)).cuda()
        # self.out_loss_scales = torch.zeros(len(den_scales)).cuda()
        # self.in_loss_scales = torch.zeros(len(den_scales)).cuda()
        


        for scale in range(len(den_scales)):
             # # # counting MSE loss
            assert den_scales[scale].shape == gt_den_scales[scale].shape
            # print(f"{scale} ",den_scales[scale].shape)

            self.cnt_loss_scales[scale] += F.mse_loss(den_scales[scale]*self.DEN_FACTOR, gt_den_scales[scale] * self.DEN_FACTOR) * self.den_scale_weight[scale]
            # weight = F.adaptive_avg_pool2d(confidence[:,scale,:,:].unsqueeze(1), den_scales[scale].shape[2:])
            # weighted_mse = torch.mean(weight * (den_scales[scale]*self.DEN_FACTOR - gt_den_scales[scale] * self.DEN_FACTOR)**2)
           


            # self.cnt_loss_scales[scale] += 2 * weighted_mse * self.den_scale_weight[scale]
            
             # # # mask loss
        
            
        

      
        self.mask_loss_scales = F.binary_cross_entropy(masks[:img_pair_num], gt_mask[:,0:1,:,:],reduction = "mean") + \
                               F.binary_cross_entropy(masks[img_pair_num:], gt_mask[:,1:2,:,:],reduction = "mean")
        # # # inflow/outflow loss
        

        assert (pre_outflow_map.shape == gt_io_map[:,0:1,:,:].shape) and (pre_inflow_map.shape == gt_io_map[:,1:2,:,:].shape)
        self.out_loss = F.mse_loss(pre_outflow_map,  gt_io_map[:,0:1,:,:],reduction = 'sum') / self.cfg.TRAIN_BATCH_SIZE
        self.in_loss = F.mse_loss(pre_inflow_map, gt_io_map[:,1:2,:,:], reduction='sum') / self.cfg.TRAIN_BATCH_SIZE
        




        # # # overall loss
        gt_cnt = gt_den_scales[0].sum()
        pre_cnt = den.sum()
        self.task_KPI.add({'den': {'gt_cnt': gt_cnt, 'pre_cnt': max(0,gt_cnt - (pre_cnt - gt_cnt).abs()) },
                               'mask': {'gt_cnt' : gt_out_cnt.sum()+gt_in_cnt.sum(), 'acc_cnt': \
                                        max(0,gt_out_cnt.sum()+gt_in_cnt.sum() - (pre_inf_cnt - gt_in_cnt).abs().sum() \
                                            - (pre_out_cnt - gt_out_cnt).abs().sum()) }})
        self.KPI = self.task_KPI.query()

        loss = torch.stack([self.cnt_loss  , self.out_loss + self.in_loss + self.mask_loss_scales])
#                 loss = torch.stack([counting_mse_loss ,  out_loss+in_loss ])
        weight = torch.stack([self.KPI['den'],self.KPI['mask']]).to(loss.device)
        weight = -(1-weight) * torch.log(weight+1e-8)
        self.weight = weight/weight.sum()

        all_loss = self.weight*loss
        self.cnt_loss = all_loss[0]

        scale_loss = self.cnt_loss_scales.sum()

        
#         if self.trainer.i_tb == self.cfg.Dynamic_freq:
#             self.init_scale_loss = scale_loss.item()
            
# #         if self.trainer.i_tb % self.cfg.Dynamic_freq == 0:
#         if (self.trainer.i_tb >= self.cfg.Dynamic_freq) and (self.trainer.i_tb % self.cfg.Dynamic_freq == 0):
#             self.dynamic_weight = (self.init_scale_loss - scale_loss.item())/ (self.init_scale_loss+1e-16)
        
        if self.trainer.i_tb == 1:
            self.init_scale_loss = scale_loss.item()
        self.dynamic_weight.append((self.init_scale_loss - scale_loss.item())/ (self.init_scale_loss+1e-16))
        if self.trainer.i_tb > 1000:
            self.dynamic_weight.pop(0)
            assert len(self.dynamic_weight) == 1000
            
        avg_dynamic_weight = sum(self.dynamic_weight) / len(self.dynamic_weight)
#         loss = scale_loss + self.mask_loss_scales.sum() + self.dynamic_weight * (self.cnt_loss + (self.in_loss + self.out_loss))
        # loss = scale_loss + self.mask_loss_scales.sum() + self.out_loss_scales.sum() + self.in_loss_scales.sum() + \
        #     avg_dynamic_weight * (self.cnt_loss + (self.in_loss + self.out_loss))
        loss = scale_loss  + avg_dynamic_weight * (self.cnt_loss*10) + all_loss[1]
        
        return loss


    def compute_con_loss(self, pair_idx, feature1, feature2, match_gt, pois, count_in_pair, feature_scale):
        
        
        mdesc0, mdesc1 = self.get_head_feature(pair_idx, feature1, feature2, pois, count_in_pair, feature_scale)
        con_inter_loss =   self.contrastive_loss(mdesc0, mdesc1, match_gt['a2b'][:,0], match_gt['a2b'][:,1])

        return con_inter_loss.sum()

    



    
    def get_head_feature(self, pair_idx, feature1, feature2, pois, count_in_pair, feature_scale):
        feature = torch.cat([feature1[pair_idx:pair_idx+1], feature2[pair_idx:pair_idx+1]], dim=0)
        # poi_features = prroi_pool2d(feature[pair_idx*2:pair_idx*2+2], pois, 1, 1, feature_scale)
        poi_features = prroi_pool2d(feature, pois, 1, 1, feature_scale)

        poi_features=  poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
        mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)

        return mdesc0, mdesc1
    
    def contrastive_loss(self, mdesc0, mdesc1, idx0, idx1, intra=False):
        sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)  #inner product (n,m)
    #                     #mdesc0(n,256) mdesc1(m,256) frame1 n peoples frames 2 m peoples
    #                     #contrasitve loss更改
        m0 = torch.norm(mdesc0,dim = 1) #l2norm
        m1 = torch.norm(mdesc1,dim = 1)
        norm = torch.einsum('bn,bm->bnm',m0,m1) + 1e-7 # (n,m)
        exp_term = torch.exp(sim_matrix / (256 ** .5 )/ norm)[0]
        if intra == False:
            try:
                topk = torch.topk(exp_term[idx0],50,dim = 1).values #(c,b) #取幾個negative 記得加回positive
            except:
                topk = exp_term[idx0]
            # topk = exp_term[idx0]

            denominator = torch.sum(topk,dim=1)   #分母 denominator
            numerator = exp_term[idx0, idx1]   #分子 numerator  c個重複  c個loss

            # loss =  torch.sum(-torch.log(numerator / denominator +1e-7))
            loss =  torch.sum(-torch.log(numerator / denominator +1e-7)) 
        else:
            numerator = exp_term[idx0, idx1]   #分子 numerator  c個重複  c個loss

            # loss =  torch.sum(-torch.log(numerator / denominator +1e-7))
            loss =  torch.sum(-torch.log(numerator)) 


        return loss


    # def compute_io_loss(self, masks, dens, gt_masks, gt_dens):
    # def compute_io_loss(self,pre_outflow_map, pre_inflow_map, gt_masks, gt_dens):

    
    #     img_pair_num = gt_dens[0].shape[0]//2
        
        
    #     gt_outflow_map = (gt_mask[:,0:1,:,:] == 1) * gt_dens[0][0::2,:,:,:] 
    #     gt_inflow_map = (gt_mask[:,1:2,:,:] == 1) * gt_dens[0][1::2,:,:,:] 


    #     assert (pre_outflow_map.shape == gt_outflow_map.shape) and (pre_inflow_map.shape == gt_inflow_map.shape)
       
        
        
    #     out_loss = F.mse_loss(pre_outflow_map, gt_outflow_map,reduction = 'sum')/self.cfg.TRAIN_BATCH_SIZE
    #     in_loss = F.mse_loss(pre_inflow_map, gt_inflow_map, reduction='sum')/self.cfg.TRAIN_BATCH_SIZE




    #     return in_loss, out_loss
# -




class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        # probs = torch.sigmoid(logits)
        probs = logits

        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.double())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

