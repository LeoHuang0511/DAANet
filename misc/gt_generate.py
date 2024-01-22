from typing import Any
import torch
from misc.layer import Gaussianlayer
import numpy as np
from model.points_from_den import get_ROI_and_MatchInfo
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
class GenerateGT():
    def __init__(self, cfg):
        self.cfg = cfg

        self.sigma = cfg.gaussian_sigma

        self.Gaussian = Gaussianlayer(sigma=[self.sigma]).cuda()


    
    def get_den(self, shape, target, ratio, scale_num):
        
        
        dot_map = self.get_dot(shape, target, ratio)

        gt_den = self.Gaussian(dot_map)
        assert shape == gt_den.shape

        gt_den_scales = []
        gt_den_scales.append(gt_den)

        for scale in range(1,scale_num):
            for i in range(gt_den.shape[0]):
                # multi-scale density gt
                gt_den_np = gt_den[i].detach().cpu().numpy().squeeze().copy()
                den = cv2.resize(gt_den_np,(int(gt_den_np.shape[1]/(2**scale)),int(gt_den_np.shape[0]/(2**scale))),interpolation = cv2.INTER_CUBIC)* ((2**scale)**2)
                
                
                if i == 0:
                    dengt = np.expand_dims(den,0)

                else:
                    dengt = np.vstack((dengt,np.expand_dims(den,0)))
            gt_den_scales.append(torch.Tensor(dengt[:,None,:,:]).cuda())

        return gt_den_scales
    
    def get_dot(self, shape, target, ratio):
        
        device = target[0]['points'].device
        dot_map = torch.zeros(shape).to(device)
        
        for i, data in enumerate(target):
            
            points = (data['points']* ratio).long()
            dot_map[i, 0, points[:, 1], points[:, 0]] = 1

        assert shape == dot_map.shape

        return dot_map
    def get_scale_io_masks(self, gt_mask, scale_num):
        b, c, h, w = gt_mask.shape
        # mask
        gt_mask_scales = []
        # print((gt_mask>0).Float().cuda())

        # gt_mask = gt_mask.view(b*2, h, w)
        gt_mask = gt_mask.view(b*c, h, w)

        gt_mask_scales.append(gt_mask>0)


        # for scale in range(0,scale_num):
        #     # for i in range(b*2):
        #     for i in range(b*c):

        #         # multi-scale density gt
        #         gt_mask_np = gt_mask[i].detach().cpu().numpy().squeeze().copy()
        #         if scale == 0:
        #             mask = gt_mask_np
        #         else:
        #             mask = cv2.resize(gt_mask_np,(int(gt_mask_np.shape[1]/(2**scale)),int(gt_mask_np.shape[0]/(2**scale))),interpolation = cv2.INTER_CUBIC)* ((2**scale)**2)
                
        #         if i == 0:
        #             maskgt = np.expand_dims(mask,0)

        #         else:
        #             maskgt = np.vstack((maskgt,np.expand_dims(mask,0)))
            
        #     maskgt = torch.Tensor(maskgt>0).view(b,c,h//(2**scale),w//(2**scale)).cuda()
        #     # maskgt[:,2,:,:] *= 2.

        #     seggt = torch.zeros((maskgt.shape[0],2,maskgt.shape[2],maskgt.shape[3])).cuda()
        #     seggt[:,0,:,:][maskgt[:,2,:,:].bool()] = 2
        #     seggt[:,0,:,:][maskgt[:,0,:,:].bool()] = 1 
            
        #     seggt[:,1,:,:][maskgt[:,3,:,:].bool()] = 2
        #     seggt[:,1,:,:][maskgt[:,1,:,:].bool()] = 1
            

            



        #     # gt_mask_scales.append(maskgt)
        #     gt_mask_scales.append(seggt.long())



        return gt_mask_scales
    
    

    def get_pair_io_map(self, pair_idx, target, match_gt, gt_mask, gt_outflow_cnt, gt_inflow_cnt,ratio):

        device = target[0]['points'].device
        gt_mask = gt_mask.clone()
        gt_outflow_cnt = gt_outflow_cnt.clone()
        gt_inflow_cnt = gt_inflow_cnt.clone()
        mask_out = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)
        mask_in = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)
        out_ind = match_gt['un_a']  #inflow people id
        in_ind = match_gt['un_b']   #outflow people id

        if len(out_ind) > 0:
            gt_outflow_cnt[pair_idx] += len(out_ind)
            # mask_out[0, 0, target[pair_idx * 2]['points'][out_ind, 1].long(), target[pair_idx * 2]['points'][out_ind, 0].long()] = 1
            mask_out[0, 0, (target[pair_idx * 2]['points'][out_ind, 1] * ratio).long(), (target[pair_idx * 2]['points'][out_ind, 0] * ratio).long()] = 1

        if len(in_ind) > 0:
            gt_inflow_cnt[pair_idx] += len(in_ind)
            # mask_in[0, 0, target[pair_idx * 2+1]['points'][in_ind, 1].long(), target[pair_idx * 2+1]['points'][in_ind, 0].long()] = 1  
            mask_in[0, 0, (target[pair_idx * 2+1]['points'][in_ind, 1] * ratio).long(), (target[pair_idx * 2+1]['points'][in_ind, 0] * ratio).long()] = 1  

        # mask_out = self.generate_mask(mask_out)
        # mask_in = self.generate_mask(mask_in)
        mask_out = self.Gaussian(mask_out)
        mask_in = self.Gaussian(mask_in)


        gt_mask[pair_idx,0,:,:] = mask_out
        gt_mask[pair_idx,1,:,:] = mask_in
        
        
        return gt_mask, gt_inflow_cnt, gt_outflow_cnt
    
    # def get_pair_seg_map(self, pair_idx, target, match_gt, gt_mask, gt_outflow_cnt, gt_inflow_cnt,ratio):

    #     device = target[0]['points'].device
    #     gt_mask = gt_mask.clone()
    #     gt_outflow_cnt = gt_outflow_cnt.clone()
    #     gt_inflow_cnt = gt_inflow_cnt.clone()
    #     mask_out = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)
    #     mask_in = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)
    #     mask_match_1 = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)
    #     mask_match_2 = torch.zeros(1, 1, gt_mask.size(2), gt_mask.size(3)).to(device)


    #     out_ind = match_gt['un_a']  #inflow people id
    #     in_ind = match_gt['un_b']   #outflow people id
    #     match_ind = match_gt['a2b'] #match people id



    #     if len(match_ind) > 0:
    #         # gt_outflow_cnt[pair_idx] += len(out_ind)
            
    #         # mask_out[0, 0, target[pair_idx * 2]['points'][out_ind, 1].long(), target[pair_idx * 2]['points'][out_ind, 0].long()] = 1
    #         mask_match_1[0, 0, (target[pair_idx * 2]['points'][match_ind[:,0]  , 1] * ratio).long(), (target[pair_idx * 2]['points'][match_ind[:,0]  , 0] * ratio).long()] = 1
    #         mask_match_2[0, 0, (target[pair_idx * 2+1]['points'][match_ind[:,1]  , 1] * ratio).long(), (target[pair_idx * 2+1]['points'][match_ind[:,1]  , 0] * ratio).long()] = 1



    #     if len(out_ind) > 0:
    #         gt_outflow_cnt[pair_idx] += len(out_ind)
    #         # mask_out[0, 0, target[pair_idx * 2]['points'][out_ind, 1].long(), target[pair_idx * 2]['points'][out_ind, 0].long()] = 1
    #         mask_out[0, 0, (target[pair_idx * 2]['points'][out_ind, 1] * ratio).long(), (target[pair_idx * 2]['points'][out_ind, 0] * ratio).long()] = 1

    #     if len(in_ind) > 0:
    #         gt_inflow_cnt[pair_idx] += len(in_ind)
    #         # mask_in[0, 0, target[pair_idx * 2+1]['points'][in_ind, 1].long(), target[pair_idx * 2+1]['points'][in_ind, 0].long()] = 1  
    #         mask_in[0, 0, (target[pair_idx * 2+1]['points'][in_ind, 1] * ratio).long(), (target[pair_idx * 2+1]['points'][in_ind, 0] * ratio).long()] = 1  
    #     # mask_out = self.generate_mask(mask_out)
    #     # mask_in = self.generate_mask(mask_in)
    #     mask_out = self.Gaussian(mask_out)
    #     mask_in = self.Gaussian(mask_in) 
    #     mask_match_1 = self.Gaussian(mask_match_1) 
    #     mask_match_2 = self.Gaussian(mask_match_2) 




    #     gt_mask[pair_idx,0,:,:] = mask_out
    #     gt_mask[pair_idx,1,:,:] = mask_in
    #     gt_mask[pair_idx,2,:,:] = mask_match_1
    #     gt_mask[pair_idx,3,:,:] = mask_match_2



        
        
    #     return gt_mask, gt_inflow_cnt, gt_outflow_cnt
    

    def get_confidence(self, masks, gt_masks):
        ce_scales = []
        img_pair_num = masks[0].shape[0] // 2
        device = masks[0].device

        for scale in range(len(masks)):

            ce = torch.zeros((masks[scale].shape[0],1,masks[scale].shape[2],masks[scale].shape[3]))
            ce[:img_pair_num] = -F.cross_entropy(masks[scale][:img_pair_num], gt_masks[scale][:,0,:,:],reduction = "none").unsqueeze(1)
            ce[img_pair_num:] = -F.cross_entropy(masks[scale][img_pair_num:], gt_masks[scale][:,1,:,:],reduction = "none").unsqueeze(1)
            ce = F.upsample_nearest(ce, scale_factor=2**scale)
            ce = F.adaptive_avg_pool2d(ce, output_size=(int(self.cfg.TRAIN_SIZE[0] * self.cfg.feature_scale), int(self.cfg.TRAIN_SIZE[1] * self.cfg.feature_scale)))


            ce_scales.append(ce)
        ce_scales = torch.cat(ce_scales, dim=1) #(b,3,h,w)
        gt_confidence = (torch.ones_like(ce_scales)*-1).to(device)
        for scale in range(gt_confidence.shape[1]):
            gt_confidence[:,scale][torch.where(torch.argmax(ce_scales,dim=1).squeeze()==scale)] = 1
            gt_confidence[:,scale][torch.where(torch.argmin(ce_scales,dim=1).squeeze()==scale)] = 0



        return gt_confidence





                
