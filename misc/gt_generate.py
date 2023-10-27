from typing import Any
import torch
from misc.layer import Gaussianlayer
import numpy as np
from model.points_from_den import get_ROI_and_MatchInfo
import math

class GenerateGT():
    def __init__(self, cfg):
        self.cfg = cfg

        self.sigma = cfg.gaussian_sigma

        self.Gaussian = Gaussianlayer(sigma=[self.sigma]).cuda()


    
    def get_den(self, shape, target, ratio):
        
        
        dot_map = self.get_dot(shape, target, ratio)

        gt_den = self.Gaussian(dot_map)
        assert shape == gt_den.shape

        return gt_den
    
    def get_dot(self, shape, target, ratio):
        
        device = target[0]['points'].device
        dot_map = torch.zeros(shape).to(device)
        
        for i, data in enumerate(target):
            
            points = (data['points']* ratio).long()
            dot_map[i, 0, points[:, 1], points[:, 0]] = 1

        assert shape == dot_map.shape

        return dot_map

    def get_io_mask(self, pair_idx, target, match_gt, gt_mask, gt_outflow_cnt, gt_inflow_cnt,ratio):

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

    def generate_mask(self, dot_map):
        mask = self.Gaussian(dot_map)# > 0
    

        return mask


                
                
