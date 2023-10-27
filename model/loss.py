import torch
import torch.nn.functional as F
import torch.nn as nn
from misc.KPI_pool import Task_KPI_Pool
from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
from misc.gt_generate import GenerateGT




class ComputeKPILoss(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg
        self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'pre_cnt'], 'mask': ['gt_cnt', 'acc_cnt']}, maximum_sample=1000)
        self.DEN_FACTOR = cfg.DEN_FACTOR
        self.gt_generater = GenerateGT(cfg)
        
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2)
        
        

    def __call__(self, den, gt_den, den_scales, gt_den_scales, mask, gt_mask, pre_inf_cnt, pre_out_cnt, gt_in_cnt, gt_out_cnt):
        img_pair_num = den.shape[0]//2
        # # # counting MSE loss

        assert den.shape == gt_den.shape
        self.cnt_loss = F.mse_loss(den*self.DEN_FACTOR, gt_den * self.DEN_FACTOR)

        self.cnt_loss_scales = 0
        scale_weight = [1, 0.5,0.05]
        for i in range(len(den_scales)):
            
            assert den_scales[i].shape == gt_den_scales[i].shape
            self.cnt_loss_scales += F.mse_loss(den_scales[i]*self.DEN_FACTOR, gt_den_scales[i] * self.DEN_FACTOR) * scale_weight[i]
            # print(F.mse_loss(den_scales[i]*self.DEN_FACTOR, gt_den_scales[i] * self.DEN_FACTOR) )
        # # # mask loss
        self.mask_loss = 0
        for i in range(len(mask)):
            assert (mask[i][:img_pair_num].shape == gt_mask[i][:,0:1,:,:].shape)and (mask[i][img_pair_num:].shape == gt_mask[i][:,1:2,:,:].shape)
            self.mask_loss += F.binary_cross_entropy(mask[i][:img_pair_num], gt_mask[i][:,0:1,:,:],reduction = "mean")+ \
                                F.binary_cross_entropy(mask[i][img_pair_num:], gt_mask[i][:,1:2,:,:],reduction = "mean")
            # print(F.binary_cross_entropy(mask[i][:img_pair_num], gt_mask[i][:,0:1,:,:],reduction = "mean")+ \
            #                     F.binary_cross_entropy(mask[i][img_pair_num:], gt_mask[i][:,1:2,:,:],reduction = "mean"))
            # self.mask_loss = self.focal_loss(mask[:img_pair_num], gt_mask[:,0:1,:,:]) + \
            #                     self.focal_loss(mask[img_pair_num:], gt_mask[:,1:2,:,:])
        
        
        # # # inflow/outflow loss
        # self.in_loss, self.out_loss = self.compute_io_loss(mask, den, gt_mask, gt_den)
        self.in_loss, self.out_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()


        # # # overall loss
        img_pair_num = den.shape[0]//2
        pre_cnt = den.sum()
        
        gt_cnt = gt_den.sum()



        self.task_KPI.add({'den': {'gt_cnt': gt_cnt, 'pre_cnt': max(0,gt_cnt - (pre_cnt - gt_cnt).abs()) },
                            'mask': {'gt_cnt' : gt_out_cnt.sum()+gt_in_cnt.sum(), 'acc_cnt': \
                                            max(0,gt_out_cnt.sum()+gt_in_cnt.sum() - (pre_inf_cnt - gt_in_cnt).abs().sum() \
                                                - (pre_out_cnt - gt_out_cnt).abs().sum()) }})
        self.KPI = self.task_KPI.query()

        # loss = torch.stack([1*self.cnt_loss   , 0*self.out_loss + 0*self.in_loss+ 1*self.mask_loss])
        loss = torch.stack([0*self.cnt_loss   ,0*self.mask_loss])

        weight = torch.stack([self.KPI['den'],self.KPI['mask']]).to(loss.device)
        weight = -(1-weight) * torch.log(weight+1e-8)
        weight = weight/weight.sum()

        # return weight*loss+ 0.25*self.cnt_loss_scales
        return 0.3*self.mask_loss + 0.3*self.cnt_loss_scales


    def compute_con_loss(self, pair_idx, feature1, feature2, match_gt, pois, count_in_pair, feature_scale):
        
        inter_loss = 0 # inter two frame pairs with same scale
        intra_loss = 0 # intra same frame with different scales

        head_f0 = []
        head_f1 = []

        for scale in range(len(feature1)):

                            
            mdesc0, mdesc1 = self.get_head_feature(pair_idx, feature1[scale], feature2[scale], pois, count_in_pair, feature_scale / (scale+1))

            inter_loss =  inter_loss + \
                self.contrastive_loss(mdesc0, mdesc1, match_gt['a2b'][:,0], match_gt['a2b'][:,1]) * self.cfg.con_scale**(-scale)
            head_f0.append(mdesc0)
            head_f1.append(mdesc1)
        # mdesc0, mdesc1 = self.get_head_feature(pair_idx, feature1, feature2, pois, count_in_pair, feature_scale)

        # inter_loss =  inter_loss + \
        #     self.contrastive_loss(mdesc0, mdesc1, match_gt['a2b'][:,0], match_gt['a2b'][:,1]) 
        
        
        if self.cfg.intra_loss:
            for scale in range(len(feature1)):
                if (scale+1) < len(feature1):
                    intra0 = self.contrastive_loss(head_f0[scale], head_f0[scale+1], match_gt['a2b'][:,0], match_gt['a2b'][:,0],True)
                    intra1 = self.contrastive_loss(head_f1[scale], head_f1[scale+1], match_gt['a2b'][:,1], match_gt['a2b'][:,1],True)
                elif (scale+1) == len(feature1):
                    intra0 = self.contrastive_loss(head_f0[scale], head_f0[0], match_gt['a2b'][:,0], match_gt['a2b'][:,0],True)
                    intra1 = self.contrastive_loss(head_f1[scale], head_f1[0], match_gt['a2b'][:,1], match_gt['a2b'][:,1],True)

                intra_loss = intra_loss + intra0 + intra1
        
        return inter_loss + intra_loss * self.cfg.intra_loss_alpha



    
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


    def compute_io_loss(self, mask, den, gt_mask, gt_den):
        img_pair_num = gt_den.shape[0]//2
        pre_outflow_map = mask[:img_pair_num,:,:,:]* den[0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
        pre_inflow_map = mask[img_pair_num:,:,:,:] * den[1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)
        
        gt_outflow_map = gt_mask[:,0:1,:,:] * gt_den[0::2,:,:,:] 
        gt_inflow_map = gt_mask[:,1:2,:,:] * gt_den[1::2,:,:,:] 

        assert (pre_outflow_map.shape == gt_outflow_map.shape) and (pre_inflow_map.shape == gt_inflow_map.shape)
        
        out_loss = F.mse_loss(pre_outflow_map, gt_outflow_map,reduction = 'sum')
        in_loss = F.mse_loss(pre_inflow_map, gt_inflow_map, reduction='sum')
        # out_loss = F.mse_loss(pre_outflow_map, gt_outflow_map)
        # in_loss = F.mse_loss(pre_inflow_map, gt_inflow_map)

        return in_loss, out_loss
    
    def scale_mask_loss(self, mask, gt_mask):
        img_pair_num = mask[0].shape[0]//2
        mask_loss = F.binary_cross_entropy(mask[0][:img_pair_num], gt_mask[0][:,0:1,:,:],reduction = "mean") + \
                    F.binary_cross_entropy(mask[0][img_pair_num:], gt_mask[0][:,1:2,:,:],reduction = "mean") + \
                    F.binary_cross_entropy(mask[1][:img_pair_num], gt_mask[1][:,0:1,:,:],reduction = "mean") + \
                    F.binary_cross_entropy(mask[1][img_pair_num:], gt_mask[1][:,1:2,:,:],reduction = "mean") + \
                    F.binary_cross_entropy(mask[2][:img_pair_num], gt_mask[2][:,0:1,:,:],reduction = "mean") + \
                    F.binary_cross_entropy(mask[2][img_pair_num:], gt_mask[2][:,1:2,:,:],reduction = "mean") 
        return mask_loss

    
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
        