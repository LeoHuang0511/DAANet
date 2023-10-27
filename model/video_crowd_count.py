# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from .VGG.VGG16_FPN import VGG16_FPN
# # from .optical_deformable_module import optical_deformable_alignment_module
# from .VGG.conv import BasicDeconv, ResBlock
# # from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
# from model.points_from_den import *
# BN_MOMENTUM = 0.01

# # +
# class video_crowd_count(nn.Module):
#     def __init__(self,cfg, cfg_data):
#         super(video_crowd_count, self).__init__()
#         self.cfg = cfg
#         # self.device = torch.device(f'cuda:{cfg.GPU_ID}')
#         # self.device = torch.device("cuda:"+torch.cuda.current_device())


#         self.Extractor = VGG16_FPN().cuda()
#         if cfg.TemporalScale == 2:
#             self.optical_defromable_layer = optical_deformable_alignment_module(cfg).cuda()
#         elif cfg.TemporalScale == 3:
#             self.optical_defromable_layer = MTFA(cfg).cuda()

               
#         # self.feature_proj = nn.Conv2d(256, 256, kernel_size=1).cuda()

#         self.mask_predict_layer = nn.Sequential(
#             nn.Dropout2d(0.2),
#             ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
#             ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),

#             nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
#             nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
# #             nn.ReLU(inplace=True),

#             nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8, momentum=BN_MOMENTUM),
# #             nn.ReLU(inplace=True),

#             nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
#             nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
# #             nn.ReLU(inplace=True),

#             nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
# #             nn.ReLU(inplace=True)
#             ).cuda()

        
#         self.cfg = cfg
       
    
#     def forward(self, img): #if frame1 size is [B, C, H, W]
        
        

#         img_pair_num = img.size(0)// self.cfg.TemporalScale
#         feature, den = self.Extractor(img)
#         den = den / self.cfg.DEN_FACTOR

#         if self.cfg.TemporalScale == 2:
#             compare, flow , back_flow = self.optical_defromable_layer(feature[0::self.cfg.TemporalScale,:,:,:], feature[1::self.cfg.TemporalScale,:,:,:])
#         elif  self.cfg.TemporalScale == 3:
#             compare, flow , back_flow = self.optical_defromable_layer(feature[0::self.cfg.TemporalScale,:,:,:], feature[1::self.cfg.TemporalScale,:,:,:],feature[2::self.cfg.TemporalScale,:,:,:])

        
#         # compare, flow , back_flow, attn_ref, attn_sou = self.optical_defromable_layer(feature[0::2,:,:,:], feature[1::2,:,:,:])
        
#         mask = self.mask_predict_layer(compare)
#         mask = torch.sigmoid(mask)

#         # pre_outflow_map =(mask[:img_pair_num,:,:,:] >= self.cfg.mask_threshold)* den[0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
#         # pre_inflow_map = (mask[img_pair_num:,:,:,:] >= self.cfg.mask_threshold) * den[1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)
#         pre_outflow_map =(mask[:img_pair_num,:,:,:])* den[0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
#         pre_inflow_map = (mask[img_pair_num:,:,:,:] ) * den[1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)





#         # feature = self.feature_proj(feature)


        
#         return  den, mask, pre_outflow_map, pre_inflow_map, flow, back_flow, feature

#         # return  den, mask, pre_outflow_map, pre_inflow_map, flow, back_flow, attn_ref, attn_sou, feature

