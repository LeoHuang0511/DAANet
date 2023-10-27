import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.VGG.conv import BasicConv
from .dcn import DeformableConv2d
from .VGG.conv import BasicConv, ResBlock
from .attention import  CrossAttention, LocalCrossAttention

# import pytorch_ssim


class multi_scale_deformable_module(nn.Module):
    def __init__(self, cfg):
        super(multi_scale_deformable_module, self).__init__()
        offset_groups = 4
        deformable_kernel_size = 3
        padding = (deformable_kernel_size - 1) // 2
        self.deformable_conv1 = DeformableConv2d(256, 256, offset_groups, kernel_size=deformable_kernel_size, padding = padding)
        self.deformable_conv2 = DeformableConv2d(256, 256, offset_groups, kernel_size=deformable_kernel_size, padding = padding)
        # self.deformable_conv3 = DeformableConv2d(256, 256, offset_groups, kernel_size=deformable_kernel_size, padding = padding)
        
        # self.residual_block1 = ResBlock(512,256)
        # self.residual_block2 = ResBlock(512,256)

    def forward(self,ref_1, ref_2, source): #b c h w
        # print(ref_1.shape)
        # print(ref_2.shape)
        # print(ref_3.shape)


#       #scale 1
        # ref_1_R, _ = self.deformable_conv1(ref_1, ref_2) #256 
        # ref_2_R, _ = self.deformable_conv1(ref_2, source) #256 

        # ref_1_R, _ = self.deformable_conv2(ref_1_R, ref_2_R) #256 

        ref_2_R, _ = self.deformable_conv1(ref_2, source) #256 
    
    

        

#         ref_1_RR, offset_ref1_2_sou = self.deformable_conv2(ref_1, source) #256 
#         ref_1_RR = self.residual_block2(torch.cat([ref_1_RR,ref_1], dim=1))
        ref_1_R, offset_ref1_2_sou = self.deformable_conv2(ref_1, source) #256 

        
        

        

#         offset_ref1_2_ref2 = F.interpolate(offset_ref1_2_ref2,scale_factor=4,mode='bilinear',align_corners=True) * 4 
#         offset_ref1R_2_ref2 = F.interpolate(offset_ref1R_2_ref2,scale_factor=4,mode='bilinear',align_corners=True) * 4 
        offset_ref1_2_sou = F.interpolate(offset_ref1_2_sou,scale_factor=4,mode='bilinear',align_corners=True) * 4 


       
#         z = torch.concat([ref_1_R_1, ref_1_RR, source], axis = 1) #768  to compute outflow
        z = torch.concat([ref_2_R, ref_1_R, source], axis = 1) #768  to compute outflow


     
        return z, offset_ref1_2_sou


class MTFA(nn.Module):
    def __init__(self, cfg):
        super(MTFA, self).__init__()
        self.multi_scale_deformable_layer = multi_scale_deformable_module(cfg)

        self.reduce_channel = nn.Sequential(
            ResBlock(768, 768),
            ResBlock(768, 512),
            ResBlock(512, 256),
            ResBlock(256, 128)
        )
    
    def forward(self, f, ft, f2t):
        z_f2t, offset_022 = self.multi_scale_deformable_layer(f, ft, f2t)
        z_f,  offset_220 = self.multi_scale_deformable_layer(f2t, ft, f)
        compare = torch.concat([z_f, z_f2t], axis = 0)  #first half compute out, second half compute in
        compare = self.reduce_channel(compare)

        return compare, offset_220, offset_022




