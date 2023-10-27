# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from model.VGG.conv import BasicConv
# from .dcn import DeformableConv2d, MultiSscaleDCN
# from .VGG.conv import BasicConv, ResBlock

# # import pytorch_ssim


# class optical_deformable_alignment_module(nn.Module):
#     def __init__(self, cfg):
#         super(optical_deformable_alignment_module, self).__init__()
#         offset_groups = 4
#         deformable_kernel_size = 3
#         padding = (deformable_kernel_size - 1) // 2
#         # self.deformable_conv1 = MultiSscaleDCN(256, 256, offset_groups, kernel_size=deformable_kernel_size, padding = padding)
        
#         self.deformable_conv1 = DeformableConv2d(256, 256, offset_groups, kernel_size=deformable_kernel_size, padding = padding)
#         # self.deformable_conv1 = DeformableConv2d(128, 128, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)



       
#         self.reduce_channel = nn.Sequential(
#             ResBlock(512, 512),
#             ResBlock(512, 256),
#             ResBlock(256, 128)
#         )
# #       
#     def forward(self,reference, source): #b c h w
# #         pre_warp_ref = optical_flow_warping(reference, -flow)
#         batch = source.size(dim = 0)
#         ref_refined_feature,offset2sou = self.deformable_conv1(reference, source) #256 
#         sour_refined_feature,offset2ref = self.deformable_conv1(source, reference)
#         offset2sou = F.interpolate(offset2sou,scale_factor=4,mode='bilinear',align_corners=True) * 4 
#         offset2ref = F.interpolate(offset2ref,scale_factor=4,mode='bilinear',align_corners=True) * 4 

#         # self.offset_loss = self.deformable_conv1.offset_loss
# #         img1 = img[0::2,:,:,:]
# #         img2 = img[1::2,:,:,:]
# #         print("opti:",ref_refined_feature.shape)
# #         print("opti:",sour_refined_feature.shape)
# #         self.img1_warp = optical_flow_warping(img1, offset2s)
# #         self.img2_warp = optical_flow_warping(img2, offset2r)  #offset2r is forward_flow

# #         self.ssim_loss = 2 - self.ssim(img1,self.img2_warp) - self.ssim(img2,self.img1_warp) + F.l1_loss(img1,self.img2_warp) +F.l1_loss(img2,self.img1_warp)

#         refcorsou = torch.concat([sour_refined_feature, reference], axis = 1) #512  to compute outflow
#         soucorref = torch.concat([ref_refined_feature, source], axis = 1)    # compute inflow
#         # refcorsou = torch.concat([sour_refined_feature-ref_refined_feature, reference], axis = 1) #512  to compute outflow
#         # soucorref = torch.concat([ref_refined_feature-sour_refined_feature, source], axis = 1)    # compute inflow

#         # attn_ref, f_ref = self.attn(sour_refined_feature,reference)
#         # attn_ref, refcorsou = self.attn(reference,sour_refined_feature)
# # 
#         # refcorsou = torch.concat([sour_refined_feature, f_ref,reference], axis=1)
#         # refcorsou = torch.concat([f_ref, f_ref], axis=1)

#         # attn_ref = torch.sum(attn_ref, dim=(-1,-2))

#         # attn_sou, f_sou = self.attn(ref_refined_feature,source)
#         # attn_sou, soucorref = self.attn(source,ref_refined_feature)

#         # soucorref = torch.concat([ref_refined_feature, f_sou,source], axis=1)
#         # soucorref = torch.concat([f_sou, f_sou], axis=1)

#         # attn_sou = torch.sum(attn_sou, dim=(-1,-2))



        
#         compare = torch.concat([refcorsou, soucorref], axis = 0)  #first half compute out, second half compute in
#         # compare = torch.concat([f_ref, f_sou], axis = 0)  #first half compute out, second half compute in

#         compare = self.reduce_channel(compare)
        

#         # return compare,offset2r, offset2s, attn_ref, attn_sou
#         return compare, offset2ref, offset2sou

