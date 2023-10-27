import torch
import torch.nn as nn
from .VGG.VGG16_FPN import VGG16_FPN
from .attention import MultiScaleFeatureFusion, SpatialWeightLayer
from .MSDA import MultiScaleDeformableAlingment#, VariantRegionAttention
from .dcn import DeformableConv2d
from .VGG.conv import ResBlock
import torch.nn.functional as F

BN_MOMENTUM = 0.01





class SMDCANet(nn.Module):

    def __init__(self, cfg, cfg_data):
        super(SMDCANet, self).__init__()
        self.cfg = cfg

        self.Extractor = VGG16_FPN(cfg)
        num_feat = 128
        
        self.deformable_alignment = SMDCAlignment(cfg, num_feat, scale_num=3)
        
    
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
# # #             nn.ReLU(inplace=True),

#             nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
# # #             nn.ReLU(inplace=True)
#             )

        self.mask_predict_layer = nn.ModuleList()
        for i in range(3):
        
            self.mask_predict_layer.append(nn.Sequential(

            

            nn.Dropout2d(0.2),

            ResBlock(in_dim=256, out_dim=128, dilation=0, norm="bn"),
            ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
            ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),



            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
# #             nn.ReLU(inplace=True),

            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
# #             nn.ReLU(inplace=True)
            ))
        

    
        self.cfg = cfg

#         


    



    def forward(self, img):

        img_pair_num = img.size(0)//2 
        feature, den, den_scales = self.Extractor(img)



        feature1 = []
        feature2 = []


        for fea in feature:
    
            feature1.append(fea[0::2,:,:,:])
            feature2.append(fea[1::2,:,:,:])

        

        
        den = den / self.cfg.DEN_FACTOR
        for i in range(len(den_scales)):
            den_scales[i] = den_scales[i] / self.cfg.DEN_FACTOR

        # mask, flow , back_flow, attn_1, attn_2= self.deformable_alignment(feature1, feature2)
        f_out, f_in, flow , back_flow, attn_1, attn_2= self.deformable_alignment(feature1, feature2)



        mask = []
        for scale in range(len(f_out)):
            f_out[scale] = torch.sigmoid(self.mask_predict_layer[scale](f_out[scale]))
            f_in[scale] = torch.sigmoid(self.mask_predict_layer[scale](f_in[scale]))
            mask.append(torch.cat([f_out[scale],  f_in[scale]],dim=0))

        # mask = self.mask_predict_layer(mask)
        # mask = torch.sigmoid(mask)




        # pre_outflow_map =(mask[:img_pair_num,:,:,:])* den[0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
        # pre_inflow_map = (mask[img_pair_num:,:,:,:]) * den[1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)
        pre_outflow_map =(mask[0][:img_pair_num,:,:,:])* den_scales[0][0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
        pre_inflow_map = (mask[0][img_pair_num:,:,:,:]) * den_scales[0][1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)
       



        return den, den_scales, mask, pre_outflow_map, pre_inflow_map, flow, back_flow, feature1, feature2, attn_1, attn_2 #, scale_den

    

class SMDCAlignment(nn.Module):

    def __init__(self,cfg, num_feat, scale_num):
        super(SMDCAlignment, self).__init__()
        
        channel_size = num_feat
        # channel_size = 256

        # self.feature_fusion = MultiScaleFeatureFusion(channel_size*2, scales_num=scale_num)


        # self.deformable_conv = DeformableConv2d(128, 128, 4, kernel_size=3, padding = 1)



        self.multi_scale_dcn_alignment = MultiScaleDeformableAlingment(cfg, num_feat=channel_size, deformable_groups=4)

        # self.reduce_channel_1 = nn.Sequential(
        #         ResBlock(in_dim=channel_size*3, out_dim=channel_size*2, dilation=0, norm="bn"),
        #         ResBlock(in_dim=channel_size*2, out_dim=256, dilation=0, norm="bn")
        #     )
        # self.reduce_channel_2 = nn.Sequential(
        #         ResBlock(in_dim=channel_size*3, out_dim=channel_size*2, dilation=0, norm="bn"),
        #         ResBlock(in_dim=channel_size*2, out_dim=256, dilation=0, norm="bn")
        #     )
        # self.reduce_channel_3 = nn.Sequential(
        #         ResBlock(512, 256, dilation=0, norm="bn"),
        #         ResBlock(256, 128, dilation=0, norm="bn")
        #     )




        # self.region_attn = VariantRegionAttention(kernel_size=3)

     


    def forward(self, f1, f2):

        # Fuse the multi-scale feature to generate source feature to be compared
       


        f1_aligned, f_flow = self.multi_scale_dcn_alignment(f1, f2)
        f2_aligned, b_flow = self.multi_scale_dcn_alignment(f2, f1)
       

    
        attn_2 = []
        attn_1 = []

        f1_aligned = f1_aligned[::-1]
        f2_aligned = f2_aligned[::-1]
        

        f_out = []
        f_in = []

        for scale in range(3):
            f_out.append(torch.cat([f1[scale], f2_aligned[scale]], dim=1))
            f_in.append(torch.cat([f2[scale], f1_aligned[scale]], dim=1))

            attn_1.append([f1[scale],f2_aligned[scale]])
            attn_2.append([f2[scale],f1_aligned[scale]])
            
        # f1_aligned = torch.cat([f1_aligned[0],  F.interpolate(f1_aligned[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f1_aligned[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        # f2_aligned = torch.cat([f2_aligned[0],  F.interpolate(f2_aligned[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f2_aligned[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        # f1 = torch.cat([f1[0],  F.interpolate(f1[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f1[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        # f2 = torch.cat([f2[0],  F.interpolate(f2[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f2[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        
        # f1, f2 = self.reduce_channel_1(f1), self.reduce_channel_1(f2)
        # f1_aligned, f2_aligned = self.reduce_channel_2(f1_aligned), self.reduce_channel_2(f2_aligned)
      



        # attn_1.insert(0,[f1,f2_aligned])
        # attn_2.insert(0,[f2,f1_aligned])

        # f_out = self.reduce_channel_3(torch.concat([f1, f2_aligned],dim=1))
        # f_in = self.reduce_channel_3(torch.concat([f2, f1_aligned], dim=1))


   
        # return torch.cat([f_out, f_in], dim=0), f_flow, b_flow, attn_1[::-1], attn_2[::-1] 
        return f_out, f_in, f_flow, b_flow, attn_1[::-1], attn_2[::-1] 






class VariantRegionAttention(nn.Module):

    def __init__(self, in_channel):
        super(VariantRegionAttention,self).__init__()
        # kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)


    def forward(self, target, compare):

        # diff = (target - compare)
        diff = (target - compare)
        # diff = (compare - target)

        diff = self.conv(diff)
        diff = torch.sigmoid(diff)
        
        return diff