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
        
    

        self.mask_predict_layer = nn.ModuleList()
        for i in range(3):
        
            self.mask_predict_layer.append(nn.Sequential(

            nn.Dropout2d(0.2),

            ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),
            ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
            ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),

            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8, momentum=BN_MOMENTUM),

            nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),

            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
            ))
        

    
        self.cfg = cfg

#         


    



    def forward(self, img):

        img_pair_num = img.size(0)//2 
        feature, den_scales = self.Extractor(img)



        feature1 = []
        feature2 = []
        for fea in feature:
    
            feature1.append(fea[0::2,:,:,:])
            feature2.append(fea[1::2,:,:,:])

        

        
        for scale in range(len(den_scales)):
            den_scales[scale] = den_scales[scale] / self.cfg.DEN_FACTOR

        f_out, f_in, flow , back_flow, attn_1, attn_2= self.deformable_alignment(feature1, feature2)



        masks = []
        pre_outflow_maps = []
        pre_inflow_maps = []
        for scale in range(len(f_out)):
            f = torch.cat([f_out[scale],  f_in[scale]],dim=0)
            f = self.mask_predict_layer[scale](f)
            f = torch.sigmoid(f)
            masks.append(f)

            pre_outflow_maps.append((f[:img_pair_num,:,:,:])* den_scales[scale][0::2,:,:,:].detach())
            pre_inflow_maps.append((f[img_pair_num:,:,:,:])* den_scales[scale][1::2,:,:,:].detach())

        

        return  den_scales, masks, pre_outflow_maps, pre_inflow_maps, flow, back_flow, feature1, feature2, attn_1, attn_2

    

class SMDCAlignment(nn.Module):

    def __init__(self,cfg, num_feat, scale_num):
        super(SMDCAlignment, self).__init__()
        
        self.channel_size = num_feat

        self.multi_scale_dcn_alignment = MultiScaleDeformableAlingment(cfg, self.channel_size, deformable_groups=4)

        self.weight_convs = nn.ModuleList()

        for i in range(scale_num):
            self.weight_convs.append(nn.Sequential(
                                ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size*2, dilation=0, norm="bn"),
                                ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size, dilation=0, norm="bn")
            ))

     


    def forward(self, f1, f2):

       


        f1_aligned, f_flow = self.multi_scale_dcn_alignment(f1, f2)
        f2_aligned, b_flow = self.multi_scale_dcn_alignment(f2, f1)
        f1_aligned = f1_aligned[::-1]
        f2_aligned = f2_aligned[::-1]
        f_flow = f_flow[::-1]
        b_flow = b_flow[::-1]
    


        attn_2 = []
        attn_1 = []
        f_out = []
        f_in = []

        for scale in range(3):
            f_out.append(self.weight_convs[scale](torch.cat([f1[scale], f2_aligned[scale]], dim=1)))
            f_in.append(self.weight_convs[scale](torch.cat([f2[scale], f1_aligned[scale]], dim=1)))

            attn_1.append([f1[scale],f2_aligned[scale]])
            attn_2.append([f2[scale],f1_aligned[scale]])
            
    
        return f_out, f_in, f_flow, b_flow, attn_1, attn_2



