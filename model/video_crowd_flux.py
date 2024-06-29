import torch
import torch.nn as nn
from .encoder.backbone_FPN import backbone_FPN
from .dcn import MultiScaleDeformableConv
from .encoder.conv import ResBlock
import torch.nn.functional as F

BN_MOMENTUM = 0.01





# +
class SOFANet(nn.Module):

    def __init__(self, cfg, cfg_data):
        super(SOFANet, self).__init__()
        self.cfg = cfg

        self.Extractor = backbone_FPN(cfg)
        num_feat = 128
        
        self.deformable_alignment = MOFAlignment(cfg, num_feat, scale_num=3)
        

        self.mask_predict_layer = nn.Sequential(

            nn.Dropout2d(0.2),
            ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
            ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),

            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8, momentum=BN_MOMENTUM),

            nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),

            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
            )


        self.ASAM = nn.Sequential(


            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),           
        )

        nn.init.constant_(self.ASAM[6].weight, 0.)
        nn.init.constant_(self.ASAM[6].bias, 0.)
        

    
        self.cfg = cfg


    def DDA(self, feature, attns):
        feature1 = []
        feature2 = []
        for scale in range(len(feature)):
            attn = attns[:,scale,:,:].detach().unsqueeze(1)
            attn = F.adaptive_avg_pool2d(attn, feature[scale].shape[2:])
            feature[scale] = attn * feature[scale]
            feature1.append(feature[scale][0::2,:,:,:]) # (b,c,h,w)
            feature2.append(feature[scale][1::2,:,:,:])

        return feature1, feature2


    def forward(self, img):

        img_pair_num = img.size(0)//2 
        size = img.shape
        
        feature, den_scales, feature_den = self.Extractor(img)

        dens = []
        
        for scale in range(len(den_scales)):

            den_scales[scale] = den_scales[scale] / self.cfg.DEN_FACTOR
            dens.append(F.interpolate(den_scales[scale], scale_factor=2**scale,mode='bilinear',align_corners=True) / 2**(2*scale))

            feature_den[scale] = F.adaptive_avg_pool2d(feature_den[scale], (size[2]//self.cfg.CONF_BLOCK_SIZE, size[3]//self.cfg.CONF_BLOCK_SIZE)) # (b*2, 128, h/conf_block_size, w/conf_block_size)
        

        dens = torch.cat(dens, dim=1) # (b*2,3,h,w)
        dens = torch.sum(dens, dim=1).unsqueeze(1)


        f_attn = torch.cat([feature_den[0],  feature_den[1], feature_den[2]], dim=1) # (b*2, 384, 48, 64)
        attns = self.ASAM(f_attn)
        attns = F.upsample_nearest(attns, scale_factor = self.cfg.CONF_BLOCK_SIZE).cuda()
        attns = torch.softmax(attns,dim=1) # (b*2,3,h,w)
        



        feature1, feature2 = self.DDA(feature, attns)
        f, f_flow , b_flow, f1, f2 = self.deformable_alignment(feature1, feature2)
        
        mask = self.mask_predict_layer(f)
        mask = torch.sigmoid(mask)
        
        
        out_den = dens[0::2,:,:,:].clone().detach() * mask[:img_pair_num,:,:,:]
        in_den = dens[1::2,:,:,:].clone().detach() * mask[img_pair_num:,:,:,:]



        return  den_scales, dens, mask, out_den, in_den, attns, f_flow, b_flow, f1, f2




class MOFAlignment(nn.Module):

    def __init__(self,cfg, num_feat, scale_num):
        super(MOFAlignment, self).__init__()
        
        self.channel_size = num_feat

     
        self.feature_head = nn.ModuleList()
        self.multi_scale_dcn_alignment = nn.ModuleList()

        for scale in range(3):
            self.feature_head.append(nn.Sequential(
                                nn.Dropout2d(0.2),

                                ResBlock(in_dim=self.channel_size, out_dim=self.channel_size, dilation=0, norm="bn"),

                                nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(self.channel_size, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1)
                                ))
            
            self.multi_scale_dcn_alignment.append(
                                MultiScaleDeformableConv(cfg, self.channel_size, self.channel_size, offset_groups=4, kernel_size=3, mult_column_offset=True, scale=scale)
            )


        

        self.weight_conv = nn.Sequential(
                                ResBlock(in_dim=self.channel_size*6, out_dim=self.channel_size*3, dilation=0, norm="bn"),
                                ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size*2, dilation=0, norm="bn"),
                                ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size, dilation=0, norm="bn")

        )

     


    def forward(self, f1, f2):

        
       
       

        f_flow = []
        b_flow = []
        f1_aligned = []
        f2_aligned = []

        for scale in range(len(f1)):
            f1[scale] = self.feature_head[scale](f1[scale])
            f2[scale] = self.feature_head[scale](f2[scale])

            f1_align, ff = self.multi_scale_dcn_alignment[scale](f1[scale], f2[scale])
            f2_align, bf = self.multi_scale_dcn_alignment[scale](f2[scale], f1[scale])
            f_flow.append(ff)
            b_flow.append(bf)
            f1_aligned.append(f1_align)
            f2_aligned.append(f2_align)

        
        f1 =torch.cat([f1[0],  F.interpolate(f1[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f1[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        f2 =torch.cat([f2[0],  F.interpolate(f2[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f2[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        f1_aligned =torch.cat([f1_aligned[0],  F.interpolate(f1_aligned[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f1_aligned[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        f2_aligned =torch.cat([f2_aligned[0],  F.interpolate(f2_aligned[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f2_aligned[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)



        
       
        f_out = self.weight_conv(torch.cat([f1, f2_aligned], dim=1))
        f_in = self.weight_conv(torch.cat([f2, f1_aligned], dim=1))

        f_mask = torch.cat([f_out,  f_in],dim=0)



    
        return f_mask, f_flow, b_flow, f1, f2



