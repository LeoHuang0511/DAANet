import torch
import torch.nn as nn
from .VGG.VGG16_FPN import VGG16_FPN
from .dcn import DeformableConv2d
from .VGG.conv import ResBlock
import torch.nn.functional as F

BN_MOMENTUM = 0.01





# +
class SMDCANet(nn.Module):

    def __init__(self, cfg, cfg_data):
        super(SMDCANet, self).__init__()
        self.cfg = cfg

        self.Extractor = VGG16_FPN(cfg)
        num_feat = 128
        
        self.deformable_alignment = SMDCAlignment(cfg, num_feat, scale_num=3)
        

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


        self.confidence_predict_layer = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),

            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),

            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
            
            
        )
        nn.init.constant_(self.confidence_predict_layer[6].weight, 0.)
        nn.init.constant_(self.confidence_predict_layer[6].bias, 0.)
        

    
        self.cfg = cfg
# -

#         

# +

    



    def forward(self, img):

        img_pair_num = img.size(0)//2 
        size = img.shape
        
        feature, den_scales, feature_den = self.Extractor(img)

        dens = []
        
        for scale in range(len(den_scales)):

            den_scales[scale] = den_scales[scale] / self.cfg.DEN_FACTOR
            dens.append(F.interpolate(den_scales[scale], scale_factor=2**scale,mode='bilinear',align_corners=True) / 2**(2*scale))

        #     f = torch.cat([f_out[scale],  f_in[scale]],dim=0)
        #     mask = self.mask_predict_layer[scale](f)
        #     masks.append(mask)

            feature_den[scale] = F.adaptive_avg_pool2d(feature_den[scale], (size[2]//self.cfg.CONF_BLOCK_SIZE, size[3]//self.cfg.CONF_BLOCK_SIZE)) # (b*2, 128, h/conf_block_size, w/conf_block_size)
        
        dens = torch.cat(dens, dim=1) # (b*2,3,h,w)

        f_con = torch.cat([feature_den[0],  feature_den[1], feature_den[2]], dim=1) # (b*2, 384, 48, 64)
        confidences = self.confidence_predict_layer(f_con)
        confidences = F.upsample_nearest(confidences, scale_factor = self.cfg.CONF_BLOCK_SIZE).cuda()
        confidences = torch.softmax(confidences,dim=1) # (b*2,3,h,w)

        dens = torch.sum(dens, dim=1).unsqueeze(1)


        # conf = F.adaptive_avg_pool2d(confidences, output_size=feature[0].shape[2:])
        # feature =torch.cat([feature[0] * conf[:,0,:,:].unsqueeze(1),  \
        #                     F.interpolate(feature[1],scale_factor=2,mode='bilinear',align_corners=True) * conf[:,1,:,:].unsqueeze(1),
        #                     F.interpolate(feature[2],scale_factor=4, mode='bilinear',align_corners=True) * conf[:,2,:,:].unsqueeze(1)], dim=1)

        feature1 = []
        feature2 = []
        for scale in range(len(feature)):
            
            conf = confidences[:,scale,:,:].detach().unsqueeze(1)
            conf = F.adaptive_avg_pool2d(conf, feature[scale].shape[2:])


    
            # feature[scale] = F.sigmoid(conf) * feature[scale]
            feature[scale] = conf * feature[scale]

           

            feature1.append(feature[scale][0::2,:,:,:]) # (b,c,h,w)
            feature2.append(feature[scale][1::2,:,:,:])
        

        f, flow , back_flow, attn_1, attn_2, f1, f2 = self.deformable_alignment(feature1, feature2)
        mask = self.mask_predict_layer(f)



        # mask_prob = torch.softmax(mask, dim=1)
        mask = torch.sigmoid(mask)
        
        # den_prob = torch.sum(mask_prob[:,1:3,:,:], dim=1).unsqueeze(1)
        # io_prob = mask_prob[:,1,:,:].unsqueeze(1)


        # out_den = dens[0::2,:,:,:] * io_prob[:img_pair_num,:,:,:]
        # in_den = dens[1::2,:,:,:] * io_prob[img_pair_num:,:,:,:]
        
        out_den = dens[0::2,:,:,:].clone().detach() * mask[:img_pair_num,:,:,:]
        in_den = dens[1::2,:,:,:].clone().detach() * mask[img_pair_num:,:,:,:]




        # return  den_scales, masks, confidences, flow, back_flow, feature1, feature2, attn_1, attn_2
        # return  den_scales, dens, mask, out_den, in_den, den_prob, io_prob, confidences, flow, back_flow, feature1, feature2, attn_1, attn_2
        return  den_scales, dens, mask, out_den, in_den, mask, mask, confidences, flow, back_flow, f1, f2, attn_1, attn_2



class SMDCAlignment(nn.Module):

    def __init__(self,cfg, num_feat, scale_num):
        super(SMDCAlignment, self).__init__()
        
        self.channel_size = num_feat

        # self.feature_head = nn.Sequential(
        #                         nn.Dropout2d(0.2),

        #                         # ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size*3, dilation=0, norm="bn"),
        #                         # ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size, dilation=0, norm="bn")
        #                         ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size*2, dilation=0, norm="bn"),
        #                         ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size*2, dilation=0, norm="bn"),

        #                         nn.Conv2d(self.channel_size*2, self.channel_size*2, kernel_size=3, stride=1, padding=1, bias=False),
        #                         nn.BatchNorm2d(self.channel_size*2, momentum=BN_MOMENTUM),
        #                         nn.ReLU(inplace=True),
        #                         nn.Conv2d(self.channel_size*2, self.channel_size*2, kernel_size=3, stride=1, padding=1)
        # )
        # self.multi_scale_dcn_alignment = DeformableConv2d(cfg, self.channel_size*2, self.channel_size*2, offset_groups=4, kernel_size=3, mult_column_offset=True)

        # self.weight_conv = nn.Sequential(
        #                         ResBlock(in_dim=self.channel_size*4, out_dim=self.channel_size*2, dilation=0, norm="bn"),
        #                         ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size, dilation=0, norm="bn")
        # )

        self.feature_head = nn.ModuleList()
        self.multi_scale_dcn_alignment = nn.ModuleList()

        for scale in range(3):
            self.feature_head.append(nn.Sequential(
                                nn.Dropout2d(0.2),

                                # ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size*3, dilation=0, norm="bn"),
                                # ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size, dilation=0, norm="bn")
                                ResBlock(in_dim=self.channel_size, out_dim=self.channel_size, dilation=0, norm="bn"),

                                nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(self.channel_size, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.channel_size, self.channel_size, kernel_size=3, stride=1, padding=1)
                                ))
            self.multi_scale_dcn_alignment.append(
                                DeformableConv2d(cfg, self.channel_size, self.channel_size, offset_groups=4, kernel_size=3, mult_column_offset=True, scale=scale)
            )


        
        

        self.weight_conv = nn.Sequential(
                                ResBlock(in_dim=self.channel_size*6, out_dim=self.channel_size*3, dilation=0, norm="bn"),
                                ResBlock(in_dim=self.channel_size*3, out_dim=self.channel_size*2, dilation=0, norm="bn"),
                                ResBlock(in_dim=self.channel_size*2, out_dim=self.channel_size, dilation=0, norm="bn")

        )

     


    def forward(self, f1, f2):

        
       
        # f1 =torch.cat([f1[0],  F.interpolate(f1[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f1[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        # f2 =torch.cat([f2[0],  F.interpolate(f2[1],scale_factor=2,mode='bilinear',align_corners=True),
        #               F.interpolate(f2[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)

        
        # f1 = self.feature_head(f1)
        # f2 = self.feature_head(f2)
        # f1_aligned, f_flow = self.multi_scale_dcn_alignment(f1, f2)
        # f2_aligned, b_flow = self.multi_scale_dcn_alignment(f2, f1)

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
        attn_1 = [f1, f2_aligned]
        attn_2 = [f2, f1_aligned]

        f_mask = torch.cat([f_out,  f_in],dim=0)


            
    
        return f_mask, f_flow, b_flow, attn_1, attn_2, f1, f2



