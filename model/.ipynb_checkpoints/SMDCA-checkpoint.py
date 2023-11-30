import torch
import torch.nn as nn
from .VGG.VGG16_FPN import VGG16_FPN
from .attention import MultiScaleFeatureFusion, SpatialWeightLayer
from .MSDA import MultiScaleDeformableAlingment#, VariantRegionAttention
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
        
    

#         self.mask_predict_layer = nn.ModuleList()
#         for i in range(3):
        
#             self.mask_predict_layer.append(nn.Sequential(

#             nn.Dropout2d(0.2),

#             ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),
#             ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
#             ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),

#             nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
#             nn.BatchNorm2d(16, momentum=BN_MOMENTUM),

#             nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8, momentum=BN_MOMENTUM),

#             nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
#             nn.BatchNorm2d(4, momentum=BN_MOMENTUM),

#             # nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0),

#             ))

        self.mask_predict_layer = nn.Sequential(

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

            # nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0),

            )

            
#         self.confidence_predict_layer = nn.ModuleList()
#         for i in range(3):
        
#             self.confidence_predict_layer.append(nn.Sequential(

          
#             nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
#             # nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),

#             nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),

#             ))
            
#             nn.init.constant_(self.confidence_predict_layer[i][3].weight, 0.)
#             nn.init.constant_(self.confidence_predict_layer[i][3].bias, 0.)
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
        feature, den_scales, feature_den = self.Extractor(img)



        feature1 = []
        feature2 = []
        feature_den1 = []
        feature_den2 = []
        for scale in range(len(feature)):
    
            feature1.append(feature[scale][0::2,:,:,:])
            feature2.append(feature[scale][1::2,:,:,:])

#             feature_den1.append(feature_den[scale][0::2,:,:,:])
#             feature_den2.append(feature_den[scale][1::2,:,:,:])

        

        

        f_out, f_in, flow , back_flow, attn_1, attn_2= self.deformable_alignment(feature1, feature2)



        masks = []
        confidences = []

        pre_outflow_maps = []
        pre_inflow_maps = []
        den_prob = []
        io_prob = []
        upsampled_den = []
        
        for scale in range(len(f_out)):

            den_scales[scale] = den_scales[scale] / self.cfg.DEN_FACTOR


            f = torch.cat([f_out[scale],  f_in[scale]],dim=0)
#             mask = self.mask_predict_layer[scale](f)
            mask = self.mask_predict_layer(f)


#             confidence = self.confidence_predict_layer[scale](feature_den[scale])


#             confidence = F.upsample_nearest(confidence, scale_factor=2**(scale))
            
            # f = torch.sigmoid(f)
            masks.append(mask)
#             confidences.append(confidence)
#         confidences = torch.cat(confidences, dim=1)
        
        f_con = torch.cat([feature_den[0],  F.interpolate(feature_den[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(feature_den[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
#         f_con = F.adaptive_avg_pool2d(f_con, (size[2]//self.cfg.CONF_BLOCK_SIZE, size[3]//self.cfg.CONF_BLOCK_SIZE))
        confidences = self.confidence_predict_layer(f_con)



        return  den_scales, masks, confidences, flow, back_flow, feature1, feature2, attn_1, attn_2
    
    def scale_fuse(self, den_scales, masks, confidence, mode):

        img_pair_num = den_scales[0].shape[0]//2
        dens = []
        out_dens = []
        in_dens = []
        den_probs = []
        io_probs = []

        for scale in range(len(masks)):
            den = torch.zeros_like(den_scales[scale]).cuda()
            mask_prob = torch.softmax(masks[scale], dim=1)
            den_prob = torch.sum(mask_prob[:,1:3,:,:], dim=1).unsqueeze(1)
            io_prob = mask_prob[:,1,:,:].unsqueeze(1)

            den_probs.append(den_prob)
            io_probs.append(io_prob)


            den[0::2,:,:,:] = den_scales[scale][0::2,:,:,:] * den_prob[:img_pair_num,:,:,:]
            den[1::2,:,:,:] = den_scales[scale][1::2,:,:,:] * den_prob[img_pair_num:,:,:,:]
            out_den = den_scales[scale][0::2,:,:,:] * io_prob[:img_pair_num,:,:,:]
            in_den = den_scales[scale][1::2,:,:,:] * io_prob[img_pair_num:,:,:,:]

            
            den = F.interpolate(den,scale_factor=2**scale,mode='bilinear',align_corners=True) / 2**(2*scale)
            out_den = F.interpolate(out_den,scale_factor=2**scale,mode='bilinear',align_corners=True) / 2**(2*scale)
            in_den = F.interpolate(in_den,scale_factor=2**scale,mode='bilinear',align_corners=True) / 2**(2*scale)
            

            
            dens.append(den)
            out_dens.append(out_den)
            in_dens.append(in_den)
        
        dens = torch.cat(dens, dim=1)

        out_dens = torch.cat(out_dens, dim=1)
        in_dens = torch.cat(in_dens, dim=1)

        confidence = F.upsample_nearest(confidence, scale_factor = 1//self.cfg.feature_scale).cuda()


        # if mode == "train" or mode == 'val':
#         if mode == "train":

#             conf_mask = torch.zeros_like(confidence).cuda()
#             confidence = torch.softmax(confidence,dim=1)
#             for scale in range(conf_mask.shape[1]):
#         #         # try:
#                 conf_mask[:,scale][torch.where(torch.argmax(confidence,dim=1).squeeze()==scale)] = 1
#         #         # except:
#         #         #     print(conf_mask[:,scale])
#         #         #     break
#         else:
#             conf_mask = torch.softmax(confidence,dim=1)
        conf_mask = torch.softmax(confidence,dim=1)
        


#         final_den = torch.zeros((dens.shape[0],1,dens.shape[2], dens.shape[3])).cuda()

#         final_den[0::2,:,:,:] = torch.sum(dens[0::2,:,:,:] * conf_mask[:img_pair_num,:,:,:], dim=1).unsqueeze(1)
#         final_den[1::2,:,:,:] = torch.sum(dens[1::2,:,:,:] * conf_mask[img_pair_num:,:,:,:], dim=1).unsqueeze(1)
        final_den = torch.sum(dens * conf_mask, dim=1).unsqueeze(1)



#         out_dens = torch.sum(out_dens * conf_mask[:img_pair_num,:,:,:], dim=1).unsqueeze(1)
#         in_dens = torch.sum(in_dens * conf_mask[img_pair_num:,:,:,:], dim=1).unsqueeze(1)
        
        out_dens = torch.sum(out_dens * conf_mask[0::2,:,:,:], dim=1).unsqueeze(1)
        in_dens = torch.sum(in_dens * conf_mask[1::2,:,:,:], dim=1).unsqueeze(1)

        return final_den, out_dens, in_dens, den_probs, io_probs, conf_mask
# -



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



