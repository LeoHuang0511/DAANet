import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from model.attention import MultiScaleFeatureFusion, OffsetFusion, OffsetFusion_2, OffsetFusion_3
# from model.SMDCA import VariantRegionAttention
# from .dcn.dcn import DCNv2Pack


class MultiScaleDeformableAlingment(nn.Module):

    def __init__(self,cfg, num_feat, deformable_groups=4, deform_kernel_size=3):

        super(MultiScaleDeformableAlingment, self).__init__()
        self.cfg = cfg

        self.offset_conv0 = nn.ModuleDict()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        # self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        # self.feat_conv = nn.ModuleDict()
        # self.fusion = nn.ModuleDict()
        self.deformable_groups = deformable_groups
        self.deform_kernel_size = deform_kernel_size
        

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            # self.offset_conv0[level] = nn.Sequential(
            #     nn.Conv2d((num_feat // deformable_groups)*2, (num_feat // deformable_groups), 3, 1, 1),
            #     # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            #     nn.Conv2d((num_feat // deformable_groups), (num_feat // deformable_groups), 3, 1, 1),
            #     # nn.LeakyReLU(negative_slope=0.1, inplace=True)
            #     )
            

            
            self.offset_conv1[level] = nn.Conv2d((num_feat // deformable_groups)*2, 
                            2 * self.deform_kernel_size * self.deform_kernel_size,
                            kernel_size=3, 
                            stride=1,
                            padding=1, 
                            bias=True)
            
            
            nn.init.constant_(self.offset_conv1[level].weight, 0.)
            nn.init.constant_(self.offset_conv1[level].bias, 0.)
            
            # if i == 3:
            #     self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # if i != 3:
                # self.offset_conv2[level] = nn.Conv2d(2 * self.deform_kernel_size * self.deform_kernel_size * deformable_groups * 2, 
                #                                     2 * self.deform_kernel_size * self.deform_kernel_size * deformable_groups, 
                #                                     kernel_size=3, 
                #                                     stride=1,
                #                                     padding=1, 
                #                                     bias=True)

                # self.offset_conv2[level] = OffsetFusion(2 * self.deform_kernel_size * self.deform_kernel_size * deformable_groups)

                # self.offset_conv2[level] = OffsetFusion(2 * self.deform_kernel_size * self.deform_kernel_size * deformable_groups, deformable_groups=self.deformable_groups)
                # self.offset_conv2[level] = OffsetFusion(2 * self.deform_kernel_size * self.deform_kernel_size , deformable_groups=self.deformable_groups)

                # self.offset_conv2[level] = OffsetFusion_2(2 * self.deform_kernel_size * self.deform_kernel_size * deformable_groups)
                # self.offset_conv2[level] = OffsetFusion_3(num_feat, deformable_groups=self.deformable_groups)
                
              




                # self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DeformableConv(cfg, i, num_feat, num_feat, deformable_groups, 3, padding=1,)
        


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)



    def forward(self, sou, ref):
        """Align neighboring frame features to the reference frame features.

        Args:
            sou (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        num_group_channel = ref[0].size()[1] // self.deformable_groups

        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        aligned_feats = [] 
        offset_vs = []
        cat_feat = []
        for i in range(3, 0, -1):
            level = f'l{i}'

            b, _, h, w = sou[i - 1].shape

            offset = []
            for j in range(self.deformable_groups):

                # offset_input = self.offset_conv0[level](torch.concat([sou[i - 1][:,j*num_group_channel:(j+1)*num_group_channel,:,:], 
                #                                                       ref[i - 1][:,j*num_group_channel:(j+1)*num_group_channel,:,:]], axis = 1))
                offset_input = torch.concat([sou[i - 1][:,j*num_group_channel:(j+1)*num_group_channel,:,:], 
                                            ref[i - 1][:,j*num_group_channel:(j+1)*num_group_channel,:,:]], axis = 1)
                
                offset.append(self.offset_conv1[level](offset_input))
            offset = torch.concat(offset, axis = 1)

           
            # if i != 3:
                # print(offset.shape)
                # print(upsampled_offset.shape)

              
                # offset = self.offset_conv2[level](offset, upsampled_offset)

                # offset = self.offset_conv2[level](offset, upsampled_offset)


            

            # print(offset.shape)
            
            feat, offset_v = self.dcn_pack[level](sou[i - 1], offset)
            offset_v = offset_visualization(offset_v, 1//(self.cfg.feature_scale/(2**(i-1))))

            offset_vs.append(offset_v)
            # print(offset.shape)
            # if i < 3:
            #     feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            # if i > 1:
                # feat = self.lrelu(feat)

            # aligned_feats.append(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                # upsampled_offset = self.upsample(offset) * 2
                upsampled_offset = self.upsample(offset) * 2

                # upsampled_feat = self.upsample(feat)


            # attn = self.fusion[level](ref[i - 1], feat)
            # attn = torch.ones(feat.shape[0],1,feat.shape[2], feat.shape[3]).cuda()
            # cat_feat.append(attn*torch.cat([ref[i - 1], feat],dim=1))
            cat_feat.append(feat)

        # feats = self.fusion(aligned_feats[::-1])

        # Cascading
        # offset_c = torch.cat([feat, ref[0]], dim=1)
        # offset_c = []
        # for j in range(self.deformable_groups):
        #         offset_input = torch.concat([feat[:,j*num_group_channel:(j+1)*num_group_channel,:,:], 
        #                 ref[0][:,j*num_group_channel:(j+1)*num_group_channel,:,:]], axis = 1)
        #         offset_c.append(self.cas_offset_conv1(offset_input))
        # offset_c = torch.concat(offset_c, axis = 1)
        # offset_c = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset_c))))
        # offset_c = self.cas_offset_conv2(self.cas_offset_conv1(offset_c))


        # feat, _ = self.cas_dcnpack(feat, offset_c)
        # feat = self.lrelu(feat)



        # return feat, offset_vs
        # return cat_feat, offset_vs, attn
        return cat_feat, offset_vs

    
class DeformableConv(nn.Module):
    def __init__(self,
                 cfg,
                 level,
                 in_channels,
                 out_channels,
                 offset_groups = 4,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 ):

        super(DeformableConv, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        self.cfg = cfg
        self.level = level
        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_groups = offset_groups
        # self.offset_conv = nn.Conv2d(2 * self.kernel_size[0] * self.kernel_size[1] * self.offset_groups, 
        #                              2 * self.kernel_size[0] * self.kernel_size[1] * self.offset_groups,
        #                              kernel_size=kernel_size, 
        #                              stride=stride,
        #                              padding=self.padding, 
        #                              bias=True)

        # nn.init.constant_(self.offset_conv.weight, 0.)
        # nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     offset_groups * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    def forward(self, warp_ref, offset_map):
        h, w = warp_ref.shape[2:]
        # max_offset = max(h, w)/4.
        # num_group_channel = offset.size()[1] // self.offset_groups
        

        # offset_map = []
        
        
        # offset_map = self.offset_conv(offset)
        #offset range
        # offset_range = 50 / 2**(self.level-1)
        offset_range = (min(h, w)*0.5) /2


        # offset_range = 50 / 2**(self.level)

        # offset_range = 50

        # offset_map = 100* torch.sigmoid(offset_map) - 50
        offset_map = offset_range * 2 * torch.sigmoid(offset_map) - offset_range


        # offset_vis = offset_visualization(offset_map, 1 // self.cfg.feature_scale)
        # offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(warp_ref))
        # self.offset_loss = F.l1_loss(offset_x,offset_x_mean.detach()) + F.l1_loss(offset_y, offset_y_mean.detach())
        x = torchvision.ops.deform_conv2d(input=warp_ref, 
                                          offset=offset_map, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x,offset_map
    

def offset_visualization(offset, feature_scale):
    offset_y = offset[:,0::2,:,:] #vertical
    offset_x = offset[:,1::2,:,:] #horizontal
    offset_y_mean = torch.mean(offset_x,dim = 1, keepdims = True)
    offset_x_mean = torch.mean(offset_y,dim = 1, keepdims = True)

    offset = torch.concat([offset_x_mean,offset_y_mean],axis = 1)
    offset = F.interpolate(offset,scale_factor=feature_scale,mode='bilinear',align_corners=True) * feature_scale
    return offset
    
   
# class VariantRegionAttention(nn.Module):

#     def __init__(self, kernel_size, stride=1):
#         super(VariantRegionAttention,self).__init__()
#         kernel_size = kernel_size
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)
#         # self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)


#     def forward(self, target, compare):

#         diff = (target - compare)
#         # diff = (target - compare)**2


#         b,c,h,w = diff.shape
#         diff = torch.cat((torch.max(diff, 1)[0].unsqueeze(1), torch.mean(diff, 1).unsqueeze(1)), dim=1) #channel pool
#         diff = torch.sigmoid(self.conv(diff))



#         # diff = -1 * pixel_cos_sim(target, compare) + 1
#         # b,c,h,w = diff.shape


#         # # diff = torch.cat((torch.max(diff, 1)[0].unsqueeze(1), torch.mean(diff, 1).unsqueeze(1)), dim=1) #channel pool
#         # diff = torch.softmax(self.conv(diff).view(b,c,h*w),dim=2).view(b,c,h,w)
        
        
#         return diff
    
def pixel_cos_sim(fa, fb):
    dot = fa * fb
    dot = torch.sum(dot, dim=1)
    sim = dot / (torch.sqrt(torch.sum(fa**2,dim=1)) * torch.sqrt(torch.sum(fb**2,dim=1)))
    sim = sim[:,None,:,:]


    return sim
    





