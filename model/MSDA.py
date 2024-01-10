import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
# from model.attention import MultiScaleFeatureFusion, OffsetFusion, OffsetFusion_2, OffsetFusion_3
from .dcn import DeformableConv2d

# from model.SMDCA import VariantRegionAttention
# from .dcn.dcn import DCNv2Pack


# class MultiScaleDeformableAlingment(nn.Module):

#     def __init__(self,cfg, num_feat, deformable_groups=4, deform_kernel_size=3):

#         super(MultiScaleDeformableAlingment, self).__init__()
#         self.cfg = cfg

        
#         self.deformable_groups = deformable_groups
#         self.deform_kernel_size = deform_kernel_size
#         self.channel_size = num_feat


#         self.deformable_convs = nn.ModuleDict()

            
        

#         # Pyramids
#         for i in range(3, 0, -1):
#             level = f'l{i}'
          
#             self.deformable_convs[level] = DeformableConv2d(self.channel_size, self.channel_size, self.deformable_groups, kernel_size=self.deform_kernel_size, padding = 1)
        





#     def forward(self, sou, ref):
        

        
#         offset_vs = []
#         cat_feat = []
#         for i in range(3, 0, -1):
#             level = f'l{i}'

#             b, _, h, w = sou[i - 1].shape

         
#             feat, offset_v = self.deformable_convs[level](sou[i - 1], ref[i - 1])
            
#             offset_v = offset_visualization(offset_v, 1 //( self.cfg.feature_scale / (2 ** (i - 1))))


#             offset_vs.append(offset_v)
          

#             cat_feat.append(feat)

 
#         return cat_feat, offset_vs


class OffsetVariantDeformableAlingment(nn.Module):

    def __init__(self,cfg, num_feat, deformable_groups=4, deform_kernel_size=3):

        super(OffsetVariantDeformableAlingment, self).__init__()
        self.cfg = cfg

        
        self.deformable_groups = deformable_groups
        self.deform_kernel_size = deform_kernel_size
        self.channel_size = num_feat


        # self.deformable_convs = nn.ModuleDict()

            
        

        # # Pyramids
        # for i in range(3, 0, -1):
        #     level = f'l{i}'
          
        #     self.deformable_convs[level] = DeformableConv2d(self.channel_size, self.channel_size, self.deformable_groups, kernel_size=self.deform_kernel_size, padding = 1)
        
        deformable_conv = DeformableConv2d(self.channel_size, self.channel_size, self.deformable_groups, kernel_size=self.deform_kernel_size, padding = 1, mult_column_offset=True)




    def forward(self, sou, ref):
        

        
        

        feat, offset_v = self.deformable_conv(sou, ref)
        offset_v = offset_visualization(offset_v, 1 //( self.cfg.feature_scale))
        
        

 
        return feat, offset
    
    

def offset_visualization(offset, feature_scale):
    offset_y = offset[:,0::2,:,:] #vertical
    offset_x = offset[:,1::2,:,:] #horizontal
    offset_y_mean = torch.mean(offset_x,dim = 1, keepdims = True)
    offset_x_mean = torch.mean(offset_y,dim = 1, keepdims = True)

    offset = torch.concat([offset_x_mean,offset_y_mean],axis = 1)
    offset = F.interpolate(offset,scale_factor=feature_scale,mode='bilinear',align_corners=True) * feature_scale
    return offset
    
   

    

    





