import torch.nn as nn
import torch
from .VGG.conv import ResBlock

class MultiScaleFeatureFusion(nn.Module):

    def __init__(self, in_channel, scales_num):
        super(MultiScaleFeatureFusion, self).__init__()


        self.channel_attn = ChannelWeightLayer(in_channel*scales_num)

        self.spatial_attns = nn.ModuleList()
        for i in range(scales_num):
            self.spatial_attns.append(SpatialWeightLayer(kernel_size=3))

        # self.reduce_channel = nn.Sequential(
        #     ResBlock(in_channel*scales_num, in_channel*2),\
        #     ResBlock(in_channel*2, in_channel)
        # )

       


    def forward(self, f):
        size =  (f[0].shape[2], f[0].shape[3])

        f = [spatial_attn(f[i])[0] for i, spatial_attn in enumerate(self.spatial_attns)]

        f = torch.cat((f[0], nn.Upsample(size=size, mode='bilinear', align_corners=True)(f[1]),
                              nn.Upsample(size=size, mode='bilinear', align_corners=True)(f[2])), 1)
        
        f = self.channel_attn(f)
        # f = self.reduce_channel(f)

        return f

class OffsetFusion(nn.Module):

    def __init__(self, in_channel, scales_num=2, deformable_groups=4):
        super(OffsetFusion, self).__init__()


        # self.channel_attn = ChannelWeightLayer(in_channel*2)

        # self.spatial_attns = SpatialWeightLayer(kernel_size=3)

        # self.offset_conv = nn.Conv2d(in_channel * 2, 
        #                             in_channel, 
        #                             kernel_size=3, 
        #                             stride=1,
        #                             padding=1, 
        #                             bias=True)


        self.deformable_groups = deformable_groups


        # self.channel_attn = ChannelWeightLayer((in_channel//self.deformable_groups)*2)
        self.channel_attn = ChannelWeightLayer((in_channel)*2)


        self.spatial_attn = SpatialWeightLayer(kernel_size=3)

        # self.offset_conv = nn.Conv2d((in_channel//self.deformable_groups) * 2, 
        #                             (in_channel//self.deformable_groups), 
        #                             kernel_size=3, 
        #                             stride=1,
        #                             padding=1, 
        #                             bias=True)
        
        self.offset_conv = nn.Conv2d((in_channel) * 2, 
                                    (in_channel), 
                                    kernel_size=3, 
                                    stride=1,
                                    padding=1, 
                                    bias=True)
        

    def forward(self, offset, previous_offset):

        size =  (offset.shape[2], offset.shape[3])


        b, c, h, w = offset.shape

        # previous_offset = self.spatial_attns(previous_offset)

        # f = torch.cat([offset, nn.Upsample(size=size, mode='bilinear', align_corners=True)(previous_offset)], 1)

        # f = self.channel_attn(f)
        # f = self.offset_conv(f)

        # return f

        # previous_offset = previous_offset.view(b, self.deformable_groups, -1, h, w)
        # offset = offset.view(b, self.deformable_groups, -1, h, w)

        # fs = []
        # for j in range(self.deformable_groups):
            
        #     previous = previous_offset[:,j,:,:,:]
        #     previous = self.spatial_attn(previous)

        #     f = torch.cat([offset[:,j,:,:,:], previous], dim=1)
        #     f = self.channel_attn(f)
        #     f = self.offset_conv(f)
        #     fs.append(f)
        # fs = torch.concat(fs, axis = 1)
        
        previous = previous_offset
        previous = self.spatial_attn(previous)

        f = torch.cat([offset, previous], dim=1)
        f = self.channel_attn(f)
        f = self.offset_conv(f)
        
        
        
        return f


class OffsetFusion_2(nn.Module):

    def __init__(self, in_channel):
        super(OffsetFusion_2, self).__init__()
       

    
        # self.channel_attn = ChannelWeightLayer(in_channel*2)

        self.spatial_attns = SpatialWeightLayer(kernel_size=3)

        self.conv1 = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1,bias=True)
        nn.init.constant_(self.conv1.weight, 0.)
        nn.init.constant_(self.conv1.bias, 0.)


        # self.offset_conv = nn.Conv2d(in_channel * 2, 
        #                             in_channel, 
        #                             kernel_size=3, 
        #                             stride=1,
        #                             padding=1, 
        #                             bias=True)


       

    def forward(self, offset, previous_offset):

        size =  (offset.shape[2], offset.shape[3])


        b, c, h, w = offset.shape

        _, attn = self.spatial_attns(torch.cat([offset, previous_offset], dim=1))
        residual = torch.cat([ (1-attn) * offset, attn * previous_offset ], dim=1)
        residual = self.conv1(residual)

        
        return offset + residual


class OffsetFusion_3(nn.Module):

    def __init__(self, in_channel, scales_num=2, deformable_groups=4):
        super(OffsetFusion_3, self).__init__()
        self.deformable_groups = deformable_groups
        self.num_group_channel = in_channel // deformable_groups

        self.offset_conv = nn.Sequential(
            nn.Conv2d( self.num_group_channel + 2*3*3, 
                                    self.num_group_channel, 
                                    kernel_size=3, 
                                    stride=1,
                                    padding=1, 
                                    bias=True),
            nn.BatchNorm2d(self.num_group_channel))

        
        

    def forward(self, offset_input, previous_offset):

        size =  (offset_input.shape[2], offset_input.shape[3])


        b, c, h, w = offset_input.shape

   
        # previous_offset = previous_offset.view(b, self.deformable_groups, -1, h, w)

       
            
        previous = previous_offset
        current = offset_input

            
        fs = self.offset_conv(torch.cat([offset_input,previous_offset],dim=1))
        
        
        
        return fs




class ChannelWeightLayer(nn.Module):
    def __init__(self,  in_channel):
        super(ChannelWeightLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


class SpatialWeightLayer(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(SpatialWeightLayer, self).__init__()
        kernel_size = kernel_size
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False)

    def forward(self, x):
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1) #channel pool
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)  # broadcasting
        return scale
