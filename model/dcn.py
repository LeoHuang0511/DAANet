import torch
import torchvision.ops
from torch import nn
import torch.nn.functional as F
# from .MSDA import offset_visualization

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 offset_groups = 4,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 mult_column_offset=False,
                 bias=False,
                 offset = None
                 ):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_groups = offset_groups
        if mult_column_offset:
            self.offset_conv = MultiColumnOffsetConv(in_channels // offset_groups * 2, self.kernel_size, self.stride, self.padding)
        else:
            self.offset_conv = nn.Conv2d(in_channels // offset_groups * 2, 
                                        2 * self.kernel_size[0] * self.kernel_size[1],
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=self.padding, 
                                        bias=True)

            nn.init.constant_(self.offset_conv.weight, 0.)
            nn.init.constant_(self.offset_conv.bias, 0.)
        self.offset = offset
        
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
        self.offset_loss = 0
    def forward(self, warp_ref, source):
        h, w = warp_ref.shape[2:]

        num_group_channel = warp_ref.size()[1] // self.offset_groups
        offset_map = []
        if self.offset != None:
            offset_map = self.offset.repeat(1,self.kernel_size[0] * self.kernel_size[1],1,1)
        else:
            for i in range(self.offset_groups):
                offset_input = torch.concat([warp_ref[:,i*num_group_channel:(i+1)*num_group_channel,:,:], 
                    source[:,i*num_group_channel:(i+1)*num_group_channel,:,:]], axis = 1)
                offset_map.append(self.offset_conv(offset_input))
            offset_map = torch.concat(offset_map, axis = 1)

        #offset range
        offset_range = (min(h, w)*0.5) /2
        offset_map = offset_range * 2 * torch.sigmoid(offset_map) - offset_range

       
        # offset = offset_visualization(offset_map, 8)

        modulator = 2. * torch.sigmoid(self.modulator_conv(warp_ref))


        x = torchvision.ops.deform_conv2d(input=warp_ref, 
                                          offset=offset_map, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
                                          
        offset_vis = offset_visualization(offset_visualization(offset_v, 1 //( self.cfg.feature_scale)))
        return x, offset_vis

class MultiColumnOffsetConv(nn.Module):

    def __init__(self, in_dim, kernel_size, stride, padding):

        super(MultiColumnOffsetConv,self).__init__()

        self.conv1 =  nn.Conv2d(in_dim, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True)

        self.column1 = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True)
        )
        self.column2 = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=2, stride=stride, padding=2, bias=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True)
        )
        self.column3 = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=3, stride=stride, padding=3,bias=True),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True)
        )
        self.conv2 = nn.Conv2d(in_dim//2*3, in_dim//2, kernel_size=kernel_size, dilation=1, stride=stride, padding=1,bias=True)
        
        self.offset_conv = nn.Conv2d(in_dim//2*3, 
                                2 * kernel_size[0] * kernel_size[1], 
                                kernel_size=3, 
                                dilation=1, 
                                stride=stride, 
                                padding=1,
                                bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([self.column1(x1), self.column2(x1), self.column3(x1)], dim=1)
        x2 = self.conv2(x2)
        x3 = self.offset_conv(x1 + x2)

        return x3
        
def offset_visualization(offset, feature_scale):
    offset_y = offset[:,0::2,:,:] #vertical
    offset_x = offset[:,1::2,:,:] #horizontal
    offset_y_mean = torch.mean(offset_x,dim = 1, keepdims = True)
    offset_x_mean = torch.mean(offset_y,dim = 1, keepdims = True)

    offset = torch.concat([offset_x_mean,offset_y_mean],axis = 1)
    offset = F.interpolate(offset,scale_factor=feature_scale,mode='bilinear',align_corners=True) * feature_scale
    return offset