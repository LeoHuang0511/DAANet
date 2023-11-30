from  torchvision import models
import sys
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
# from torchsummary import summary
from model.necks import FPN
from .conv import ResBlock, BasicConv

from model.attention import MultiScaleFeatureFusion

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

# +
class VGG16_FPN(nn.Module):
    def __init__(self,cfg):
        super(VGG16_FPN, self).__init__()

        self.cfg = cfg
        # vgg = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        vgg = models.vgg16_bn()

        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:23])
        self.layer2 = nn.Sequential(*features[23:33])
        self.layer3 = nn.Sequential(*features[33:43])

        in_channels = [256,512,512]
        self.neck = FPN(in_channels,192,len(in_channels))

        self.neck2f = FPN(in_channels, 128, len(in_channels))
        

        self.idx = 1

        

        self.scale_loc_bottleneck = nn.ModuleList()
        self.scale_loc_head = nn.ModuleList()
        
        for i in range(len(in_channels)):
            self.scale_loc_bottleneck.append(nn.Sequential(
                                nn.Dropout2d(0.2),
                             
                                ResBlock(in_dim=192, out_dim=128, dilation=0, norm="bn"),
                                ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),
            ))
#             self.scale_loc_head.append(nn.Sequential(


#                                 nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
#                                 nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
#                                 nn.ReLU(inplace=True),

#                                 nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#                                 nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
#                                 nn.ReLU(inplace=True),

#                                 nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
#                                 nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
#                                 nn.ReLU(inplace=True),

#                                 nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#                                 nn.ReLU(inplace=True)
#                             ))
        self.loc_head=nn.Sequential(

                    nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
                    nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),

                    nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
                    nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True)
                )
            
            
        self.feature_head = nn.ModuleList()
        for i in range(len(in_channels)):
            self.feature_head.append(nn.Sequential(
                            nn.Dropout2d(0.2),
                            ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),
                            ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),

                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                            BatchNorm2d(128, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        ))


    def forward(self, x):
        f_list = []
        x1 = self.layer1(x)
        f_list.append(x1)
        x2 = self.layer2(x1)
        f_list.append(x2)
        x3 = self.layer3(x2)
        f_list.append(x3)



        f_den = self.neck(f_list)
        den_scale = []
        for scale in range(len(f_den)):
            
            f_den[scale] = self.scale_loc_bottleneck[scale](f_den[scale])
#             den_scale.append(self.scale_loc_head[scale](f_den[scale]))
            den_scale.append(self.loc_head(f_den[scale]))

      


        f_mask = self.neck2f(f_list)
        for scale in range(len(f_mask)):
            f_mask[scale] = self.feature_head[scale](f_mask[scale])
        


        return f_mask, den_scale, f_den
# -



    

