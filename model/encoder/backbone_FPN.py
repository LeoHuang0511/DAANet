from  torchvision import models
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
from model.necks import FPN
from .conv import ResBlock, BasicConv

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

# +
class backbone_FPN(nn.Module):
    def __init__(self,cfg):
        super(backbone_FPN, self).__init__()

        self.cfg = cfg
        if cfg.BACKBONE == "swin":
            swin = models.swin_b(weights='Swin_B_Weights.IMAGENET1K_V1')
            features = list(swin.features.children())
            self.layer1 = nn.Sequential(*features[0:2])
            self.layer2 = nn.Sequential(*features[2:4])
            self.layer3 = nn.Sequential(*features[4:6])

            in_channels = [128,256,512]
            

        elif cfg.BACKBONE == "vgg":
            vgg = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
            features = list(vgg.features.children())
            self.layer1 = nn.Sequential(*features[0:23])
            self.layer2 = nn.Sequential(*features[23:33])
            self.layer3 = nn.Sequential(*features[33:43])

            in_channels = [256,512,512]

        elif cfg.BACKBONE == "res101":
            res101 = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
            features = list(res101.children())
            self.layer1 = nn.Sequential(*features[0:5])
            self.layer2 = nn.Sequential(*features[5:6])
            self.layer3 = nn.Sequential(*features[6:7])

            in_channels = [256,512,1024]

        elif cfg.BACKBONE == "res50":
            res50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
            features = list(res50.children())
            self.layer1 = nn.Sequential(*features[0:5])
            self.layer2 = nn.Sequential(*features[5:6])
            self.layer3 = nn.Sequential(*features[6:7])

            in_channels = [256,512,1024]

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
            self.scale_loc_head.append(nn.Sequential(
             

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
                              
                            ))
            
            

           


    def forward(self, x):
        f_list = []
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        if self.cfg.BACKBONE == 'swin':
            x1 = x1.permute(0,3,1,2)
            x2 = x2.permute(0,3,1,2)
            x3 = x3.permute(0,3,1,2)

        f_list.append(x1)
        f_list.append(x2)
        f_list.append(x3)



        f_den = self.neck(f_list)
        den_scale = []
        for scale in range(len(f_den)):
            
            f_den[scale] = self.scale_loc_bottleneck[scale](f_den[scale])
            den_scale.append(self.scale_loc_head[scale](f_den[scale]))


        f_mask = self.neck2f(f_list)
       
        


        return f_mask, den_scale, f_den
# -





