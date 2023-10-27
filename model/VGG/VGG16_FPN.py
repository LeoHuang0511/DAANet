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

class VGG16_FPN(nn.Module):
    def __init__(self,cfg):
        super(VGG16_FPN, self).__init__()

        self.cfg = cfg
        vgg = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:23])
        self.layer2 = nn.Sequential(*features[23:33])
        self.layer3 = nn.Sequential(*features[33:43])

        in_channels = [256,512,512]
        # self.neck = FPN(in_channels,192,len(in_channels))
        self.neck = FPN(in_channels,192,len(in_channels))

        self.neck2f = FPN(in_channels, 128, len(in_channels))
        
        # self.fuse = MultiScaleFeatureFusion(192, scales_num=len(in_channels))

        self.idx = 1

        # self.conv = nn.Conv2d(576, 576, kernel_size=3, padding=1)
        # self.f_con2 = nn.Conv2d(512, 256, kernel_size=1)
        # self.f_con3 = nn.Conv2d(512, 256, kernel_size=1)

        
        self.loc_head = nn.Sequential(
            nn.Dropout2d(0.2),
            ResBlock(in_dim=576, out_dim=256, dilation=0, norm="bn"),
            ResBlock(in_dim=256, out_dim=128, dilation=0, norm="bn"),
            # ResBlock(in_dim=192, out_dim=128, dilation=0, norm="bn"),
            # ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),

            # nn.ConvTranspose2d(192, 64, 2, stride=2, padding=0, output_padding=0, bias=False),

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

        self.scale_loc_head = nn.ModuleList()
        for i in range(len(in_channels)):
            self.scale_loc_head.append(nn.Sequential(
                                nn.Dropout2d(0.2),
                             
                                ResBlock(in_dim=192, out_dim=128, dilation=0, norm="bn"),
                                ResBlock(in_dim=128, out_dim=128, dilation=0, norm="bn"),



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



        f_loc = self.neck(f_list)


        feat1 = np.max((f_loc[0]).detach().cpu().numpy(),axis=1)[0]
        feat1 = (255 * feat1 / (feat1.max() + 1e-10))
        feat1 = cv2.applyColorMap(feat1.astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        feat1 = Image.fromarray(cv2.cvtColor(feat1, cv2.COLOR_BGR2GRAY))
        
        feat2 = np.max((F.interpolate(f_loc[1],scale_factor=2,mode='bilinear',align_corners=True)).detach().cpu().numpy(),axis=1)[0]
        feat2 = (255 * feat2 / (feat2.max() + 1e-10))
        feat2 = cv2.applyColorMap(feat2.astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        feat2 = Image.fromarray(cv2.cvtColor(feat2, cv2.COLOR_BGR2GRAY))
        
        
        feat3 = np.max(F.interpolate(f_loc[2],scale_factor=4, mode='bilinear',align_corners=True).detach().cpu().numpy(),axis=1)[0]
        feat3 = (255 * feat3 / (feat3.max() + 1e-10))
        feat3 = cv2.applyColorMap(feat3.astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        feat3 = Image.fromarray(cv2.cvtColor(feat3, cv2.COLOR_BGR2GRAY))



        
        den_scale = []
        for i, f in enumerate(f_loc):
            
            den_scale.append(self.scale_loc_head[i](f))
      
            

        x =torch.cat([f_loc[0],  F.interpolate(f_loc[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f_loc[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        
        feat4 = np.max((x).detach().cpu().numpy(),axis=1)[0]
        feat4 = (255 * feat4 / (feat4.max() + 1e-10))
        feat4 = cv2.applyColorMap(feat4.astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        feat4 = Image.fromarray(cv2.cvtColor(feat4, cv2.COLOR_BGR2GRAY))

        
        
#    
        
        
        imgs = [feat1, feat2,\
                feat3, feat4]
        w_num , h_num=2, 2
        UNIT_W, UNIT_H = feat4.size

        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            xx, yy = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
            target.paste(img, (xx, yy, xx + UNIT_W, yy + UNIT_H))
            count+=1
        

        
        if (self.training == True):
            self.idx += 1

            if ((self.idx % self.cfg.SAVE_VIS_FREQ )== 0):
                dir = os.path.join(self.cfg.EXP_PATH, self.cfg.EXP_NAME, 'vis_loc')
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                target.save( os.path.join(dir, f"visualization{self.idx}.jpg"))

        x = self.loc_head(x)



        f_mask = self.neck2f(f_list)
        for scale in range(len(f_mask)):
            f_mask[scale] = self.feature_head[scale](f_mask[scale])
        


        return f_mask, x, den_scale
    


    

        
# class MultiBranchModule(nn.Module):
#     def __init__(self, in_channels, sync=False):
#         super(MultiBranchModule, self).__init__()
#         self.branch1x1 = BasicConv(in_channels, in_channels//2, kernel_size=1, relu=True)
#         self.branch1x1_1 = BasicConv(in_channels//2, in_channels, kernel_size=1, relu=True)

#         self.branch3x3_1 = BasicConv(in_channels, in_channels//2, kernel_size=1, relu=True)
#         self.branch3x3_2 = BasicConv(in_channels // 2, in_channels, kernel_size=(3, 3), padding=(1, 1), relu=True)

#         self.branch3x3dbl_1 = BasicConv(in_channels, in_channels//2, kernel_size=1)
#         self.branch3x3dbl_2 = BasicConv(in_channels // 2, in_channels, kernel_size=5, padding=2, relu=True)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#         branch1x1 = self.branch1x1_1(branch1x1)

#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = self.branch3x3_2(branch3x3)

#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

#         outputs = [branch1x1, branch3x3, branch3x3dbl, x]
#         return torch.cat(outputs, 1)
