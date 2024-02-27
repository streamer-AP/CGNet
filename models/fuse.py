import torch
import torch.nn as nn
import torch.nn.functional as F

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class FFN(nn.Module):
    def __init__(self, in_chans, mid_chans, out_chans):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, mid_chans, 1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_chans, out_chans, 1)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(192,96,1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(384, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.ms_cam=MS_CAM(96+96+96+2)
        self.out1 = nn.Sequential(            
            nn.Conv2d(96+96+96+2, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3,dilation=2,padding=2),
            nn.GELU(),
            nn.Conv2d(128, 64, 3,dilation=2,padding=2),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.GELU(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z,c1,c2):
        # print(x.shape)
        # torch.Size([1, 96, 256, 256])
        # torch.Size([1, 192, 128, 128])
        # torch.Size([1, 384, 64, 64])
        # torch.Size([1, 768, 32, 32])
        z1,z2=z
        x1, x2 , x3  = z1
        y1, y2 , y3  = z2
        x1 = self.conv1(torch.cat([x1,y1],dim=1))
        x2 = self.conv2(torch.cat([x2,y2],dim=1))
        x3 = self.conv3(torch.cat([x3,y3],dim=1))
        # x3 = self.conv3(x3)
        # x4 = self.conv4(x4)

        x3=  F.interpolate(x3, scale_factor=4)
        x2 = F.interpolate(x2, scale_factor=2)

        z=torch.cat([x1,x2,x3,c1,c2],dim=1)
        z=self.ms_cam(z)

        out1 = self.out1(z)
        return out1


def build_head():
    return Head()
