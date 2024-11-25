import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_raw(nn.Module):
    def __init__(self, ch_in, ch_out,msfa_size=5):
        super(conv_block_raw, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=msfa_size, stride=msfa_size, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# class Unet(nn.Module):
#     def __init__(self, img_ch=1, output_ch=112,msfa_size=5, device='cuda'):
#         super(Unet, self).__init__()
#         self.device = device
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         #
#         self.Conv1 = conv_block_raw(ch_in=img_ch, ch_out=64,msfa_size=msfa_size)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         self.Conv5 = conv_block(ch_in=512, ch_out=1024)
#
#         self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
#
#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
#
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
#
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
#
#         self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
#
#     def get_device(self):
#         return self.device
#
#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#         #
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#         #
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
#         #
#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)
#
#         # decoding + concat path
#         d5 = self.Up5(x5)
#         b,c,m,n = d5.size()
#         d5 = torch.cat((x4[:,:,:m,:n], d5), dim=1)
#         d5 = self.Up_conv5(d5)
#         #
#         d4 = self.Up4(d5)
#         b,c,m,n = d4.size()
#         d4 = torch.cat((x3[:,:,:m,:n], d4), dim=1)
#         d4 = self.Up_conv4(d4)
#         #
#         d3 = self.Up3(d4)
#         b,c,m,n = d3.size()
#         d3 = torch.cat((x2[:,:,:m,:n], d3), dim=1)
#         d3 = self.Up_conv3(d3)
#         #
#         d2 = self.Up2(d3)
#         b,c,m,n = d2.size()
#         d2 = torch.cat((x1[:,:,:m,:n], d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         #
#         d1 = self.Conv_1x1(d2)
#         d1 = nn.functional.interpolate(d1, size=x.shape[-2:], mode="bilinear",align_corners=False)
#         # d1 = F.softmax(d1,dim=1)
#
#         return d1


class Unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=112,msfa_size=5, device='cuda'):
        super(Unet, self).__init__()
        self.device = device
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.Conv1 = conv_block_raw(ch_in=img_ch, ch_out=64,msfa_size=msfa_size)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        # self.Conv4 = conv_block(ch_in=256, ch_out=512)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        #
        # self.Up4 = up_conv(ch_in=512, ch_out=256)
        # self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def get_device(self):
        return self.device

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        #
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        #
        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)
        # #
        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # b,c,m,n = d5.size()
        # d5 = torch.cat((x4[:,:,:m,:n], d5), dim=1)
        # d5 = self.Up_conv5(d5)
        # #
        # d4 = self.Up4(d5)
        # b,c,m,n = d4.size()
        # d4 = torch.cat((x3[:,:,:m,:n], d4), dim=1)
        # d4 = self.Up_conv4(d4)
        #
        d3 = self.Up3(x3)
        b,c,m,n = d3.size()
        d3 = torch.cat((x2[:,:,:m,:n], d3), dim=1)
        d3 = self.Up_conv3(d3)
        #
        d2 = self.Up2(d3)
        b,c,m,n = d2.size()
        d2 = torch.cat((x1[:,:,:m,:n], d2), dim=1)
        d2 = self.Up_conv2(d2)
        #
        d1 = self.Conv_1x1(d2)
        d1 = nn.functional.interpolate(d1, size=x.shape[-2:], mode="bilinear",align_corners=False)
        # d1 = F.softmax(d1,dim=1)
        return d1


# prediction = nn.functional.interpolate(outputs, size=mask[:, :, :256, :256].shape[-2:], mode="bilinear",
#                                        align_corners=False)
