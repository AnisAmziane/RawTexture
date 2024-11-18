# -*- coding: utf-8 -*-
""" Title: Implementation of different CNN models using PyTorch
    Author: Anis Amziane <anisamziane6810@gmail.com>
    Created: 10-Nov-2022
  """
import torch
random_seed = 42 # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.nn import init
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d,AvgPool2d, Module, Softmax, BatchNorm2d
import gc

class MsfaNet(Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            # init.constant_(m.bias,0)
    def freeze(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def __init__(self,input_channels,pattern_size,n_classes,use_maxpool=True,msfa_type='5x5'):
        super().__init__() # if Python2.x modify this line to super(MsfaNet,self).__init__()
        self.pattern_size = pattern_size
        self.msfa_type = msfa_type
        self.use_maxpool= use_maxpool
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.raw_conv = Conv2d(self.input_channels, 128, kernel_size=pattern_size, stride=pattern_size, padding=0)
        self.conv2 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.maxpool2x2 = MaxPool2d((2,2))
        self.relu = ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.bnorm1 = BatchNorm2d(128)
        self.bnorm2 = BatchNorm2d(256)
        self.bnorm3 = BatchNorm2d(384)
        self.bnorm1d = nn.BatchNorm1d(128)
        self.fc1 = Linear(384, 128)
        self.fc2 = Linear(128,self.n_classes)
        self.apply(self.weight_init)
    def pixel_unshuffle(self, input, upscale_factor):
        r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
        tensor of shape :math:`(*, r^2C, H, W)`.
        Authors:
            Zhaoyi Yan, https://github.com/Zhaoyi-Yan
            Kai Zhang, https://github.com/cszn/FFDNet
        Date:
            01/Jan/2019
        """
        batch_size, channels, in_height, in_width = input.size()
        out_height = in_height // upscale_factor
        out_width = in_width // upscale_factor
        input_view = input.contiguous().view(
            batch_size, channels, out_height, upscale_factor,
            out_width, upscale_factor)
        channels *= upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def first_layer(self,x):
        x = self.raw_conv(x)
        x = self.relu(x)
        x = self.bnorm1(x)
        return x
    def descriptor(self, x):
        x = self.first_layer(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bnorm2(x)
        x = self.maxpool2x2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bnorm3(x)
        x = self.maxpool2x2(x)
        x = self.avgpool(x)
        x = self.fc1(x.reshape(x.size(0),-1))
        x = self.relu(x)
        x = self.bnorm1d(x)
        return x

    # def descriptor(self, x):
    #     x = self.first_layer(x)
    #     x = self.maxpool2x2(x)
    #     x = self.avgpool(x).reshape(x.size(0),-1)
    #     x = self.relu(x)
    #     x = self.bnorm1d(x)
    #     return x

    def classifier(self,x):
        x = self.fc2(x)
        return x
    def forward(self,x):
        x = self.descriptor(x)
        # x = self.feature_extractor_simple(x)
        x = self.classifier(x)
        return x

