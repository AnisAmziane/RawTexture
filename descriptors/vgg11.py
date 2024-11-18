# the VGG11 architecture

import torch
import torch.nn as nn
import gc
class VGG11(nn.Module):
    def __init__(self, in_channels,patch_size, num_classes):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        # convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.NbMaps, self.w, self.h = self._get_required_sizes()
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            # nn.Linear(in_features=512*7*7, out_features=4096),
            nn.Linear(in_features=self.NbMaps * self.w* self.h, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )
        self.head = nn.Linear(in_features=512, out_features=self.num_classes)


    def _get_required_sizes(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tmp = torch.zeros((1, self.in_channels,self.patch_size, self.patch_size))
        tmp = tmp.to(device)
        tmp = self.conv_blocks.to(device)(tmp)
        _,NbMaps, w, h = tmp.size()
        del tmp
        gc.collect()
        torch.cuda.empty_cache()
        return NbMaps, w, h

    def descriptor(self,x):
            x = self.conv_blocks(x)
            # flatten to prepare for the fully connected layers
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    def classifier(self, x):
            x = self.head(x)
            return x

    def forward(self, x):
        x = self.descriptor(x)
        x = self.classifier(x)
        return x