import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange,Reduce
from torch.nn import functional as F


class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)

        if gamma is not None:
            self.gamma = torch.tensor([gamma], device='cuda')
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0], device='cuda'))
        self.eps = torch.tensor([1e-6], device='cuda')

        self.gamma = gamma
        self.use_beta = use_beta

    # def prepare_input(self, x,gap_cls):
    #         B, d, H, W = x.shape
    #         x = x.reshape(B, d, H * W).permute(0, 2, 1)  # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
    #         gap_cls = gap_cls.squeeze(2).permute(0,2,1)  # (B, d,1,1) -> (B, 1, d)
    #         return gap_cls, x


    def forward(self, x,gap_cls):
        # Prepare input tensor and perform GAP as initialization
        # gap_cls, x = self.prepare_input(x,gap_cls)

        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)

        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape

        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv

        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        # qq = q.reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1,
        #                                                                           3)  # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)

        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1,
                                                                                      3)  # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)

        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1,
                                                                             3)  # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)

        # Compute attention scores
        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)

        # If gamma scaling is used
        if self.gamma is not None:
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma),1 / self.gamma)  # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)

        return x.squeeze()

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):

    def __init__(self, dim, num_patch, mlp_dim, dropout = 0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x


# def ConvMixer(in_channels,dim, depth, kernel_size=5, patch_size=5,act=nn.GELU()):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
#         act,
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                    act,
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                act,
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)])

class ConvMixer(nn.Module):
    def __init__(self, channels,dim, depth, sfa_width=5, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.sfa_width = sfa_width
        self.kernel_size = kernel_size

        self.first_conv = nn.Sequential(
        nn.Conv2d(self.channels, self.dim, kernel_size=self.sfa_width, stride=self.sfa_width),
        nn.GELU(),
        nn.BatchNorm2d(self.dim))  # this is what I use before

        # self.first_conv = rawconv_block(channels, dim, kernel_size1, reduction=kernel_size1) # just trying with this

        self.mixer_conv = nn.Sequential(*[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(self.dim)
                )),
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(self.dim)
        ) for i in range(self.depth)])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,x):
        x = self.first_conv(x)
        x = self.mixer_conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class SqEx(nn.Module):

    def __init__(self, with_residual=True):
        super(SqEx, self).__init__()
        self.with_residual = with_residual
    def forward(self, x, y):
        texture = x * y
        if self.with_residual:
            texture += x
        return texture


class ResMLP(nn.Module):

    def __init__(self, in_channels, dim, num_classes,sfa_width, image_size, depth, mlp_dim):
        super().__init__()

        self.num_patch = (image_size// sfa_width) ** 2
        self.sfa_width = sfa_width
        # shuffle = torch.nn.PixelShuffle(sfa_width)

        self.mixer_texture = ConvMixer(in_channels, dim, 2, sfa_width = sfa_width, kernel_size = 3)
        self.proj_spectra = nn.Linear(sfa_width**2, dim)

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        self.affine = Aff(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        self.simpool = SimPool(dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reduce = Reduce('b n c -> b c', 'mean')
        self.reshape = Rearrange('b c h w -> b (h w) c')

    def descriptor(self, x):
        unshuffled_x = F.pixel_unshuffle(x, self.sfa_width)
        spectra = self.proj_spectra(self.reshape(self.avg_pool(unshuffled_x)))
        texture = self.reshape(self.mixer_texture(x))

        for mlp_block in self.mlp_blocks:
            texture = mlp_block(texture)

        # for mlp_block in self.mlp_blocks:
        #     spectra = mlp_block(spectra)

        texture = self.affine(texture)
        spectra = self.affine(spectra)
        # feat = torch.cat((texture, spectra), dim=1)
        feat = self.simpool(spectra,self.reduce(texture).unsqueeze(1))

        # feat = feat.mean(dim=1)

        return feat

    def classifier(self, x):
        x = self.mlp_head(x)
        return x

    def forward(self, x):
        x = self.descriptor(x)
        x = self.classifier(x)
        return x


# in_channels =1
# dim = 128
# num_classes= 112
# sfa_width= 5
# image_size = 125
# depth= 2
# mlp_dim =256
#
# m = ResMLP(1, 128, 112, 5, 125, 2, 256)
# x = torch.randn(3,1,125,125)
# m(x)