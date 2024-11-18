import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from torchmetrics import Accuracy

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Image2Patch_Embedding(nn.Module):
    def __init__(self, image_size,pattern_size,channels , dim):
        super().__init__()
        patch_height, patch_width = pair(image_size)
        self.conv_raw = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=pattern_size, stride=pattern_size, padding=0),
            nn.GELU())
        self.im2patch = Rearrange('b c h w -> b (h w) c', h=patch_height, w=patch_width)

    def forward(self, x):
        x = self.conv_raw(x)
        x = self.im2patch(x)
        return x



class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        # self.layer_norm_1 = nn.LayerNorm(embed_dim)
        # self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.proj3 = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        # self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x,ca):
        Q = self.proj1(x)
        K = self.proj2(x)
        V = self.proj3(x) * ca
        x = x + self.attn(Q, K, V)[0]
        x = x + self.linear(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class PatchAggregation(nn.Module):
    """down sample the feature resolution, build with conv 2x2 stride 2
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride_size=2,padding=0):
        super(PatchAggregation, self).__init__()
        self.patch_aggregation = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=pair(kernel_size),
            stride=pair(stride_size),
            padding=pair(padding)

        )

    def forward(self, x):
        x = self.patch_aggregation(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self,H,W, dim, heads, hidden_channels, dropout=0.):
        super().__init__()
        self.H = H
        self.W = W
        #
        self.attn = AttentionBlock(dim, hidden_channels, heads, dropout=dropout)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,groups=dim, kernel_size=3, stride=1, padding=1),
            nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1x1 =  nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU())
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=dim, kernel_size=1),
            nn.Sigmoid())
        self.channel_interaction = nn.Sequential(self.pool,self.conv1x1,self.conv1x1_2)
        self.spatial_interaction = nn.Sequential(self.conv1x1,self.conv1x1_2)
        self.layer_norm = nn.LayerNorm(dim)

    # x = torch.rand((2,169,128))
    def forward(self, x):
        x0 = Rearrange('b (h w) c -> b c h w',h=self.H,w=self.W)(x)
        x0 = self.dwconv(x0)
        ca = self.channel_interaction(x0)
        ca = ca.reshape(ca.size(0),-1).unsqueeze(1)
        ## Channel interaction
        out_1 = self.attn(x,ca)
        _,_,c = out_1.size()
        out_1 = Rearrange('b (h w) (c p n) -> (b h w) c p n',c=c,n=1,p=1, h=self.H,w=self.W)(out_1)
        ## Spatial interaction
        out_2 = self.spatial_interaction(out_1)
        out_2 = Rearrange('(b h w) c p n -> b (h w) (c p n)',c=c,n=1,p=1, h=self.H,w=self.W)(out_2)
        out_3 = self.pool(x0).reshape(out_2.size(0), -1).unsqueeze(1)*out_2

        return out_3



class MixFormer(nn.Module):
    def __init__(self, dim, heads, hidden_channels, pattern_size, image_size, channels,classes,dropout=0.):
        super().__init__()
        H1 = W1 = image_size//pattern_size
        self.raw_embedding = Image2Patch_Embedding(H1,pattern_size,channels , dim)
        self.stage1 = MixerBlock(H1,W1, dim, heads, hidden_channels, dropout=dropout)
        self.raarrange_stage1 = Rearrange('b (h w) c -> b c h w',h=H1,w=W1)
        self.conv_pool = PatchAggregation(dim, dim*3, kernel_size=3, stride_size=2,padding=0)
        # tmp = self.conv_pool(self.raarrange_stage1(self.stage1(self.raw_embedding(torch.randn((2,1,image_size,image_size)))))).to(device)
        # _,_,H2,W2 = tmp.size()
        # self.raarrange_stage2 = Rearrange('b c h w -> b (h w) c',h=H2,w=W2)
        # self.stage2 = MixerBlock(H2,W2, dim*3, heads, hidden_channels, dropout=dropout)
        # self.raarrange_stage3 = Rearrange('b (h w) c -> b c h w', h=H2, w=W2)
        # self.conv_pool2 = PatchAggregation(dim*3, dim, kernel_size=3, stride_size=2, padding=0)
        # self.raarrange_stage3 = Rearrange('b (h w) c -> b c h w',h=H2,w=W2)
        # self.avg_pool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(dim*3,dim)
        self.gelu = nn.GELU()
        self.head = nn.Linear(dim,classes)

    def descriptor(self,x):
        x = self.raw_embedding(x)
        x = self.stage1(x)
        x = self.raarrange_stage1(x)
        x = self.gelu(self.conv_pool(x))

        #
        # x = self.raarrange_stage2(x)
        # x = self.stage2(x)
        # x = self.raarrange_stage3(x)
        # x = self.conv_pool2(x)
        x = x.mean((2, 3))
        #
        x = self.gelu(self.fc(x))
        return x
    def classifier(self,x):

        x = self.head(x)
        return x

    def forward(self, x):
        x = self.descriptor(x)
        x = self.classifier(x)

        return x


