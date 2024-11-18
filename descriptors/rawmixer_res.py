import torch
import torch.nn as nn
from torch import nn, einsum
import sys
sys.path.append('C:/Users/anisa/PycharmProjects/MyProjects/venv/Raw/TextureExp/descriptors/')


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        # x = x + self.attn(inp_x)
        x = x + self.linear(self.layer_norm_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads,  hidden_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(embed_dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=0.0),
                FeedForward(embed_dim, hidden_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class ConvMixer_maxpool(nn.Module):
    def __init__(self, channels,dim, depth, sfa_width=5, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.depth = depth
        self.sfa_width = sfa_width
        self.kernel_size = kernel_size

        self.first_conv = nn.Sequential(
        nn.Conv2d(self.channels, self.dim, kernel_size=self.sfa_width, stride=self.sfa_width),
        nn.SELU(),
        nn.BatchNorm2d(self.dim))  # this is what I use before

        # self.first_conv = rawconv_block(channels, dim, kernel_size1, reduction=kernel_size1) # just trying with this
        self.mixer_conv = nn.Sequential(*[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(self.dim, self.dim, self.kernel_size, groups=self.dim, padding="same"),
                    nn.SELU(),
                    nn.BatchNorm2d(self.dim)
                )),
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.SELU(),
                nn.BatchNorm2d(self.dim)
        ) for i in range(self.depth)])
        self.residual_mixer = Residual(self.mixer_conv)

        self.maxpool2x2 = torch.nn.MaxPool2d((2,2))
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,x):
        res = self.first_conv(x)
        x = self.residual_mixer(res)
        x = self.maxpool2x2(x)
        x = x.flatten(2,3).permute(0,2,1)
        # x = self.avgpool(x)
        return x


#-- classic ViT
class RawMixerRes(nn.Module):
    def __init__(
        self,
            in_C,
            sfa_width,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        input_size,
        dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()
        H, W = input_size
        num_patches = (H // sfa_width) * (W // sfa_width)
        # self.patch_size = patch_size

        # Layers/Networks
        self.convmixer = ConvMixer_maxpool(in_C,embed_dim, 2, sfa_width=sfa_width, kernel_size=3) # marche mieux avec depth=2
        self.transformer = Transformer(embed_dim, num_layers, num_heads,  hidden_dim, dropout = 0.)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.mlp_head = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def descriptor(self,x):
        x =  self.convmixer(x)
        B, T, _ = x.shape
        # Apply Transformer
        # x = x + self.pos_embedding[:, : T]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls = torch.mean(x, dim=0)
        # Preprocess input
        dense = self.layer_norm(cls)
        dense = self.layer_norm2(self.fc(dense))
        return dense

    def classifier(self,x):
        # Perform classification prediction
        out = self.mlp_head(x)
        return out

    def forward(self, x):
        features = self.descriptor(x)
        out = self.classifier(features)
        return out


