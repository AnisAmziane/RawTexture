import torch
import torch.nn as nn
from torch.nn import functional as F
# from functools import partial
# from einops import rearrange,repeat
# import math

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
        x = x + self.linear(self.layer_norm_2(x))
        return x




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
    def forward(self,x):
        x = self.first_conv(x)
        x = self.mixer_conv(x)
        x = x.flatten(2,3).permute(0,2,1)
        return x

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
        self.maxpool2x2 = torch.nn.MaxPool2d((2,2))
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,x):
        x = self.first_conv(x)
        x = self.mixer_conv(x)
        x = self.maxpool2x2(x)
        x = x.flatten(2,3).permute(0,2,1)
        # x = self.avgpool(x)
        return x


#-- classic ViT
class RawMixer(nn.Module):
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
        # self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        # self.input_layer = nn.Linear(num_channels * (patch_size**2), 128)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.mlp_head = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def descriptor(self,x):
        x =  self.convmixer(x)
        B, T, _ = x.shape
        # Add CLS token and positional encoding
        # cls_token = self.cls_token.repeat(B, 1, 1)
        # x = torch.cat([cls_token, x], dim=1)
        # x = x + self.pos_embedding[:, : T + 1]
        x = x + self.pos_embedding[:, : T]
        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls = torch.mean(x, dim=0)
        # cls = x[0] # first row is the class learned tokens (features)
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


# class RawFormer2(nn.Module):
#     def __init__(
#         self,
#             in_C,
#             sfa_width,
#         embed_dim,
#         hidden_dim,
#         num_heads,
#         num_layers,
#         num_classes,
#         input_size,
#         dropout=0.0):
#         """
#         Inputs:
#             embed_dim - Dimensionality of the input feature vectors to the Transformer
#             hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
#                          within the Transformer
#             num_channels - Number of channels of the input (3 for RGB)
#             num_heads - Number of heads to use in the Multi-Head Attention block
#             num_layers - Number of layers to use in the Transformer
#             num_classes - Number of classes to predict
#             patch_size - Number of pixels that the patches have per dimension
#             num_patches - Maximum number of patches an image can have
#             dropout - Amount of dropout to apply in the feed-forward network and
#                       on the input encoding
#         """
#         super().__init__()
#         H, W = input_size
#         num_patches = (H // sfa_width) * (W // sfa_width)
#         # self.patch_size = patch_size
#
#         # Layers/Networks
#         self.convmixer = ConvMixer(in_C,embed_dim, 3, sfa_width=sfa_width, kernel_size=3) # marche mieux avec depth=2
#         # self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
#         # self.input_layer = nn.Linear(num_channels * (patch_size**2), 128)
#         self.transformer = nn.Sequential(
#             *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
#         )
#         self.layer_norm = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, 128)
#         self.layer_norm2 = nn.LayerNorm(128)
#         self.mlp_head = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(dropout)
#
#         # Parameters/Embeddings
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
#
#     def descriptor(self,x):
#         x =  self.convmixer(x)
#         B, T, _ = x.shape
#         # Add CLS token and positional encoding
#         cls_token = self.cls_token.repeat(B, 1, 1)
#         x = torch.cat([cls_token, x], dim=1)
#         x = x + self.pos_embedding[:, : T + 1]
#         # Apply Transformer
#         x = self.dropout(x)
#         x = x.transpose(0, 1)
#         x = self.transformer(x)
#         cls = x[0] # first row is the class learned tokens (features)
#         # Preprocess input
#         dense = self.layer_norm(cls)
#         dense = self.layer_norm2(self.fc(cls))
#         return dense
#
#     def classifier(self,x):
#         # Perform classification prediction
#         out = self.mlp_head(x)
#         return out
#
#     def forward(self, x):
#         features = self.descriptor(x)
#         out = self.classifier(features)
#         return out






# m = RawFormer(1,
#             5,
#         128,
#         384,
#         2,
#         2,
#         112,
#         (125,125),
#         dropout=0.0)
#
x = torch.randn(128,1,125,125)
# m.descriptor(x)