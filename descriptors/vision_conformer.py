# Adapted from https://github.com/uchidalab/vision-conformer/tree/main
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class Image2Patch_Embedding(nn.Module):
    def __init__(self, patch_size, channels, dim):
        super().__init__()
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        self.im2patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.patch2latentv = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.im2patch(x)
        x = self.patch2latentv(x)
        return x


class Latentv2Image(nn.Module):
    def __init__(self, patch_size, channels, dim):
        patch_height, patch_width = pair(patch_size)
        self.latentv2patch = nn.Linear(dim, channels * patch_height * patch_width)
        self.vec2square = Rearrange('(c h w) -> c h w', c=channels, h=patch_height, w=patch_width)

    def forward(self, x):
        return x


class CNN(nn.Module):
    def __init__(self, channels, hidden_channels, depth):
        super().__init__()
        if depth == 1:
            m_list = [self.conv_block(channels, channels)]
        else:
            m_list = [self.conv_block(channels, hidden_channels)]
            for i in range(depth - 2):
                m_list.append(self.conv_block(hidden_channels, hidden_channels))
            m_list.append(self.conv_block(hidden_channels, channels))
        self.blocks = nn.ModuleList(m_list)

    def conv_block(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, *args,
                      **kwargs),
            nn.GELU())

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ReconstructionConv(nn.Module):
    def __init__(self, dim, hidden_channels, patch_size, image_size, cnn_depth, channels=3):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        patch_height, patch_width = pair(patch_size)
        self.patch_dim = channels * patch_height * patch_width

        self.latent2patchv = nn.Linear(in_features=self.dim, out_features=self.patch_dim)
        self.patchv2patch = Rearrange('b p (c h w) -> b p c h w', c=channels, h=patch_height, w=patch_width)
        self.net = CNN(channels=channels, hidden_channels=hidden_channels, depth=cnn_depth)
        self.embedding = Image2Patch_Embedding(patch_size=patch_height, channels=channels, dim=dim)
        self.fc = nn.Linear(in_features=self.dim, out_features=self.dim)

    def reconstruct(self, patchs):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        h = int(image_height / patch_height)
        w = int(image_width / patch_width)

        images = []
        for i in range(h):
            raw = []
            for j in range(w):
                raw.append(patchs[:, i * h + j, :, :])
            raw = torch.cat(raw, dim=3)
            images.append(raw)
        images = torch.cat(images, dim=2)
        return images

    def forward(self, x):
        num_patchs = x.size()[1] - 1
        cls_tokens, x = torch.split(x, [1, num_patchs], dim=1)

        x = self.latent2patchv(x)
        x = self.patchv2patch(x)
        x = self.reconstruct(x)
        x = self.net(x)
        x = self.embedding(x)
        x = x.view(-1, num_patchs, self.dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.fc(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ConvolutionalTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, hidden_channels, patch_size, image_size, cnn_depth,
                 dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, ReconstructionConv(dim=dim, hidden_channels=hidden_channels, patch_size=patch_size,
                                                image_size=image_size, cnn_depth=cnn_depth)),
                PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, rc, ff in self.layers:
            x = attn(x) + x
            identity = x
            x = rc(x)
            x = ff(x) + identity
        return x


class VisionConformer(nn.Module):
    def __init__(
            self,
            *,
            image_size=25,
            patch_size=5,
            num_classes=2,
            dim=256,
            depth=2,
            heads=4,
            mlp_dim=384,
            hidden_channels=512,
            cnn_depth=2,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = Image2Patch_Embedding(patch_size=patch_size, channels=channels, dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConvolutionalTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                    mlp_dim=mlp_dim, dropout=dropout, hidden_channels=hidden_channels,
                                                    patch_size=patch_size, image_size=image_size, cnn_depth=cnn_depth)

        self.pool = pool
        self.to_latent = nn.LayerNorm(dim)

        self.mlp_head = nn.Linear(dim, num_classes)

    def descriptor(self,x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

    def classifier(self,x):
        return self.mlp_head(x)


    def forward(self, x):
        x = self.descriptor(x)
        x = self.classifier(x)
        return x



