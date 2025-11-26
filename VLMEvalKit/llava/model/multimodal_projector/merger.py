import torch
import torch.nn as nn
from torch.nn import functional as F 
from functools import partial

import numpy as np
import math
import torchvision.ops.roi_align as RoIAlign
from einops import rearrange

class Qwen2vlPatchMerger(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_embed_dim=1024,
        compression_factor=(2,2),
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embed_dim = image_embed_dim
        self.hidden_size = image_embed_dim * (compression_factor[0]*compression_factor[1])
        self.nl = norm_layer(image_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, embed_dim),
        )
        self.compression_factor = compression_factor


    def forward(self, x, tgt_size=(24,24), attn_mask=None):
        
        x = x.to(torch.bfloat16)
        dtype = x.dtype
        height, width = tgt_size

        if height * width != x.shape[1]:
            x = x[:, :height * width]
        x = self.nl(x)

        x = x.permute(0, 2, 1).unflatten(-1, (height, width))  # b, dim, h, w
        batch_size, dim, height, width = x.shape
        # 计算输出空间的高度和宽度
        h_compressed = (height + self.compression_factor[0] - 1) // self.compression_factor[0]
        w_compressed = (width + self.compression_factor[1] - 1) // self.compression_factor[1]
        
        unfolded = x.unfold(2, self.compression_factor[0], self.compression_factor[0]).unfold(3, self.compression_factor[1], self.compression_factor[1])
        unfolded = unfolded.contiguous().view(batch_size, dim, -1, self.compression_factor[0] * self.compression_factor[1])
        unfolded = unfolded.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, dim*self.compression_factor[0] * self.compression_factor[1]) 
        compressed_x = self.mlp(unfolded)
        return compressed_x

class Qwen2_5vlInvalid(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, tgt_size):
        x = x.unsqueeze(0)
        return x

class Qwen2vlPatchMergerFused(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_embed_dim=1024,
        compression_factor=(2,2),
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embed_dim = image_embed_dim
        self.hidden_size = image_embed_dim * (compression_factor[0]*compression_factor[1])
        self.nl = norm_layer(image_embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(3 * kv_dim, image_embed_dim),
            nn.GELU(),
            nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, embed_dim),
        )
        self.compression_factor = compression_factor


    def forward(self, x, tgt_size=(24,24), attn_mask=None):
        
        image_features, fused_features = x
        image_features = image_features[:, :tgt_size[0] * tgt_size[1], :]

        image_features = torch.cat((image_features, fused_features), dim=0)
        image_features = rearrange(image_features, 'm n d -> n (m d)')

        image_features = self.proj(image_features).unsqueeze(0)

        image_features = image_features.to(torch.bfloat16)
        dtype = image_features.dtype
        height, width = tgt_size
        image_features = self.nl(image_features)

        image_features = image_features.permute(0, 2, 1).unflatten(-1, (height, width))  # b, dim, h, w
        batch_size, dim, height, width = image_features.shape
        # 计算输出空间的高度和宽度
        h_compressed = (height + self.compression_factor[0] - 1) // self.compression_factor[0]
        w_compressed = (width + self.compression_factor[1] - 1) // self.compression_factor[1]
        
        unfolded = image_features.unfold(2, self.compression_factor[0], self.compression_factor[0]).unfold(3, self.compression_factor[1], self.compression_factor[1])
        unfolded = unfolded.contiguous().view(batch_size, dim, -1, self.compression_factor[0] * self.compression_factor[1])
        unfolded = unfolded.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, dim*self.compression_factor[0] * self.compression_factor[1]) 
        compressed_x = self.mlp(unfolded)
        return compressed_x