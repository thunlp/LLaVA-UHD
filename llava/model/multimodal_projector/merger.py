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
