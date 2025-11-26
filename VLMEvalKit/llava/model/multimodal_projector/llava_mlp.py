# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from einops import rearrange




class LLaVA_MLP(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            config,
            embed_dim,
            kv_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config

        self.proj = nn.Sequential(
            nn.Linear(kv_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x, tgt_size=(24, 24)):
        x = x[:, :tgt_size[0] * tgt_size[1], :]
        return self.proj(x)

class LLaVA_AvgPooling_MLP(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            config,
            embed_dim,
            merge_size=2,
            kv_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config

        self.proj = nn.Sequential(
            nn.Linear(kv_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.merge_size = merge_size
    
    def forward(self, x, tgt_size=(24, 24)):
        x = x[:, :tgt_size[0] * tgt_size[1], :][0]
        x = rearrange(x, '(h w) c -> c h w' , h=tgt_size[0], w=tgt_size[1])
        x = F.avg_pool2d(x, kernel_size=self.merge_size, stride=self.merge_size)  # nxn 池化
        x = rearrange(x, 'c h w -> (h w) c')  # 重新展平
        x = x.unsqueeze(0)
        x = self.proj(x)
        x = self.norm(x)
        return x

class LLaVA_MLP_norm(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            config,
            embed_dim,
            kv_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config

        self.proj = nn.Sequential(
            nn.Linear(kv_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, x, tgt_size=(24, 24)):
        x = x[:, :tgt_size[0] * tgt_size[1], :]
        x = self.proj(x)
        x = self.norm(x)
        return x

class LLaVA_MLP_Fused(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            config,
            embed_dim,
            kv_dim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config
        self.proj = nn.Sequential(
            nn.Linear(3 * kv_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x, tgt_size=(24, 24)):
        image_features, fused_features = x
        image_features = image_features[:, :tgt_size[0] * tgt_size[1], :]
        image_features = torch.cat((image_features, fused_features), dim=0)
        image_features = rearrange(image_features, 'm n d -> n (m d)')
        return self.proj(image_features).unsqueeze(0)