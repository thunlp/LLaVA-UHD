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

from llava.slice_process import slice_image_feature_minicpm
import torchvision.ops.roi_align as RoIAlign

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype

    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AdaptSpatialResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_best_pooling_size(self, feature_wh_ratio=1.0):
        candidate_pooling_sizes = [
            (4, 2), (3, 2), (4, 3), (3, 3), 
            (2, 4), (2, 3), (3, 4)
        ] # w, h
        log_feature_wh_ratio = math.log(feature_wh_ratio)
        best_pooling_size = (3, 3) # w, h
        min_error = float("inf")
        for candidate_pooling_size in candidate_pooling_sizes:
            w, h = candidate_pooling_size
            error = abs(log_feature_wh_ratio - math.log(w/h))
            if error < min_error:
                best_pooling_size = (h, w)
                min_error = error
        return best_pooling_size

    def adapt_unflod(self, input_embeds, spatial_size=(24, 24), best_grid=(1, 1), sampler_bins=1):
        # input_embeds: bs, n, c
        # spatial_size: feature map height, width
        # sampler_bins越大，采样点越多，细节越多
        input_embeds = input_embeds.permute(0, 2, 1).unflatten(-1, spatial_size)
        resample_regions, best_grid, wh_ratio = slice_image_feature_minicpm(input_embeds, self.num_queries)

        output_size = self.cal_best_pooling_size(wh_ratio)
        aligned_feature = RoIAlign(input_embeds.float(), resample_regions.float(), output_size, 
                                    spatial_scale=1.0).to(dtype=input_embeds.dtype)
        unfold_input_embeds = aligned_feature.flatten(-2).permute(0, 2, 1)
        # bs*N, c, h, w -> bs*N,c,h*w -> bs*N, h*w, c
        return unfold_input_embeds

    def unfold(self, input_embeds, spatial_size=(24, 24), kernel_size=2, stride=2):
        # input_embeds: bs, n, c
        # spatial_size: feature map height, width
        input_embeds = input_embeds.permute(0, 2, 1).unflatten(-1, spatial_size)
        unfold_func = nn.Unfold(kernel_size=kernel_size, stride=stride)
        unfold_input_embeds = unfold_func(input_embeds) # bs, c* k**2, l
        unfold_input_embeds = unfold_input_embeds.unflatten(1, [-1, kernel_size ** 2]).permute(0, 3, 2, 1).flatten(0, 1)
        # bs, c*k**2, l -> bs, c, k**2, l -> bs, l, k**2, c -> bs*l, k**2, c
        return unfold_input_embeds

    def forward(self, x, tgt_size=(24, 24), attn_mask=None):
        dtype = x.dtype
        bs = x.shape[0]
        key_height, key_width = tgt_size
        key_pos_embed = get_abs_pos(self.pos_embed, (key_height, key_width))


        x = self.ln_kv(self.kv_proj(x))

        q = self.ln_q(self.query) #[:num_valid_query]


        query = self._repeat(q, bs) + self.pos_embed[None].to(dtype=dtype)
        key = x + key_pos_embed[None].to(dtype=dtype)
        value = x

        query = self.unfold(query, spatial_size=(self.grid_size, self.grid_size), kernel_size=1, stride=1)
        key = self.adapt_unflod(key, spatial_size=(key_height, key_width))
        value = self.adapt_unflod(value, spatial_size=(key_height, key_width))

        out, attn_weights = self.attn(
            query.permute(1, 0, 2),
            key.permute(1, 0, 2),
            value.permute(1, 0, 2),
            attn_mask=attn_mask
        )
        # out->1, bs*l, c
        x = out[0].unflatten(0, [bs, -1]) # bs, l, c
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(0).repeat(N, 1, 1)

