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
from einops import rearrange
import time
from llava.model.multimodal_encoder.hubconf import featup

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


class PerceiverResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            config,
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
        self.config = config
        self.feature_scale_mask =  getattr(self.config, 'feature_scale_mask', 7)

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        
        vision_tower =  getattr(self.config, 'mm_vision_tower', '')
        self.vision_tower_name = 'clip-large'
        if 'clip' in vision_tower:
            self.vision_tower_name = 'clip-large'
        elif 'siglip' in vision_tower:
            self.vision_tower_name = 'siglip'

        if self.feature_scale_mask & 8:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='8x')
        elif self.feature_scale_mask & 4:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='4x')
        elif self.feature_scale_mask & 2:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='2x')

        
        # Latent tokens to be used in attention
        self.latent_tokens = nn.Parameter(torch.randn(1, 4096, config.mm_hidden_size))  # Shape: (1, latent_dim, input_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_proj = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(config.mm_hidden_size, embed_dim))

        self.cat_proj = nn.Linear(4*embed_dim, embed_dim)
        self.apply(self._init_weights)

        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=2, stride=2)

        # Feed-forward layers (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mm_hidden_size, config.hidden_size)
        )



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def prepare_resized_feat(self, feature_1x, feature_2x, feature_4x, feature_8x, tgt_size=(24, 24), attn_mask=None, dtype=torch.bfloat16):
        #torch.Size([1, 1024, 25, 22])
        feature_list = [feature_1x, feature_2x, feature_4x, feature_8x]


        resized_feat_list = []

        for idx in range(len(feature_list)):
            if feature_list[idx] is None:
                continue
            feature = feature_list[idx].to(torch.bfloat16)

            scale = 2 ** idx

            feature = F.interpolate(feature, size=(24*scale, 24*scale), mode='bilinear', align_corners=False)
            #torch.Size([1, 1024, 24, 24])
            feature = feature.reshape(feature.shape[0],feature.shape[1], -1)
            feature = feature.permute(0, 2, 1)
            feature = self.kv_proj(feature)
            resized_feat_list.append(feature)
            

        resized_feat = torch.cat(resized_feat_list, dim=1)
        

        x = resized_feat

        x = self.ln_kv(x).permute(1, 0, 2) #torch.Size([432, 1, 3584]) torch.Size([12096, 1, 3584])

        N = x.shape[1]
        q = self.ln_q(self.query) # Q * D torch.Size([144, 3584])
        
        out = self.attn(
            self._repeat(q, N), # Q * N * D
            x,
            x)[0]
        
        x = out.permute(1, 0, 2)[0]
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(N, 1, 1)
    
    def partition_list(self, input_list, lengths):
        """
        按照指定的长度划分列表。

        参数:
        input_list (list): 要划分的原始列表。
        lengths (list): 一个包含划分长度的整数列表。

        返回:
        list: 一个包含子列表的列表，每个子列表的长度由 lengths 指定。
        """
        result = []
        current_index = 0
        for length in lengths:
            if current_index + length > len(input_list):
                raise ValueError("划分长度超过了列表的总长度")
            sublist = input_list[current_index:current_index + length]
            result.append(sublist)
            current_index += length
        if current_index != len(input_list):
            raise ValueError("划分长度和列表总长度不一致")
        return result
    
    def forward_with_featup(self, features, patch_sizes, images, num_images):
        # achor2 = time.time() - start  #0.38
        # print(f'achor2: {achor2 - achor1}')

        bs = len(images)

        features_1x = [] #list torch.Size([1, 1024, 25, 22])

        for i in range(len(features)):
            h, w = patch_sizes[i]

            if type(features) is list:
                feature = features[i][:h * w, :]
            else:
                feature = features[i][:h * w, :].unsqueeze(0)
            feature = feature.permute(0, 2, 1)  #torch.Size([1, 1024, 25*22])
            feature = feature.unflatten(2, [h, w])  #torch.Size([1, 1024, 25, 22])
            features_1x.append(feature)

        feature_scale_mask =  getattr(self.config, 'feature_scale_mask', 7)
        features_2x, features_4x, features_8x = self.upsampler.forward_with_internal_features(images, features_1x)
        
        if feature_scale_mask & 1 == 0:
            features_1x = []
        if feature_scale_mask & 2 == 0:
            features_2x = []
        if feature_scale_mask & 4 == 0:
            features_4x = []
        if feature_scale_mask & 8 == 0:
            features_8x = []
        
        projected_image_features = []
        
        def get_element(features, index):
            if len(features) == 0:
                return None
            return features[index]
        
        resized_feat_list = []
        for i in range(len(patch_sizes)):
            resized_feat = self.prepare_resized_feat(
                get_element(features_1x, i), 
                get_element(features_2x, i), 
                get_element(features_4x, i), 
                get_element(features_8x, i),
                get_element(patch_sizes, i)) # (144, 36, 5120)
            resized_feat_list.append(resized_feat)
        #[(144,3582)]
        projected_image_features = self.partition_list(resized_feat_list, num_images)
        return projected_image_features