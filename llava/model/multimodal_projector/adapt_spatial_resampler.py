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


class AdaptSpatialResampler(nn.Module):
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
        self.config = config
        self.grid_size = grid_size
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mm_hidden_size = self.config.mm_hidden_size
        self.feature_scale_mask =  getattr(self.config, 'feature_scale_mask', 7)
        vision_tower =  getattr(self.config, 'mm_vision_tower', '')
        self.vision_tower_name = 'clip-large'
        if 'clip' in vision_tower:
            self.vision_tower_name = 'clip-large'
        elif 'siglip' in vision_tower:
            self.vision_tower_name = 'siglip'

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        
        if self.feature_scale_mask & 8:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='8x')
        elif self.feature_scale_mask & 4:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='4x')
        elif self.feature_scale_mask & 2:
            self.upsampler = featup(self.vision_tower_name, pretrained=False, use_norm=True, scale='2x')

        # four learnable expert embeddings
        self.feature_1x_embedding = nn.Parameter(torch.zeros(1,1, self.embed_dim))
        self.feature_2x_embedding = nn.Parameter(torch.zeros(1,1, self.embed_dim))
        self.feature_4x_embedding = nn.Parameter(torch.zeros(1,1, self.embed_dim))
        self.feature_8x_embedding = nn.Parameter(torch.zeros(1,1, self.embed_dim))

        # It is a 144 diverse embedding, not 
        self.query_1 = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query_1, std=.02)
            
        self.query_2 = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query_2, std=.02)
            
        self.query_3 = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query_3, std=.02)
            
        self.query_4 = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query_4, std=.02)

        self.features_1x_projector = nn.Linear(in_features=self.mm_hidden_size, out_features=self.embed_dim)
        self.features_2x_projector = nn.Linear(in_features=self.mm_hidden_size, out_features=self.embed_dim)
        self.features_4x_projector = nn.Linear(in_features=self.mm_hidden_size, out_features=self.embed_dim)
        self.features_8x_projector = nn.Linear(in_features=self.mm_hidden_size, out_features=self.embed_dim)
                    
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_proj = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.cat_proj = nn.Linear(4*embed_dim, embed_dim)
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

    def adapt_unfold(self, input_embeds, spatial_size=(24, 24), best_grid=(1, 1), sampler_bins=1):
        # input_embeds: bs, n, c
        # spatial_size: feature map height, width
        # sampler_bins越大，采样点越多，细节越多
        input_embeds = input_embeds.permute(0, 3,1,2)

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

    def forward_with_muti_res(self, feature_1x, feature_2x, feature_4x, feature_8x, tgt_size=(24, 24), attn_mask=None, dtype=torch.bfloat16):
        """Prepare KV in a 4*9 manner"""
        muti_res_feat_keys = []
        muti_res_feat_values = []
        bs = 1

        feature_list = [feature_1x, feature_2x, feature_4x, feature_8x]
        embedding_list =  [self.feature_1x_embedding, self.feature_2x_embedding, self.feature_4x_embedding, self.feature_8x_embedding]
        projector_list = [self.features_1x_projector, self.features_2x_projector, self.features_4x_projector, self.features_8x_projector]

        for feature, embedding, projector in zip(feature_list, embedding_list, projector_list):
            if feature is None:
                continue
            
            feature = feature.to(torch.bfloat16)
            feature = projector(feature.permute(0,2,3,1))
            
            key_height = feature.shape[1]
            key_width = feature.shape[2]
            key_pos_embed = get_abs_pos(self.pos_embed, (key_height, key_width)) #torch.Size([550, 4096])
            feature = rearrange(feature,'b h w c -> b (h w) c')  #torch.Size([1, 50, 44, 4096]) to torch.Size([1, 2200, 4096])
            feature = self.ln_kv(feature) #torch.Size([1, 2304, 4096]) #torch.Size([1, 9216, 4096])
            key = feature + key_pos_embed[None].to(dtype=dtype) + embedding.to(dtype=dtype) 
            value = feature
            key = key.reshape(bs, key_height, key_width, self.embed_dim) #torch.Size([1, 48, 48, 4096]) #torch.Size([1, 96, 96, 4096])
            key = self.adapt_unfold(key) #torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            value = value.reshape(bs, key_height, key_width, self.embed_dim)
            value = self.adapt_unfold(value)# torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            muti_res_feat_keys.append(key) #torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            muti_res_feat_values.append(value)

        muti_res_feat_keys = torch.cat(muti_res_feat_keys, dim=1) # (144, 36, 5120)
        muti_res_feat_values = torch.cat(muti_res_feat_values, dim=1) # (144, 36, 5120)

        # achor2 = time.time() - start  #0.38
        # print(f'kv: {achor2}')

        """Prepare Q and do attn"""
        attn_results = []
        for query_now in [self.query_1, self.query_2, self.query_3, self.query_4]:
            q = self.ln_q(query_now)
            query = self._repeat(q, bs) + self.pos_embed[None].to(dtype=dtype)
            query = self.unfold(query, spatial_size=(self.grid_size, self.grid_size), kernel_size=1, stride=1)
            
            out, attn_weights = self.attn(
                query.permute(1, 0, 2),  #torch.Size([1, 144, 4096])                #Q * B * D
                muti_res_feat_keys.permute(1, 0, 2),  #torch.Size([18, 144, 4096])  #L * B * D
                muti_res_feat_values.permute(1, 0, 2),  #torch.Size([18, 144, 4096])
                attn_mask=attn_mask
            )
            #out : torch.Size([1, 144, 4096])
            # out->1, bs*l, c
            get = out[0].unflatten(0, [bs, -1]) # bs, l, c  #torch.Size([1, 144, 4096])
            get = self.ln_proj(get)
            attn_results.append(get)

        x = torch.cat(attn_results, dim=2)  #torch.Size([1, 144, 16384])
        x = self.cat_proj(x)  #torch.Size([1, 144, 4096])
        x = self.ln_post(x)  #torch.Size([1, 144, 4096])
        x = x @ self.proj #torch.Size([1, 144, 4096])
        
        # achor3 = time.time() - start  #0.38
        # print(f'query: {achor3 - achor2}')

        return x
    
    def prepare_single_key_value(self, feature_1x, feature_2x, feature_4x, feature_8x, tgt_size=(24, 24), attn_mask=None, dtype=torch.bfloat16):
        """Prepare KV in a 4*9 manner"""
        muti_res_feat_keys = []
        muti_res_feat_values = []
        bs = feature_1x.shape[0]

        feature_list = [feature_1x, feature_2x, feature_4x, feature_8x]
        embedding_list =  [self.feature_1x_embedding, self.feature_2x_embedding, self.feature_4x_embedding, self.feature_8x_embedding]
        projector_list = [self.features_1x_projector, self.features_2x_projector, self.features_4x_projector, self.features_8x_projector]

        for feature, embedding, projector in zip(feature_list, embedding_list, projector_list):
            if feature is None:
                continue
            
            feature = feature.to(torch.bfloat16)
            feature = projector(feature.permute(0,2,3,1))
            
            key_height = feature.shape[1]
            key_width = feature.shape[2]
            key_pos_embed = get_abs_pos(self.pos_embed, (key_height, key_width)) #torch.Size([550, 4096])
            feature = rearrange(feature,'b h w c -> b (h w) c')  #torch.Size([1, 50, 44, 4096]) to torch.Size([1, 2200, 4096])
            feature = self.ln_kv(feature) #torch.Size([1, 2304, 4096]) #torch.Size([1, 9216, 4096])
            key = feature + key_pos_embed[None].to(dtype=dtype) + embedding.to(dtype=dtype) 
            value = feature
            key = key.reshape(bs, key_height, key_width, self.embed_dim) #torch.Size([1, 48, 48, 4096]) #torch.Size([1, 96, 96, 4096])
            key = self.adapt_unfold(key) #torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            value = value.reshape(bs, key_height, key_width, self.embed_dim)
            value = self.adapt_unfold(value)# torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            muti_res_feat_keys.append(key) #torch.Size([144, 9, 4096])  #torch.Size([144, 9, 4096])
            muti_res_feat_values.append(value)

        muti_res_feat_keys = torch.cat(muti_res_feat_keys, dim=1) # (144, 36, 5120)
        muti_res_feat_values = torch.cat(muti_res_feat_values, dim=1) # (144, 36, 5120)

        return muti_res_feat_keys, muti_res_feat_values

    def query_with_parallel_attn(self, bs, key_list, value_list, dtype=torch.bfloat16):
        """Prepare Q and do attn"""
        max_len = max([key.shape[1] for key in key_list]) #36
        tgt_lengths = []

        for i in range(len(key_list)):
            for _ in range(key_list[i].shape[0] // self.num_queries):
                tgt_lengths.append(key_list[i][0].shape[0])
        
        padded_key_list = []
        for key in key_list:
            padding_size = max_len - key.shape[1]
            padding = torch.zeros((key.shape[0], padding_size, key.shape[2]), dtype=key.dtype, device=key.device) #torch.Size([144, 36, 1024])
            padded_key = torch.cat([key, padding], dim=1)
            padded_key_list.append(padded_key)

        padded_value_list = []
        for value in value_list:
            padding_size = max_len - value.shape[1]
            padding = torch.zeros((value.shape[0], padding_size, value.shape[2]), dtype=value.dtype, device=value.device) #torch.Size([144, 36, 1024])
            padded_value = torch.cat([value, padding], dim=1)
            padded_value_list.append(padded_value)

        padded_keys = torch.cat(padded_key_list, dim=0) # torch.Size([1440, 36, 4096])
        padded_values = torch.cat(padded_value_list, dim=0) # torch.Size([1440, 36, 4096])

        token_length = int(padded_keys.shape[0] / bs)  #144
        key_padding_mask = torch.ones((padded_keys.shape[0], max_len), dtype=torch.bool, device=key_list[0].device) #torch.Size([1440, 36])
        for i in range(bs):
            key_padding_mask[i*token_length : (i+1)*token_length, :tgt_lengths[i]] = False  

        attn_results = []
        for query_now in [self.query_1, self.query_2, self.query_3, self.query_4]:
            q = self.ln_q(query_now)
            query = self._repeat(q, bs) + self.pos_embed[None].to(dtype=dtype)
            query = self.unfold(query, spatial_size=(self.grid_size, self.grid_size), kernel_size=1, stride=1) #torch.Size([1440, 1, 4096])
            
            out, attn_weights = self.attn(      #[1, 1008, 4096]
                query.permute(1, 0, 2),         #torch.Size([1, 1008, 4096])
                padded_keys.permute(1, 0, 2),   #torch.Size([36, 1008, 4096])
                padded_values.permute(1, 0, 2),  #torch.Size([36, 1008, 4096])
                key_padding_mask=key_padding_mask
            )
            # out->1, bs*l, c
            get = out[0].unflatten(0, [bs, -1]) # bs, l, c  #torch.Size([7, 144, 4096])
            get = self.ln_proj(get)
            attn_results.append(get)

        x = torch.cat(attn_results, dim=2)  #torch.Size([7, 144, 16384])
        x = self.cat_proj(x)  #torch.Size([7, 144, 4096])
        x = self.ln_post(x)  #torch.Size([7, 144, 4096])
        x = x @ self.proj #torch.Size([7, 144, 4096])
        
        projected_image_features = [x[i] for i in range(bs)]

        return projected_image_features

    def _repeat(self, query, N: int):
        return query.unsqueeze(0).repeat(N, 1, 1)
    
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

        # 对features_1x 中的元素分组，如果连续且 shape 相同，则分为一组
        keys = []
        values = []
        feat_group = []
        image_group = []
        for i in range(len(features_1x)):
            if i == 0:
                feat_group.append(features_1x[i])
                image_group.append(images[i])
            elif(features_1x[i].shape != features_1x[i-1].shape):
                key, value = self.get_group_keys(feat_group, image_group)
                keys.append(key)
                values.append(value)

                feat_group = []
                image_group = []
                feat_group.append(features_1x[i])
                image_group.append(images[i])
            else:
                feat_group.append(features_1x[i])
                image_group.append(images[i])
            
        key, value = self.get_group_keys(feat_group, image_group)
        keys.append(key)
        values.append(value)

        return self.compute_atten(bs, keys, values, num_images)

    def get_group_keys(self, features_1x, image_group):
        features_1x = torch.cat(features_1x, dim=0)
        image_group = torch.stack(image_group, dim=0)

        features_2x, features_4x, features_8x = self.upsampler.forward_with_internal_features(image_group, features_1x)
        
        if self.feature_scale_mask & 1 == 0:
            features_1x = None
        if self.feature_scale_mask & 2 == 0:
            features_2x = None
        if self.feature_scale_mask & 4 == 0:
            features_4x = None
        if self.feature_scale_mask & 8 == 0:
            features_8x = None
        
        return self.prepare_single_key_value(features_1x, features_2x, features_4x, features_8x)
    
    def compute_atten(self, bs, key_list, value_list, num_images):
        projected_image_features = self.query_with_parallel_attn(bs, key_list, value_list)
        projected_image_features = self.partition_list(projected_image_features, num_images)
        return projected_image_features