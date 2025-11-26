import torch
import torch.nn as nn
import re
import math

from .pooler_projector import PoolerProjector
from .adapt_spatial_resampler import AdaptSpatialResampler
from .uhd_v1_resampler import AdaptSpatialResampler_v1
from .resampler import Resampler, Resampler_ln
from .llava_mlp import LLaVA_MLP, LLaVA_MLP_norm, LLaVA_MLP_Fused, LLaVA_AvgPooling_MLP
from .merger import Qwen2vlPatchMerger, Qwen2vlPatchMergerFused, Qwen2_5vlInvalid
from .mlp import MLP
from .mlp_v2 import MLP_v2
from .percive_sampler import PerceiverResampler

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == 'mlp':
        resampler = LLaVA_MLP(
            config=config,
            embed_dim = config.hidden_size,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    elif projector_type == 'mlp_norm':
        resampler = LLaVA_MLP_norm(
            config=config,
            embed_dim = config.hidden_size,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    elif projector_type == 'mlp_fused':
        resampler = LLaVA_MLP_Fused(
            config=config,
            embed_dim = config.hidden_size,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    elif projector_type == 'avg_mlp':
        resampler = LLaVA_AvgPooling_MLP(
            config=config,
            embed_dim = config.hidden_size,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    elif projector_type == 'avg_mlp_4':
        resampler = LLaVA_AvgPooling_MLP(
            config=config,
            embed_dim = config.hidden_size,
            kv_dim=config.mm_hidden_size,
            merge_size=4
        )
        return resampler
    
    elif projector_type == 'merger':
        resampler = Qwen2vlPatchMerger(
            embed_dim = config.hidden_size,
            image_embed_dim=config.mm_hidden_size,
            compression_factor=(2, 2),
        )
        return resampler
    
    elif projector_type == 'qwen2_5vl':
        resampler = Qwen2_5vlInvalid()
        return resampler
    
    elif projector_type == 'merger_fused':
        resampler = Qwen2vlPatchMergerFused(
            embed_dim = config.hidden_size,
            image_embed_dim=config.mm_hidden_size,
            compression_factor=(2, 2),
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    elif projector_type == 'resampler':
        target_sequence_length = 49
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    elif projector_type == 'resampler_256':
        target_sequence_length = 256
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    elif projector_type == 'resampler_256_ln':
        target_sequence_length = 256
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler_ln(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    elif projector_type == 'adapt_spatial_resampler_v1':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = AdaptSpatialResampler_v1(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    elif projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
