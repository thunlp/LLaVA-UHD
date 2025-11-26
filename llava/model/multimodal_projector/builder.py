import torch
import torch.nn as nn
import re
import math

from .uhd_v1_resampler import AdaptSpatialResampler_v1
from .resampler import Resampler
from .llava_mlp import LLaVA_MLP
from .merger import Qwen2vlPatchMerger, Qwen2_5vlInvalid

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
    
    elif projector_type == 'merger':
        resampler = Qwen2vlPatchMerger(
            embed_dim = config.hidden_size,
            image_embed_dim=config.mm_hidden_size,
            compression_factor=(2, 2),
        )
        return resampler
    
    elif projector_type == 'resampler':
        target_sequence_length = 64
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler(
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
