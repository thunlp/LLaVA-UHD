import torch
import torch.nn as nn
import re
import math

from .pooler_projector import PoolerProjector
from .adapt_spatial_resampler import AdaptSpatialResampler
from .uhd_v1_resampler import AdaptSpatialResampler_v1
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

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == 'adapt_spatial_resampler_v2':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))

        resampler = AdaptSpatialResampler(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    if projector_type == 'adapt_spatial_resampler_v2_64':
        target_sequence_length = 64
        grid_size = int(math.sqrt(target_sequence_length))

        resampler = AdaptSpatialResampler(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    if projector_type == 'adapt_spatial_resampler_v2_256':
        target_sequence_length = 256
        grid_size = int(math.sqrt(target_sequence_length))

        resampler = AdaptSpatialResampler(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    if projector_type == 'adapt_spatial_resampler_v1':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = AdaptSpatialResampler_v1(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    if projector_type == 'uhd_v1_query64':
        target_sequence_length = 64
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = AdaptSpatialResampler_v1(
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size,
        )
        return resampler
    
    if projector_type == 'mlp144':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = MLP(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    if projector_type == 'mlp-v2':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = MLP_v2(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    

    if projector_type == 'percive_sampler':
        target_sequence_length = 144
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = PerceiverResampler(
            config=config,
            grid_size=grid_size,
            embed_dim = config.hidden_size,
            num_heads = config.hidden_size // 128,
            kv_dim=config.mm_hidden_size
        )
        return resampler
    
    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
