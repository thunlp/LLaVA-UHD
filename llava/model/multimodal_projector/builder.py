import torch
import torch.nn as nn
import re
import math
from .adapt_spatial_resampler import AdaptSpatialResampler
from .uhd_v1_resampler import AdaptSpatialResampler_v1

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):

    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == 'adapt_spatial_resampler':
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
    
    raise ValueError(f'Unknown projector type: {projector_type}')


