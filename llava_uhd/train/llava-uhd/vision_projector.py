import torch
import torch.nn as nn
import re
from resampler import Resampler
import math

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
    
    target_sequence_length = 64
    grid_size = int(math.sqrt(target_sequence_length))
    resampler = Resampler(
        grid_size=grid_size,
        embed_dim = 5120,  # 保持与视觉模型输出的 embed_dim 一致
        num_heads = 1024 // 128,  # 保持与视觉模型输出的 num_heads 一致
        kv_dim=1024,  # 保持与视觉模型输出的 kv_dim 一致
    )
    
    return resampler

    raise ValueError(f'Unknown projector type: {projector_type}')
