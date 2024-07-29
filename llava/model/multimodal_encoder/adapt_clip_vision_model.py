import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union



from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoder, CLIPVisionEmbeddings


class AdaptCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def resize_pos_embedding(self, position_embedding, dst_size=(24, 24), square_size=24):
        _dtype = position_embedding.dtype
        patch_height, patch_width = dst_size
        class_position_embedding = position_embedding[:, :1] # 1, 1, c
        patch_position_embedding = position_embedding[:, 1:] # 1, 576, c

        patch_position_embedding = patch_position_embedding.permute(0, 2, 1).unflatten(-1, [square_size, square_size])
        patch_position_embedding = torch.nn.functional.interpolate(
            patch_position_embedding, size=(patch_height, patch_width), mode='bicubic'
        ).to(dtype=_dtype) # 1, c, ph, pw
        patch_position_embedding = patch_position_embedding.flatten(-2).permute(0, 2, 1) # 1, n, c
        position_embedding = torch.cat([class_position_embedding, patch_position_embedding], dim=1) # 1, n+1, c
        return position_embedding


    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]

        # add
        patch_height, patch_width = patch_embeds.shape[-2:]

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # embeddings = embeddings + self.position_embedding(self.position_ids)

        # add
        square_size = self.config.image_size // self.config.patch_size
        if patch_height == square_size and patch_width == square_size:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        else:
            position_embedding = self.position_embedding(self.position_ids)
            position_embedding = self.resize_pos_embedding(position_embedding, dst_size=(patch_height, patch_width), square_size=square_size)
            embeddings = embeddings + position_embedding
        return embeddings

class AdaptCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = AdaptCLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

class AdaptCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = AdaptCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
