import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoder, CLIPVisionEmbeddings, CLIPConfig, BaseModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

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
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid] torch.Size([1024, 19, 29])

        # add
        patch_height, patch_width = patch_embeds.shape[-2:]

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1) #torch.Size([3, 1, 1024])
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

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[list] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch_size = len(pixel_values)

        # add
        max_patches = max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

        hidden_states = []
        for i in range(batch_size):
            hidden_state = self.embeddings(pixel_values=pixel_values[i].unsqueeze(0)) #torch.Size([1, 552, 1024])
            hidden_state = self.pre_layrnorm(hidden_state)
            padding_size = max_patches + 1 - hidden_state.shape[1]
            padding = torch.zeros((1, padding_size, hidden_state.shape[2]), dtype=hidden_state.dtype, device=hidden_state.device) #torch.Size([1, 25, 1024])
            state = torch.cat([hidden_state, padding], dim=1)
            hidden_states.append(state)

        hidden_states = torch.cat(hidden_states, dim=0)
        
        patch_attention_mask = torch.zeros((batch_size, 1, max_patches + 1), dtype=torch.bool, device=hidden_states.device)#torch.Size([10, 577])
        for i in range(batch_size):
            patch_attention_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1] + 1] = True

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            attention_mask=None
        else:
            attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype) #torch.Size([10, 1, 577, 577])

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #FIXME the pooled_output here is incorrect for post_layernorm on padded features
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class AdaptCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = AdaptCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[list] = None,
        tgt_sizes: Optional[torch.IntTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            tgt_sizes=tgt_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )