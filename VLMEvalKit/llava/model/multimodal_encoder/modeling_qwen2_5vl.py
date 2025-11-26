from transformers import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from functools import partial, reduce
from typing import Any, Optional, Tuple, Union, Dict
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

class QwenVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"
    
    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range

class QwenImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(392, 392), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 392, "width": 392}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, do_resize = True, do_center_crop = True, do_rescale = True, do_normalize = True, return_tensors = 'pt'):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        # do_resize=False, do_center_crop=False, do_rescale=True, do_normalize=True, 

        transforms = [
            convert_to_rgb,
            to_numpy_array
        ]

        if do_resize:
            transforms.append(partial(resize, size=self.size, resample=self.resample, data_format=self.data_format))
        if do_rescale:
            transforms.append(partial(rescale, scale=self.rescale_factor, data_format=self.data_format))
        if do_normalize:
            transforms.append(partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format))
        
        transforms.append(partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format))

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

class Qwen2_5VLVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = QwenVisionConfig()  ### 需要定义

        self.vision_tower_name = vision_tower

        self.image_processor = QwenImageProcessor()

        if not delay_load:
            print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()

        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        
        else:
            self.cfg_only = self.config
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        self.vision_tower = Qwen2_5_VisionTransformerPretrainedModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        print('qwen2_5vl vision tower loaded')
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
    
    def forward(self, images, patch_sizes=None):
        if type(images) is list:
            pixel_values = []
            vision_grid_thws = []
            spatial_patch_size = self.vision_tower.config.spatial_patch_size
            temporal_patch_size = self.vision_tower.config.temporal_patch_size
            spatial_merge_size = 2
            data = {}
            for image in images:
                image = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                image = torch.cat([image, image], dim=0)   ### t, c, h, w
                grid_t = image.shape[0] // temporal_patch_size
                grid_h, grid_w = image.shape[2] // spatial_patch_size, image.shape[3] // spatial_patch_size
                channel = image.shape[1]
                patches = image.reshape(grid_t, temporal_patch_size, channel, 
                                          grid_h // spatial_merge_size, spatial_merge_size, spatial_patch_size, 
                                          grid_w // spatial_merge_size, spatial_merge_size, spatial_patch_size)
                patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
                flatten_patches = patches.reshape(
                    grid_t * grid_h * grid_w, 
                    channel * temporal_patch_size * spatial_patch_size * spatial_patch_size
                )

                pixel_values.extend(flatten_patches)
                vision_grid_thws.append(torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0))
            pixel_values = torch.stack(pixel_values, dim=0)
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            vision_grid_thws = torch.cat(vision_grid_thws, dim=0).to(device=self.device)
            image_embeds = self.vision_tower(pixel_values, grid_thw=vision_grid_thws)
            split_sizes = (vision_grid_thws.prod(-1) // spatial_merge_size**2).tolist()
            image_features = torch.split(image_embeds, split_sizes)
        else: 
            print('no support for parallel processing')
            exit()
        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size