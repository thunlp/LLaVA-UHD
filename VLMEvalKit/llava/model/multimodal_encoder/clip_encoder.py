import torch
import torch.nn as nn
import os
from safetensors import safe_open
from llava.utils import rank0_print
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from llava.model.multimodal_encoder.adapt_clip_vision_model import AdaptCLIPVisionModel
try:
    from s2wrapper import forward as multiscale_forward
except:
    pass

def load_vision_tower_values(model_path, device):
    """
    在给定的路径下查找所有 `.safetensors` 文件，加载它们，并返回 key 中包含 `vision_tower` 的权重值。

    参数:
    - model_path (str): Hugging Face 模型文件夹的路径。

    返回:
    - vision_tower_values (dict): 包含所有 `vision_tower` 相关的键和值的字典。
    """
    # 找到路径中的所有 `.safetensors` 文件
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    
    vision_tower_values = {}

    # 遍历每个 `.safetensors` 文件
    for safetensor_file in safetensor_files:
        safetensor_path = os.path.join(model_path, safetensor_file)
        
        # 使用 safetensors 库打开并读取文件内容
        with safe_open(safetensor_path, framework="pt", device=str(device)) as f:
            for key in f.keys():
                # 如果 key 中包含 `vision_tower`，将其加入结果字典
                if 'vision_tower' in key:
                    key_new = key.replace('model.vision_tower.vision_tower.', '')
                    vision_tower_values[key_new] = f.get_tensor(key)

    return vision_tower_values
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None, model_path=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        #self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        print('---------init adapt_vision_model---------')
        self.vision_tower = AdaptCLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        if model_path is None:
            print('---------from frozen ckpt---------')
        else:
            print('---------from ft ckpt---------')
            vision_tower_values = load_vision_tower_values(model_path, self.vision_tower.device)
            load_info = self.vision_tower.load_state_dict(vision_tower_values, strict=False)
            print(f'load info: {load_info}')

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images, patch_sizes):
        tgt_sizes = torch.tensor(patch_sizes, dtype=torch.long, device=images[0].device)

        #FIXME the pooled_output here is incorrect for post_layernorm on padded features
        image_forward_outs = self.vision_tower(images, tgt_sizes=tgt_sizes, output_hidden_states=True)
        features = self.feature_select(image_forward_outs).to(images[0].dtype)

        image_features = [] #list torch.Size([1, 1024, 25, 22])
        for i in range(len(features)):
            h, w = patch_sizes[i]
            feature = features[i][:h * w, :].unsqueeze(0)
            # feature = feature.permute(0, 2, 1)  #torch.Size([1, 1024, 25*22])
            # feature = feature.unflatten(2, [h, w])  #torch.Size([1, 1024, 25, 22])
            image_features.append(feature)

        return image_features

    def forward_uhd_v2(self, images, tgt_sizes): 
        #FIXME the pooled_output here is incorrect for post_layernorm on padded features
        image_forward_outs = self.vision_tower(images, tgt_sizes=tgt_sizes, output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images[0].dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
