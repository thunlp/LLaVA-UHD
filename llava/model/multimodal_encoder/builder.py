import os
from .modeling_siglip2 import SigLip2VisionTower
from .modeling_moonvit import MoonViTVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if "siglip2" in vision_tower:
        return SigLip2VisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif "moonvit" in vision_tower:
        return MoonViTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
