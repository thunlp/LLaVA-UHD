import math
from copy import deepcopy
from typing import Union, Tuple, Sequence, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import PytorchGELUTanh
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_flash_attn_2_available
from llava.utils import rank0_print

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

"""Image processor class for KimiVL."""

import math
import numpy as np
from PIL import Image
from typing import Optional, Union

import torch
from torchvision.transforms import functional as TF
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType

from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from typing import Any, Optional, Tuple, Union, Dict
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from functools import partial, reduce
from einops import rearrange

class MoonViTImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(400, 400), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 400, "width": 400}
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


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
):
    """Multi-head attention using flash attention 2.
    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].
    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert (
        k_cu_seqlens[-1] == k.shape[0] == v.shape[0]
    ), "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"unsupported dtype {q.dtype} for multihead attn"

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SDPA attention.
    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
    """
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros(
        [1, seq_length, seq_length], device=q.device, dtype=torch.bool
    )
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x, grid_hws) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        out = x + torch.cat(pos_embs)
        return out


class MoonVisionPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ):
        super().__init__()
        assert isinstance(
            patch_size, (int, Sequence)
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, dim=out_dim
        )

    def forward(self, x, grid_hws) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 2): grid height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hws)
        return x

class HiRope2DPosEmb(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis_dict: Dict[int, torch.Tensor] = {}

    def _precompute_freqs_cis(
        self, down_scale_rate: int, device: torch.device
    ) -> torch.Tensor:
        """
        物理坐标：grid 索引 × down_scale_rate ⇒ 原图相对像素坐标
        返回：complex tensor,  shape = (H, W, dim//2)
        """
        H = self.max_height // down_scale_rate
        W = self.max_width // down_scale_rate

        # 物理坐标（float），与原图像素坐标成正比
        y_pos, x_pos = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device),
            indexing="ij"
        )
        y_pos = y_pos.float() * down_scale_rate     # 乘回缩放倍率
        x_pos = x_pos.float() * down_scale_rate

        # 对数频率
        dim_range = torch.arange(0, self.dim, 4, device=device).float()[: self.dim // 4]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))        # [C/4]

        # 角度矩阵：pos × freq
        x_angles = x_pos.unsqueeze(-1) * freqs        # [H, W, C/4]
        y_angles = y_pos.unsqueeze(-1) * freqs        # [H, W, C/4]

        # 复数 cis(θ) = cosθ + i·sinθ
        x_cis = torch.polar(torch.ones_like(x_angles), x_angles)  # [H, W, C/4]
        y_cis = torch.polar(torch.ones_like(y_angles), y_angles)  # [H, W, C/4]

        # interleave：[..., 0] 放 x 轴，[..., 1] 放 y 轴
        freqs_cis = torch.cat([x_cis, y_cis], dim=-1)  # [H, W, C/2]
        return freqs_cis

    def get_freqs_cis(
        self,
        grid_hws: torch.Tensor,          # (t, 2) => [(h1,w1), (h2,w2), ...]
        down_scale_rate: int = 1,
    ) -> torch.Tensor:
        """
        返回：Tensor, shape = (∑_{t}(h*w) , dim//2)
        按 batch 拼接在一起，供注意力层一次性使用
        """
        device = grid_hws.device

        # ① 若该 scale 没算过就先算并缓存
        if down_scale_rate not in self.freqs_cis_dict:
            self.freqs_cis_dict[down_scale_rate] = self._precompute_freqs_cis(
                down_scale_rate, device
            )

        table = self.freqs_cis_dict[down_scale_rate]       # [H, W, dim//2]
        H_max, W_max = table.shape[:2]

        # ② 逐张裁剪再拼接
        shapes = grid_hws.tolist()                         # [[h1,w1], [h2,w2], ...]
        assert all(1 <= h <= H_max and 1 <= w <= W_max for h, w in shapes), (
            shapes, H_max, W_max
        )
        out = torch.cat(
            [table[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes], dim=0
        )
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_height={self.max_height}, "
            f"max_width={self.max_width}, theta_base={self.theta_base}"
        )

class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, down_scale_rate, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        max_height = self.max_height // down_scale_rate
        max_width = self.max_width // down_scale_rate

        N = max_height * max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % max_width
        y_pos = flat_pos // max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(max_height, max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: torch.Tensor, down_scale_rate=1, init_freqs=False) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): grid height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        max_height = self.max_height // down_scale_rate
        max_width = self.max_width // down_scale_rate
        
        if self.freqs_cis is None or init_freqs:
            self.freqs_cis = self._precompute_freqs_cis(down_scale_rate, grid_hws.device)

        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= max_height and 1 <= w <= max_width for h, w in shapes
        ), (
            shapes,
            max_height,
            max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)

###### Merger layer ######
class PatchMergingLayer(nn.Module):
    def __init__(self, embed_dim, enable_merging=True, merging_method="avg_pooling", norm_layer=nn.LayerNorm):
        """
        :param embed_dim: Transformer token 的嵌入维度
        :param enable_merging: 是否启用 token 合并功能
        :param merging_method: 选择 'mlp' 或 'avg_pooling' 作为合并方式
        """
        super().__init__()
        self.enable_merging = enable_merging
        self.merging_method = merging_method
        self.zero_init_fc = nn.Linear(embed_dim, embed_dim, bias=False)
        if self.merging_method == 'avg_pooling':
            pass
        elif self.merging_method == 'm_pooling':
            self.attn_layer = nn.Sequential( 
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.num_head = 16
    
    def forward(self, x, cu_seqlens, spatial_shapes):
        if not self.enable_merging:
            return x, cu_seqlens
        cu_seqlens_out = cu_seqlens.clone()  # (N+1, )
        feature_x = x
        x_i_list = []
        for i in range(1, len(cu_seqlens)):
            start_idx = cu_seqlens[i-1].item()
            end_idx = cu_seqlens[i].item()
            x_i = x[start_idx:end_idx, :]
            h, w = spatial_shapes[i-1]
            x_i = x_i.view(h, w, -1)  # (h, w, embed_dim)

            if self.merging_method == 'avg_pooling':
                x_i = rearrange(x_i, 'h w c -> c h w')
                x_i = F.avg_pool2d(x_i, kernel_size=2, stride=2)
                x_i = rearrange(x_i, 'c h w -> (h w) c')
            elif self.merging_method == 'm_pooling':
                x_i = rearrange(x_i, '(h p1) (w p2) c -> (h w) (p1 p2) c', p1=2, p2=2)
                pooled_x_i = x_i.mean(-2, keepdim=True).expand(-1, 4, -1)
                fused_x_i = torch.cat([x_i, pooled_x_i], dim=-1)
                attn_logits = self.attn_layer(fused_x_i)
                # multi-head attn
                attn_logits = rearrange(attn_logits, 'n s (m d) -> n m s d', m=self.num_head)
                attn_weights = F.softmax(attn_logits, dim=-2)
                attn_weights = rearrange(attn_weights, 'n m s d -> n s (m d)')
                # multi-head attn
                x_i = (x_i * attn_weights).sum(-2)
            
            x_i_list.append(x_i)
            cu_seqlens_out[i] = cu_seqlens_out[i-1] + x_i.shape[0]
        x = torch.cat(x_i_list, dim=0)  # (L, embed_dim)
        return x, cu_seqlens_out, spatial_shapes//2, feature_x

class MoonVitEncoderLayer(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "eager",
        activation=F.gelu,
        attn_bias: bool = False,
        enable_merging: bool = False,
        merging_method: str = "avg_pooling",
        merger_layer_index: List[int] = None,
        use_rope2d: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

        if merger_layer_index is not None and layer_idx in merger_layer_index:
            self.merger = PatchMergingLayer(
                embed_dim=hidden_dim,
                enable_merging=enable_merging,
                merging_method=merging_method,
                )
        else:
            self.merger = None

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Union[torch.Tensor, None] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set
        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(
            hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states

        if self.merger is not None:
            hidden_states, cu_seqlens, spatial_shapes, feature_x = self.merger(
                hidden_states, cu_seqlens, spatial_shapes
            )
            outputs = (hidden_states, cu_seqlens, spatial_shapes, feature_x)# return the feature_x for later use
        else:
            outputs = (hidden_states, cu_seqlens)

        return outputs

class MoonVitEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        use_fused_layer: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MoonVitEncoderLayer(layer_idx=i, **block_cfg) for i in range(num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)
        self.use_rope2d = block_cfg["use_rope2d"]
        if self.use_rope2d:
            self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        else:
            self.rope_2d = Rope2DPosEmb(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_hws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws=grid_hws)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_hws.dtype),
                grid_hws[:, 0] * grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
        down_scale_rate = 1
        feature_x_list = []
        for _, block in enumerate(self.blocks):
            layer_outputs = block(
                hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis, spatial_shapes=grid_hws
            )
            if len(layer_outputs) > 2:
                down_scale_rate *= 2
                hidden_states, cu_seqlens, grid_hws, feature_x = layer_outputs
                rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws=grid_hws, down_scale_rate=down_scale_rate)
                feature_x_list.append(feature_x)
            else:
                hidden_states, cu_seqlens = layer_outputs

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, grid_hws


class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MoonViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )
        
        config._attn_implementation = "sdpa" if not hasattr(config, "use_flash_attention_2") else "flash_attention_2"
        merger_layer_index = None
        use_rope2d = False
        if hasattr(config, "vision_config"):
            if hasattr(config.vision_config, "merger_layer_index"):
                merger_layer_index = config.vision_config.merger_layer_index
                merging_method = config.vision_config.merging_method
                use_rope2d = getattr(config.vision_config, "use_rope2d", False)
            # use_fused_layer = getattr(config.vision_config, "use_fused_layer", False)
        else:
            if hasattr(config, "merger_layer_index"):
                merger_layer_index = config.merger_layer_index
                merging_method = config.merging_method
                use_rope2d = getattr(config, "use_rope2d", False)
            # use_fused_layer = getattr(config, "use_fused_layer", False)

        if merger_layer_index is not None:
            enable_merging = True
            merging_method = merging_method if merging_method is not None else "avg_pooling"
        else:
            enable_merging = False
            merging_method = None        

        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
                "enable_merging": enable_merging,
                "merging_method": merging_method,
                "merger_layer_index": merger_layer_index,
                "use_rope2d": use_rope2d,
            },
            # use_fused_layer=use_fused_layer
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_hws: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hws (torch.Tensor): The grid height and width.
        Returns:
            torch.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states, grid_hws = self.encoder(hidden_states, grid_hws)
        return hidden_states, grid_hws

class MoonViTVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()
        
        self.is_loaded = False

        self.config = MoonViTConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = MoonViTImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        
        self.vision_tower = MoonVitPretrainedModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        print('moonvit')
        # 初始化为0，防止随机初始化带来的inf值的nan
        for name, param in self.vision_tower.named_parameters():
            if 'merger' in name:
                with torch.no_grad():
                    param.zero_()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
    
    def forward(self, images, patch_sizes):
        pixel_values = []
        for idx, image in enumerate(images):
            if not valid_images(image):
                raise ValueError("Invalid image input. Please provide a valid image.")
            C, H, W = image.shape
            patches = rearrange(image, "c (h p1) (w p2) -> h w c p1 p2", h=patch_sizes[idx][0], w=patch_sizes[idx][1])
            patches = rearrange(patches, "h w c p1 p2 -> (h w) c p1 p2")  # (L, C, p1, p2)
            pixel_values.append(patches)
        pixel_values = torch.concat(pixel_values, dim=0)  # (L*, C, p1, p2)

        # #### 根据pixel_values的长度，压缩原图和patch_sizes #### 
        # pixel_number = len(pixel_values)
        # max_patch_tokens = int(184*184*pixel_number)
        # min_h_w = 8
        # if pixel_values.shape[0] > max_patch_tokens:
        #     scale = (max_patch_tokens / pixel_values.shape[0]) ** 0.5
        #     resized_images, new_patch_sizes = [], []
        #     for idx, image in enumerate(images):
        #         C, H, W = image.shape
        #         gh, gw = patch_sizes[idx]
        #         p1, p2 = H // gh, W // gw
        #         new_gh = max(1, min(gh, int(gh * scale)))
        #         new_gw = max(1, min(gw, int(gw * scale)))
        #         new_gh = max(min_h_w, (min(x, new_gh) // min_h_w) * min_h_w)  # 确保新的高度和宽度是8的倍数
        #         new_gw = max(min_h_w, (min(y, new_gw) // min_h_w) * min_h_w)
        #         new_H, new_W = new_gh * p1, new_gw * p2
        #         image_rs = F.interpolate(
        #             image.unsqueeze(0), size=(new_H, new_W),
        #             mode='bilinear', align_corners=False
        #         ).squeeze(0)
        #         resized_images.append(image_rs)
        #         new_patch_sizes.append((new_gh, new_gw))
            
        #     ## 重新计算 pixel_values ##
        #     pixel_values = []
        #     for idx, image in enumerate(resized_images):
        #         gh, gw = new_patch_sizes[idx]
        #         patches = rearrange(
        #             image, "c (h p1) (w p2) -> h w c p1 p2", h=gh, w=gw
        #         )
        #         patches = rearrange(patches, "h w c p1 p2 -> (h w) c p1 p2")  # (L, C, p1, p2)  
        #         pixel_values.append(patches)
        #     pixel_values = torch.concat(pixel_values, dim=0)  # (L*, C, p1, p2)
        #     patch_sizes = new_patch_sizes

        grid_hws = torch.tensor([tuple(patch_size) for patch_size in patch_sizes], device=pixel_values.device) # (N, 2)
        image_features, grid_hws = self.vision_tower(pixel_values, grid_hws)
        feature_x_list = None
        if isinstance(image_features, tuple):
            image_features, feature_x_list = image_features
        output_features = []
        offset = 0
        for grid_hw in grid_hws:
            h, w = grid_hw
            num_tokens = h * w
            output_features.append(image_features[offset : offset + num_tokens].unsqueeze(0))  # (1, num_tokens, hidden_size)
            offset += num_tokens

        assert offset == image_features.shape[0], \
            f"Used {offset} tokens, but image_features has {image_features.shape[0]} tokens!"
        if feature_x_list is not None:
            output_features = list(zip(output_features, feature_x_list))
        return output_features


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