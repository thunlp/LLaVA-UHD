

import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math
from transformers.activations import ACT2FN
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

class TokenPacker(nn.Module):
    is_causal = False
    def __init__(
            self,
            embed_dim=1152,
            intermediate_size=4304,
            num_heads=16,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        gamma_init_eps = 1e-5
        layer_norm_eps = 1e-6
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.ln_q = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.gamma1 = nn.Parameter(gamma_init_eps * torch.ones(embed_dim), requires_grad=True)

        self.ln_ffn = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, intermediate_size),
            ACT2FN['gelu_pytorch_tanh'],
            nn.Linear(intermediate_size, self.embed_dim),
        )
        self.gamma2 = nn.Parameter(gamma_init_eps * torch.ones(embed_dim), requires_grad=True)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    def _attn(self, query, key, value, attn_mask):
        B, N_q, _ = query.shape
        B, N_k, _ = key.shape
        query_states = self.q_proj(query)
        key_states = self.k_proj(key)
        value_states = self.v_proj(value)

        query_states = query_states.view(B, N_q, self.num_heads, self.head_dim)
        key_states = key_states.view(B, N_k, self.num_heads, self.head_dim)
        value_states = value_states.view(B, N_k, self.num_heads, self.head_dim)
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attn_mask,
            N_q,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(B, N_q, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output

    def forward(self, x, x_res, query=None, key=None, value=None, attn_mask=None):
        ### x_res是((h w) 4 c)
        ### x是((h w) 1 c) 均值
        if query is not None:
            query = self.ln_q(query)
        else:
            query = self.ln_q(x)[:, None, :]
        if key is not None:
            key = key
        else:
            key = self.ln_kv(x_res)
        if value is not None:
            value = value 
        else:
            value = key

        out = self._attn(
            query,
            key,
            value,
            attn_mask=attn_mask)[0]
        x_res = out

        x = x + self.gamma1 * x_res # qkv norm，算attn，然后乘以gamma1，然后add 残差
        x = x + self.gamma2 * self.ffn(self.ln_ffn(x))
        # 这个x这个avg是直通分支，通过gamma学习引入残差分量
        return x, key, value