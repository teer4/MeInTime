from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ..model.util import checkpoint, zero_module, exists, default
from ..model.config import Config, AttnMode

# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class IPCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (xformers). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        #self.scale = id_scale  # 控制注入强度

        # 主分支
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # IP-Adapter 分支
        self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

        self.attention_op = None  # xformers 核心模块

    def forward(self, x, context=None, image_prompt=None, lambda_t=None, mask=None):
        b = x.shape[0]
        q = self.to_q(x)
        #如果context为None，则使用x作为默认值，否则保留原本的context
        context = default(context, x)     

        # 主分支注意力
        k = self.to_k(context)
        v = self.to_v(context)

        # multi-head reshape
        q_, k_, v_ = map(
            lambda t: t.view(b, -1, self.heads, self.dim_head)
                      .permute(0, 2, 1, 3)
                      .reshape(b * self.heads, -1, self.dim_head),
            (q, k, v)
        )
        main_attn = Config.xformers.ops.memory_efficient_attention(
            q_, k_, v_, attn_bias=None, op=self.attention_op
        )

        # IP-Adapter 分支注意力
        if image_prompt is not None:
            k_ip = self.to_k_ip(image_prompt)
            v_ip = self.to_v_ip(image_prompt)

            k_ip = k_ip.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, -1, self.dim_head)
            v_ip = v_ip.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, -1, self.dim_head)

            id_attn = Config.xformers.ops.memory_efficient_attention(
                q_, k_ip, v_ip, attn_bias=None, op=self.attention_op
            )
        else:
            id_attn = torch.zeros_like(main_attn)


        if lambda_t is None:
            lambda_t = torch.ones((b,), device=x.device)

        # reshape → [B, 1, 1] → broadcast to [B * H, 1, 1]
        lambda_t = lambda_t.view(b, 1, 1).repeat_interleave(self.heads, dim=0)


        # 合并输出
        out = main_attn + lambda_t * id_attn
        out = out.view(b, self.heads, -1, self.dim_head).permute(0, 2, 1, 3).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)
