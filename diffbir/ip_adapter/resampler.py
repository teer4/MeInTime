# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math

import torch
import torch.nn as nn


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


#Perceiver Attention: 将高维输入压缩为低维latent (b, n, dim) -> (b, m, dim)
#Q来自learnable latent vectors(b, n, dim)，KV来自原始输入embedding(b, m, dim)，注意力计算固定在latent长度上, 梯度更新只更新latent
#将face embedding(b, length, dim)转化为(b, num_tokens, dim)
class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=16):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads    #inner_dim = dim = 64 * 16 = 1024

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    #x: face embedding -> (b, length, dim)   latents -> (b, num_tokens, dim)   #[1, 1, 512]
    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)    #(b, num_tokens, dim)
        kv_input = torch.cat((x, latents), dim=-2)      #(b, length + num_tokens, dim)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)    #k, v -> (b, length + num_tokens, dim)
        
        q = reshape_tensor(q, self.heads)   #(b, heads, num_tokens, dim_head)
        k = reshape_tensor(k, self.heads)   #(b, heads, length + num_tokens, dim_head)
        v = reshape_tensor(v, self.heads)   #(b, heads, length + num_tokens, dim_head)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)    #(b, heads, num_tokens, length + num_tokens)
        out = weight @ v    #(b, heads, num_tokens, dim_head)
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)   #(b, num_tokens, heads * dim_head)

        return self.to_out(out)    #(b, num_tokens, dim)

#返回[2, 16, 1024]
#特征映射网络，将insightface提取的face embedding (512,)（b, length, embedding_dim）转化为 (b, num_tokens, output_dim) 适应unet交叉注意力计算
class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,    #resampler内部计算维度
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=16,
        embedding_dim=512,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)   #随机生成初始化Q向量(1, num_tokens, dim)
        
        self.proj_in = nn.Linear(embedding_dim, dim)   

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):     #x是输入的face embedding -> (b, length, embedding_dim)
        
        latents = self.latents.repeat(x.size(0), 1, 1)   #(b, num_tokens, dim)
        
        x = self.proj_in(x)    #(b, length, embedding_dim) -> (b, length, dim)
        
        #attn, ff分别是PerceiverAttention, FeedForward类的实例
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents    #加上原来查询向量，确保梯度可以反向传播
            latents = ff(latents) + latents         #经过feedforward层做非线性变换
            
        latents = self.proj_out(latents)     #(b, num_tokens, output_dim) 
        return self.norm_out(latents)

"""
# 打印模型的结构
def print_module_tree(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}- {name}: {child.__class__.__name__}")
        print_module_tree(child, prefix + "  ")

# 创建 Resampler 实例
resampler = Resampler(dim=1024, embedding_dim=512, output_dim=1024)

# 打印 Resampler 结构
print("Resampler 结构:")
print_module_tree(resampler)

Resampler 结构:
- proj_in: Linear
- proj_out: Linear
- norm_out: LayerNorm
- layers: ModuleList
  - 0: ModuleList
    - 0: PerceiverAttention
      - norm1: LayerNorm
      - norm2: LayerNorm
      - to_q: Linear
      - to_kv: Linear
      - to_out: Linear
    - 1: Sequential
      - 0: LayerNorm
      - 1: Linear
      - 2: GELU
      - 3: Linear
  - 1: ModuleList
    - 0: PerceiverAttention
      - norm1: LayerNorm
      - norm2: LayerNorm
      - to_q: Linear
      - to_kv: Linear
      - to_out: Linear
    - 1: Sequential
      - 0: LayerNorm
      - 1: Linear
      - 2: GELU
      - 3: Linear
  - 2: ModuleList
    - 0: PerceiverAttention
      - norm1: LayerNorm
      - norm2: LayerNorm
      - to_q: Linear
      - to_kv: Linear
      - to_out: Linear
    - 1: Sequential
      - 0: LayerNorm
      - 1: Linear
      - 2: GELU
      - 3: Linear
  - 3: ModuleList
    - 0: PerceiverAttention
      - norm1: LayerNorm
      - norm2: LayerNorm
      - to_q: Linear
      - to_kv: Linear
      - to_out: Linear
    - 1: Sequential
      - 0: LayerNorm
      - 1: Linear
      - 2: GELU
      - 3: Linear
"""