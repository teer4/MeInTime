import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from diffbir.model.unet import UNetModel
from omegaconf import OmegaConf

"""
def print_module_tree(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}- {name}: {child.__class__.__name__}")
        print_module_tree(child, prefix + "  ")


def print_unet_structure(unet: UNetModel):
    print("\n====== [INPUT BLOCKS] ======")
    for idx, block in enumerate(unet.input_blocks):
        print(f"\n-- input_blocks[{idx}] : {type(block).__name__}")
        print_module_tree(block, prefix="  ")

    print("\n====== [MIDDLE BLOCK] ======")
    print_module_tree(unet.middle_block, prefix="  ")

    print("\n====== [OUTPUT BLOCKS] ======")
    for idx, block in enumerate(unet.output_blocks):
        print(f"\n-- output_blocks[{idx}] : {type(block).__name__}")
        print_module_tree(block, prefix="  ")

    print("\n====== [FINAL OUTPUT HEAD] ======")
    print_module_tree(unet.out, prefix="  ")

    if hasattr(unet, "id_predictor"):
        print("\n====== [ID PREDICTOR HEAD] ======")
        print_module_tree(unet.id_predictor, prefix="  ")
"""

if __name__ == "__main__":
    print(">> Initializing UNet...")
    unet = UNetModel(
        image_size=32,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=-1,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=1024,  # 关键！启用 SpatialTransformer 的必要条件
        n_embed=None,
        legacy=True
    )

    print(">> Printing UNet structure...")
    #name_children()只返回顶层模块不递归，name_modules()返回所有子模块（递归）
    for name, module in unet.named_children():    
        print(name, module)

"""
time_embed Sequential(
  (0): Linear(in_features=320, out_features=1280, bias=True)
  (1): SiLU()
  (2): Linear(in_features=1280, out_features=1280, bias=True)
)
input_blocks ModuleList(
  (0): TimestepEmbedSequential(
    (0): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (1-2): 2 x TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=640, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
      (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=320, out_features=2560, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=1280, out_features=320, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=1024, out_features=320, bias=False)
            (to_v): Linear(in_features=1024, out_features=320, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (3): TimestepEmbedSequential(
    (0): Downsample(
      (op): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
  )
  (4): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
      (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=2560, out_features=640, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=1024, out_features=640, bias=False)
            (to_v): Linear(in_features=1024, out_features=640, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (5): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
      (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=2560, out_features=640, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=1024, out_features=640, bias=False)
            (to_v): Linear(in_features=1024, out_features=640, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (6): TimestepEmbedSequential(
    (0): Downsample(
      (op): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
  )
  (7): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v): Linear(in_features=1024, out_features=1280, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (8): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v): Linear(in_features=1024, out_features=1280, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (9): TimestepEmbedSequential(
    (0): Downsample(
      (op): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
  )
  (10-11): 2 x TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
  )
)
middle_block TimestepEmbedSequential(
  (0): ResBlock(
    (in_layers): Sequential(
      (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
      (1): SiLU()
      (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (h_upd): Identity()
    (x_upd): Identity()
    (emb_layers): Sequential(
      (0): SiLU()
      (1): Linear(in_features=1280, out_features=2560, bias=True)
    )
    (out_layers): Sequential(
      (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
      (1): SiLU()
      (2): Dropout(p=0.0, inplace=False)
      (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (skip_connection): Identity()
  )
  (1): SpatialTransformer(
    (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
    (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    (transformer_blocks): ModuleList(
      (0): BasicTransformerBlock(
        (attn1): MemoryEfficientCrossAttention(
          (to_q): Linear(in_features=1280, out_features=1280, bias=False)
          (to_k): Linear(in_features=1280, out_features=1280, bias=False)
          (to_v): Linear(in_features=1280, out_features=1280, bias=False)
          (to_out): Sequential(
            (0): Linear(in_features=1280, out_features=1280, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
        (ff): FeedForward(
          (net): Sequential(
            (0): GEGLU(
              (proj): Linear(in_features=1280, out_features=10240, bias=True)
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=5120, out_features=1280, bias=True)
          )
        )
        (attn2): IPCrossAttention(
          (to_q): Linear(in_features=1280, out_features=1280, bias=False)
          (to_k): Linear(in_features=1024, out_features=1280, bias=False)
          (to_v): Linear(in_features=1024, out_features=1280, bias=False)
          (to_k_ip): Linear(in_features=1024, out_features=1280, bias=False)
          (to_v_ip): Linear(in_features=1024, out_features=1280, bias=False)
          (to_out): Sequential(
            (0): Linear(in_features=1280, out_features=1280, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
        (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      )
    )
    (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
  )
  (2): ResBlock(
    (in_layers): Sequential(
      (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
      (1): SiLU()
      (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (h_upd): Identity()
    (x_upd): Identity()
    (emb_layers): Sequential(
      (0): SiLU()
      (1): Linear(in_features=1280, out_features=2560, bias=True)
    )
    (out_layers): Sequential(
      (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
      (1): SiLU()
      (2): Dropout(p=0.0, inplace=False)
      (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (skip_connection): Identity()
  )
)
output_blocks ModuleList(
  (0-1): 2 x TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (2): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Upsample(
      (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (3-4): 2 x TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v): Linear(in_features=1024, out_features=1280, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (5): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=2560, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v): Linear(in_features=1024, out_features=1280, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=1280, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=1280, out_features=1280, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Upsample(
      (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (6): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
      (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=2560, out_features=640, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=1024, out_features=640, bias=False)
            (to_v): Linear(in_features=1024, out_features=640, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (7): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
      (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=2560, out_features=640, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=1024, out_features=640, bias=False)
            (to_v): Linear(in_features=1024, out_features=640, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (8): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
      (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=2560, out_features=640, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=1024, out_features=640, bias=False)
            (to_v): Linear(in_features=1024, out_features=640, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=640, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=640, out_features=640, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Upsample(
      (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
  (9): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=640, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
      (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=320, out_features=2560, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=1280, out_features=320, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=1024, out_features=320, bias=False)
            (to_v): Linear(in_features=1024, out_features=320, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (10-11): 2 x TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1280, out_features=640, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0.0, inplace=False)
        (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): SpatialTransformer(
      (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
      (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      (transformer_blocks): ModuleList(
        (0): BasicTransformerBlock(
          (attn1): MemoryEfficientCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=320, out_features=320, bias=False)
            (to_v): Linear(in_features=320, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): FeedForward(
            (net): Sequential(
              (0): GEGLU(
                (proj): Linear(in_features=320, out_features=2560, bias=True)
              )
              (1): Dropout(p=0.0, inplace=False)
              (2): Linear(in_features=1280, out_features=320, bias=True)
            )
          )
          (attn2): IPCrossAttention(
            (to_q): Linear(in_features=320, out_features=320, bias=False)
            (to_k): Linear(in_features=1024, out_features=320, bias=False)
            (to_v): Linear(in_features=1024, out_features=320, bias=False)
            (to_k_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_v_ip): Linear(in_features=1024, out_features=320, bias=False)
            (to_out): Sequential(
              (0): Linear(in_features=320, out_features=320, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        )
      )
      (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
out Sequential(
  (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
  (1): SiLU()
  (2): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
"""