import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from diffbir.model.unet import UNetModel
from omegaconf import OmegaConf
from diffbir.model.controlnet import ControlledUnetModel

if __name__ == "__main__":
    print(">> Initializing UNet...")
    unet = ControlledUnetModel(
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
        use_ip_adapter=True,
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
control_fusions ModuleList(
  (0-5): 6 x GatedResidualFusion(
    (norm_h): GroupNorm32(32, 1280, eps=1e-05, affine=True)
    (norm_r): GroupNorm32(32, 1280, eps=1e-05, affine=True)
    (gate_conv): Sequential(
      (0): Conv2d(2560, 640, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(640, 1, kernel_size=(3, 3), stride=(1, 1))
      (3): Sigmoid()
    )
  )
  (6-8): 3 x GatedResidualFusion(
    (norm_h): GroupNorm32(32, 640, eps=1e-05, affine=True)
    (norm_r): GroupNorm32(32, 640, eps=1e-05, affine=True)
    (gate_conv): Sequential(
      (0): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(320, 1, kernel_size=(3, 3), stride=(1, 1))
      (3): Sigmoid()
    )
  )
  (9-12): 4 x GatedResidualFusion(
    (norm_h): GroupNorm32(32, 320, eps=1e-05, affine=True)
    (norm_r): GroupNorm32(32, 320, eps=1e-05, affine=True)
    (gate_conv): Sequential(
      (0): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(160, 1, kernel_size=(3, 3), stride=(1, 1))
      (3): Sigmoid()
    )
  )
)
"""