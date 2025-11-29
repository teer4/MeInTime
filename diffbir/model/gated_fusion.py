import torch
import torch.nn as nn
from .util import (
    zero_module,
    normalization,
)
"""
GatedResidualFusion用于融合残差和skip connection

对于每一个 UpBlock，输入包括：
    当前 block 的特征图 h（上采样流）
    对应下采样 block 的跳跃连接 skip（下采样流）
    对应的 ControlNet 输出残差 res（辅助条件）

| UpBlock ID | h+skip 通道数（in_c_fused） | res 通道数（in_c_res） |
| ---------- | ------------------------ | ------------------- |
| 0-2        | 1280                     | 1280                |
| 3-4        | 1280                     | 1280                |
| 5          | 1280                     | 640                 |
| 6-7        | 640                      | 640                 |
| 8          | 640                      | 320                 |
| 9-11       | 320                      | 320                 |

"""
#参数量：34,580,480

#contorl skip features from downsample block
class GatedResidualFusion(nn.Module):
    def __init__(self, in_c_res):
        super().__init__()
        #GroupNorm32(32, channels)
        self.norm_h = normalization(in_c_res)
        self.norm_r = normalization(in_c_res)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_c_res * 2, in_c_res, kernel_size=1),  
            nn.Conv2d(in_c_res, in_c_res, kernel_size=3, padding=1, groups=in_c_res),  # Depthwise
            zero_module(nn.Conv2d(in_c_res, in_c_res, kernel_size=1)),  # Pointwise
            nn.Sigmoid()
        )  

    def forward(self, h_skip, res):
        
        h_skip = self.norm_h(h_skip)
        res = self.norm_r(res)

        # 计算门控权重因子
        gate = self.gate_conv(torch.cat([h_skip, res], dim=1))

        # 残差加权
        gated_res = gate * res

        # 加入主干
        return h_skip + gated_res


"""
[Gate] fused: torch.Size([1, 1280, 8, 8]), res: torch.Size([1, 1280, 8, 8])
[Gate] fused: torch.Size([1, 2560, 8, 8]), res: torch.Size([1, 1280, 8, 8])
[Gate] fused: torch.Size([1, 2560, 8, 8]), res: torch.Size([1, 1280, 8, 8])
[Gate] fused: torch.Size([1, 2560, 8, 8]), res: torch.Size([1, 1280, 8, 8])
[Gate] fused: torch.Size([1, 2560, 16, 16]), res: torch.Size([1, 1280, 16, 16])
[Gate] fused: torch.Size([1, 2560, 16, 16]), res: torch.Size([1, 1280, 16, 16])
[Gate] fused: torch.Size([1, 1920, 16, 16]), res: torch.Size([1, 640, 16, 16])
[Gate] fused: torch.Size([1, 1920, 32, 32]), res: torch.Size([1, 640, 32, 32])
[Gate] fused: torch.Size([1, 1280, 32, 32]), res: torch.Size([1, 640, 32, 32])
[Gate] fused: torch.Size([1, 960, 32, 32]), res: torch.Size([1, 320, 32, 32])
[Gate] fused: torch.Size([1, 960, 64, 64]), res: torch.Size([1, 320, 64, 64])
[Gate] fused: torch.Size([1, 640, 64, 64]), res: torch.Size([1, 320, 64, 64])
[Gate] fused: torch.Size([1, 640, 64, 64]), res: torch.Size([1, 320, 64, 64])
"""
