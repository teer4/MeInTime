import torch
import torch.nn as nn
from .util import (
    zero_module,
    normalization,
)


#contorl skip features from downsample block
class GatedResidualFusion(nn.Module):
    def __init__(self, in_c_res):
        super().__init__()
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

        gate = self.gate_conv(torch.cat([h_skip, res], dim=1))
        gated_res = gate * res

        return h_skip + gated_res


