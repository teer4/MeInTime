from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage
from .spaced_sampler import SpacedSampler

class DDSSampler(SpacedSampler):
    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        #dds_lr: float = 0.5,  # 等效于 s
        dds_steps: int = 1,    # 每步梯度更新次数 N
    ) -> "DDSSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        #self.dds_lr = dds_lr
        self.dds_steps = dds_steps

    #@torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        cond_trg: Dict[str, torch.Tensor],  # 新增: 目标语义条件
        cond_src: Dict[str, torch.Tensor],  
        image_prompt: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        # 将父类的 numpy 数组转换成 torch Tensor 并放到正确设备
        #self.sqrt_one_minus_alphas_cumprod_torch = torch.tensor(self.sqrt_one_minus_alphas_cumprod, device=device, dtype=torch.float32)
        self.sqrt_alphas_cumprod_torch = torch.tensor(self.sqrt_alphas_cumprod, device=device, dtype=torch.float32)

        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)
        x = x_T

        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]

        lambda_t = torch.full((bs,), 0.75, device=x.device, dtype=torch.float32)

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)

            #scale = self.sqrt_one_minus_alphas_cumprod_torch[t]
            #scale = self.sqrt_alphas_cumprod_torch[t]
            #print(scale)
            scale = 0

            # === 标准扩散采样步骤 ===
            x = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond_trg,
                uncond,
                image_prompt,
                cur_cfg_scale,
            )

            # === DDS 优化步骤 ===
            with torch.enable_grad():
                for _ in range(self.dds_steps):
                    x = x.detach().requires_grad_()  # 生成新 leaf 张量 并设置可导
                    with torch.no_grad():
                        eps_src = model(x, model_t, cond_src, image_prompt, lambda_t)  
                        eps_trg = model(x, model_t, cond_trg, image_prompt, lambda_t)
                    delta_eps = eps_trg - eps_src
                    loss = (delta_eps * x).mean()    # 构造 loss
                    grad = torch.autograd.grad(1500 * loss, x)[0]
                    x = x - scale * grad.detach()  # detach 防止grad链过长

        return x
