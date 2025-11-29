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
        #dds_lr: float = 0.5,  
        dds_steps: int = 1,  
        age_guidance: bool = False,  
    ) -> "DDSSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        #self.dds_lr = dds_lr
        self.dds_steps = dds_steps
        self.age_guidance = age_guidance

    #@torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        cond_trg: Dict[str, torch.Tensor],  
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

            scale = self.sqrt_alphas_cumprod_torch[t]

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

            if self.age_guidance:
                with torch.enable_grad():
                    for _ in range(self.dds_steps):
                        x = x.detach().requires_grad_()  
                        with torch.no_grad():
                            eps_src = model(x, model_t, cond_src, image_prompt, lambda_t)  
                            eps_trg = model(x, model_t, cond_trg, image_prompt, lambda_t)
                        delta_eps = eps_trg - eps_src
                        loss = (delta_eps * x).mean()    
                        grad = torch.autograd.grad(1200 * loss, x)[0]
                        x = x - scale * grad.detach()  
        return x
