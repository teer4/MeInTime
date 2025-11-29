from functools import partial
from typing import Tuple

import torch
from torch import nn
import numpy as np




def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas)#.pow(2) FIXME
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = (
            np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas

#从一维张量 a 中提取与 timestep 张量 t 对应的值，并 reshape 成 [batch_size, 1, 1, 1]，用于后续广播到多维图像张量。
def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]
) -> torch.Tensor:
    #假设 t.shape == [batch_size]，这一步会令 b = batch_size，*_ 表示忽略其余维度
    b, *_ = t.shape
    #gather(dim, index) 的意思是按照 index，在 dim 维度上取元素
    #t 是 [batch_size]，里面是每个样本当前对应的 timestep 索引
    #a.gather(-1, t) 的含义是从 a 中按 t 提供的索引取出值，返回 [batch_size]
    """
    a = torch.tensor([0.1, 0.2, 0.3, 0.4])  # [4]
    t = torch.tensor([2, 1, 3])            # [3]
    out = a.gather(-1, t)                  # [3] => [0.3, 0.2, 0.4]
    """
    out = a.gather(-1, t)
    #这里 reshape 把 [batch_size] 的 out 变成 [batch_size, 1, 1, 1]
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# Copy from: https://github.com/Max-We/sf-zero-signal-to-noise/blob/main/common_diffusion_noise_schedulers_are_flawed.ipynb
# Original paper: https://arxiv.org/abs/2305.08891
def enforce_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    betas = torch.from_numpy(betas)
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas.numpy()


class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",    #指定损失函数目标，“eps”/"x0"
        zero_snr=False,
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.zero_snr = zero_snr
        self.loss_type = loss_type

        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        if zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        self.betas = betas
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

        self.identity_loss = None 

    #register_buffer 是 PyTorch 中用来注册模型 非训练参数 的方法。它将某些不需要更新的状态注册到模型中，并确保它们在模型的保存和加载时得到处理
    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    #前向生成x_noisy，q(x_t 1| x_0)
    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    """
    def p_losses(self, model, x_start, t, cond, image_prompt):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = model(x_noisy, t, cond, image_prompt)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        return loss_simple
    """
    
    def get_x0_pred_from_model_output(self, model_output, x_noisy, t):
        """
        根据 self.parameterization 自动计算 x0_pred
        """
        if self.parameterization == "x0":
            return model_output
        elif self.parameterization == "eps":
            return self.predict_start_from_eps(model_output, x_noisy, t)
        elif self.parameterization == "v":
            return self.predict_start_from_v(x_noisy, model_output, t)
        else:
            raise NotImplementedError(f"Unsupported parameterization: {self.parameterization}")

    def predict_start_from_eps(self, eps, x_t, t):
        """
        根据噪声 eps 还原 x0
        x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        """
        sqrt_alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t


    def predict_start_from_v(self, x_t, v, t):
        """
        预测 x0_pred, 用于 identity loss:
        x₀ = √ᾱ_t * x_t - √(1 - ᾱ_t) * v
        """
        sqrt_alpha_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * v


    def p_losses(self, model, x_start, t, cond, image_prompt, x_start_pixel=None, identity_loss_fn=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = model(x_noisy, t, cond, image_prompt)

        # 计算目标（parameterization）
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # 主扩散损失
        diffusion_loss = self.get_loss(model_output, target, mean=False).mean()
        
        
        # ⬇️ ID Loss 部分
        loss_id = None
        if identity_loss_fn is not None and x_start_pixel is not None:
            with torch.no_grad(), torch.cuda.amp.autocast():
                x0_pred = self.get_x0_pred_from_model_output(model_output, x_noisy, t)
                x0_pred_pixel = model.vae_decode(x0_pred.half())  # 精度转换节省显存
                loss_id = identity_loss_fn(x0_pred_pixel, x_start_pixel, t)
        return diffusion_loss, loss_id


    def p_loss(self, model, x_start, t, cond, image_prompt):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        #lambda_t = self.sqrt_alphas_cumprod[t]   # shape: [B]
        batch_size = x_start.shape[0]
        lambda_t = torch.full((batch_size,), 0.75, device=x_start.device, dtype=torch.float32)

        model_output = model(x_noisy, t, cond, image_prompt, lambda_t)

        # 计算目标（parameterization）
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # 主扩散损失
        diffusion_loss = self.get_loss(model_output, target, mean=False).mean()  
        return diffusion_loss
