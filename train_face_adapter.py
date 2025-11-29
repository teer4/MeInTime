import os
from argparse import ArgumentParser
import copy
import itertools
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config
from diffbir.sampler import SpacedSampler

from diffbir.dataset.face_adapter_dataset import FaceAdapterDataset
from pyiqa import create_metric  
from eval import evaluate_on_val

"""
Dataset contains 178,877 images
[PARAMS] image_proj_model: 51,949,568
[PARAMS] UNet to_k_ip / to_v_ip: 25,559,040
[PARAMS] control_fusions: 11,632,000  / 34,580,480
[PARAMS] Total trainable parameters (deduplicated): 89,140,608
"""

def main(args) -> None:
    # Setup accelerator
    accelerator = Accelerator(split_batches=True ) 
    print(f"[RANK {accelerator.process_index}] using device: {accelerator.device}")
    set_seed(42, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Create experiment folders
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    #  1. 创建模型
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)

    #  2. 加载 Stable Diffusion 预训练权重（不包含 controlnet）
    sd_state_dict = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd_state_dict)
    if accelerator.is_main_process:
        print(f"Loaded SD weights from {cfg.train.sd_path}")
        print(f"Unused weights: {unused}")
        print(f"Missing weights: {missing}")

    #  3. 加载 DiffBIR 训练阶段保存的 controlnet 权重
    controlnet_ckpt = torch.load(cfg.train.controlnet_path, map_location="cpu")
    cldm.controlnet.load_state_dict(controlnet_ckpt, strict=True)
    if accelerator.is_main_process:
        print(f"Loaded trained ControlNet weights from {cfg.train.controlnet_path}")


    # ---------- 初始化或 Resume 权重 ----------
    ckpt = None
    resume_path = getattr(cfg.train, "resume_path", None)
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")

        cldm.image_proj_model.load_state_dict(ckpt["image_proj_model"])
        for name, module in cldm.unet.named_modules():
            if name in ckpt["ip_attn"]:
                module.load_state_dict(ckpt["ip_attn"][name])
        for name, module in cldm.unet.named_modules():
            if name in ckpt["control_fusions"]:
                module.load_state_dict(ckpt["control_fusions"][name])
        if accelerator.is_main_process:
            print(f"[Resume] Loaded image_proj_model & ip_attn & control_fusions from {resume_path}")
    else:
        # 遍历所有 UNet 层中的 attn2，如果是 IPCrossAttention，则初始化 to_k_ip、to_v_ip
        for name, module in cldm.unet.named_modules():
            if hasattr(module, "to_k_ip") and hasattr(module, "to_v_ip"):
                print(f"Initializing to_k_ip/to_v_ip from to_k/to_v in module: {name}")
                with torch.no_grad():
                    module.to_k_ip.weight.copy_(module.to_k.weight.clone())
                    module.to_v_ip.weight.copy_(module.to_v.weight.clone())
                if accelerator.is_main_process:
                    print(f"[Init] Initialized to_k_ip/to_v_ip from to_k/to_v in {name}")


    # SwinIR作为第一阶段恢复器（冻结）
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # 冻结 cldm 中所有参数的 requires_grad
    for param in cldm.parameters():
        param.requires_grad = False

    # 解冻 image_proj_model（Resampler）参数
    for param in cldm.image_proj_model.parameters():
        param.requires_grad = True

    # 解冻 UNet 中所有 attn2 层的 to_k_ip 和 to_v_ip 参数
    for name, module in cldm.unet.named_modules():
        if hasattr(module, "to_k_ip") and hasattr(module, "to_v_ip"):
            print(f"[unfreeze] {name}.to_k_ip / to_v_ip")
            for param in module.to_k_ip.parameters():
                param.requires_grad = True
            for param in module.to_v_ip.parameters():
                param.requires_grad = True

    # 解冻 control_fusions 中的所有参数
    control_fusion_params = []
    for idx, module in enumerate(cldm.unet.control_fusions):
        print(f"[unfreeze] control_fusions.{idx}")
        for param in module.parameters():
            param.requires_grad = True
            control_fusion_params.append(param)


    """
    for name, module in unet.named_modules():
        if hasattr(module, "to_k_ip"):
            print(name, module.to_k_ip)
    >> Printing UNet structure...
    input_blocks.1.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=320, bias=False)
    input_blocks.2.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=320, bias=False)
    input_blocks.4.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=640, bias=False)
    input_blocks.5.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=640, bias=False)
    input_blocks.7.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    input_blocks.8.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    middle_block.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    output_blocks.3.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    output_blocks.4.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    output_blocks.5.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=1280, bias=False)
    output_blocks.6.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=640, bias=False)
    output_blocks.7.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=640, bias=False)
    output_blocks.8.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=640, bias=False)
    output_blocks.9.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=320, bias=False)
    output_blocks.10.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=320, bias=False)
    output_blocks.11.1.transformer_blocks.0.attn2 Linear(in_features=1024, out_features=320, bias=False)
    """

    # 自动查找 unet 中所有具有 to_k_ip / to_v_ip 的模块参数
    to_kv_ip_params = []
    for name, module in cldm.unet.named_modules():
        if hasattr(module, "to_k_ip") and hasattr(module, "to_v_ip"):
            to_kv_ip_params.extend(list(module.to_k_ip.parameters()))
            to_kv_ip_params.extend(list(module.to_v_ip.parameters()))

    
    # image_proj_model 参数
    params_proj = list(cldm.image_proj_model.parameters())

    # 合并所有待优化参数
    all_params = params_proj + to_kv_ip_params + control_fusion_params
    # 去重统计（根据 id 去重，确保不重复统计同一个参数对象）
    unique_params = {id(p): p for p in all_params}

    # 统计函数
    def count_parameters(params):
        return sum(p.numel() for p in params if p.requires_grad)

    # 分别统计
    proj_param_count = count_parameters(params_proj)
    kv_param_count = count_parameters(to_kv_ip_params)
    fusions_param_count = count_parameters(control_fusion_params)

    if accelerator.is_main_process:
        print(f"[PARAMS] image_proj_model: {proj_param_count:,}")
        print(f"[PARAMS] UNet to_k_ip / to_v_ip: {kv_param_count:,}")
        print(f"[PARAMS] control_fusions: {fusions_param_count:,}")

    # 去重后的总参数统计
    total_param_count = count_parameters(unique_params.values())

    if accelerator.is_main_process:
        print(f"[PARAMS] Total trainable parameters (deduplicated): {total_param_count:,}")
    

    # 合并 image_proj_model 和所有 to_k_ip / to_v_ip 参数
    params_to_opt = itertools.chain(
        cldm.image_proj_model.parameters(),
        to_kv_ip_params, # ✅ 现在是纯 list，无需展开 *
        control_fusion_params,  # 控制融合层的参数
    )

    optimizer = torch.optim.AdamW(params_to_opt, lr=cfg.train.learning_rate)

    # ⚠️ 只对 image_proj_model 和 to_k_ip/to_v_ip 使用 EMA
    ema_params = itertools.chain(
        cldm.image_proj_model.parameters(),
        to_kv_ip_params,  # 你之前已构建此参数列表
        control_fusion_params,
    )
    ema = ExponentialMovingAverage(ema_params, decay=cfg.train.ema_decay)
    if ckpt is not None:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if accelerator.is_main_process:
            print(f"[Resume] Optimizer and EMA state loaded from {resume_path}")

    ema.to(device)

    # Setup data:
    collate_fn = FaceAdapterDataset.collate_fn  # ✅ 注意：是类，不是实例
    train_dataset = instantiate_from_config(cfg.dataset.train)
    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=True,
                            num_workers=cfg.train.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn,
                            persistent_workers=True,       # ✅ 加速
                            prefetch_factor=2              # ✅ 提前加载
                        )
    """
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=collate_fn,
                        )
    """

    if accelerator.is_main_process:
        print(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models     
    swinir.eval().to(device)     
    diffusion.to(device)
    
    cldm, optimizer, train_loader = accelerator.prepare(cldm, optimizer, train_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    cldm.train()

    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Setup training loop
    global_step = 0
    max_steps = cfg.train.train_steps  # e.g., 50000
    epoch = 0
    if ckpt is not None:
        if "global_step" in ckpt:
            global_step = ckpt.get("global_step", 0)
        if "epoch" in ckpt:
            epoch = ckpt.get("epoch", 0)
        if accelerator.is_main_process:
            print(f"[Resume] Loaded global_step & epoch from {resume_path}")
    step_loss = []
    epoch_loss = []

    # 初始化一次，全局持有
    lpips_metric = create_metric("lpips-vgg", device=device)
    niqe_metric = create_metric("niqe", device=device)
    psnr_metric = create_metric("psnr", device=device)
    ssim_metric = create_metric("ssim", device=device)


    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            total=len(train_loader),
            disable=not accelerator.is_main_process,
            unit="batch",
            position=0,
            leave=True,
        )

        for batch in train_loader:
            # ------------------------- 数据准备 ------------------------- #
            gt = batch["gt"].to(device)                  # [B, 3, 512, 512], [-1,1]
            lq = batch["lq"].to(device)                  # [B, 3, 512, 512], [0,1]
            prompt = batch["prompt"]                     # List[str]（固定为空串）
            emb = batch["embedding"].to(device)          # [B, 512]

            # --------------------- 条dataloader件准备（cond + emb） --------------------- #
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)           # 编码 clean GT → latent
                clean = swinir(lq)                       # SwinIR 作为第一阶段还原器
                cond = pure_cldm.prepare_condition(clean, prompt)  # 生成条件

                # 可选增强：对条件图添加噪声（可跳过）
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )

            # 将 face embedding 转为 cross-attn tokens
            image_prompt = pure_cldm.encode_image_prompt(emb)

            # ------------------------- 扩散过程 ------------------------- #
            # 为每张图采样不同时间步 t ∈ [0, T)
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)

            # 计算损失
            loss_diffusion = diffusion.p_loss(
                cldm,               # model (含 face-adapter、UNet、VAE)
                z_0,                # latent 表示的 x_start
                t,                  # 当前 timestep
                cond_aug,           # 条件输入
                image_prompt,       # prompt
            )
            
            total_loss = loss_diffusion
            
            # ------------------------- 优化步骤 ------------------------- #
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
            ema.update()

            accelerator.wait_for_everyone()


            # --- 全局步骤计数与日志缓存 ---
            global_step += 1
            # 记录每步
            step_loss.append(total_loss.item())           # 用于最终 loss 曲线
            epoch_loss.append(total_loss.item())          # 用于 epoch 汇总
            
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Step: {global_step:07d}, "
                f"Total Loss: {total_loss.item():.6f}, "
            )
            """
            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                     
                    # total loss = diffusion 
                    avg_total = accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    ).mean().item()
                    writer.add_scalar("loss/total", avg_total, global_step)
                    step_loss.clear()
            """
            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    try:
                        avg_total = sum(step_loss) / len(step_loss)
                        writer.add_scalar("loss/total", avg_total, global_step)
                    except ZeroDivisionError:
                        pass  # 防止极端情况
                    step_loss.clear()


            
            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    ckpt = {
                        "image_proj_model": pure_cldm.image_proj_model.state_dict(),
                        "ip_attn": {
                            name: module.state_dict()
                            for name, module in pure_cldm.unet.named_modules()
                            if hasattr(module, "to_k_ip") and hasattr(module, "to_v_ip")
                        },
                        "control_fusions": {
                            name: module.state_dict()
                            for name, module in pure_cldm.unet.named_modules()
                            if "control_fusions" in name
                        },
                        "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch
                    }
                    ckpt_path = os.path.join(ckpt_dir, f"{global_step:07d}.pt")
                    torch.save(ckpt, ckpt_path)

            # sample
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                log_image_prompt = image_prompt[:N]
                log_identity = batch["identity"][:N]

                # 使用 EMA 权重进行 sample
                ema.store()
                ema.copy_to()
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        image_prompt=log_image_prompt,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    samples = (pure_cldm.vae_decode(z) + 1) / 2  # [0,1]
                    gts = (log_gt + 1) / 2                       # [0,1]

                    if accelerator.is_main_process:
                        # 可视化图像
                        for tag, image in [
                            ("image/samples", samples),
                            ("image/gt", gts),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                        writer.add_text(
                            "identity",
                            "\n".join(log_identity),
                            global_step
                        )

                        # 计算指标
                        lpips_vals, niqe_vals, psnr_vals, ssim_vals = [], [], [], []
                        for s, g in zip(samples, gts):
                            s = s.unsqueeze(0).to(device).clamp(0,1)
                            g = g.unsqueeze(0).to(device).clamp(0,1)
                            lpips_vals.append(lpips_metric(s, g).item())
                            psnr_vals.append(psnr_metric(s, g).item())
                            ssim_vals.append(ssim_metric(s, g).item())
                            niqe_vals.append(niqe_metric(s).item())

                        avg_lpips = sum(lpips_vals) / len(lpips_vals)
                        avg_niqe = sum(niqe_vals) / len(niqe_vals)
                        avg_psnr = sum(psnr_vals) / len(psnr_vals)
                        avg_ssim = sum(ssim_vals) / len(ssim_vals)

                        writer.add_scalar("metric/lpips", avg_lpips, global_step)
                        writer.add_scalar("metric/niqe", avg_niqe, global_step)
                        writer.add_scalar("metric/psnr", avg_psnr, global_step)
                        writer.add_scalar("metric/ssim", avg_ssim, global_step)

                ema.restore()
                cldm.train()


            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        """
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        """
        epoch += 1
        if len(epoch_loss) > 0:
            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        else:
            avg_epoch_loss = 0.0

        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)


