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
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler

from diffbir.dataset.face_adapter_dataset import FaceAdapterDataset

from diffbir.loss.identity_loss import IdentityLoss

from eval import evaluate_on_val


def main(args) -> None:
    # Setup accelerator
    accelerator = Accelerator(split_batches=True ) #gpu_ids=[0]
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
    resume_path = getattr(cfg.train, "resume_path", None)
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        cldm.image_proj_model.load_state_dict(ckpt["image_proj_model"])
        for name, module in cldm.unet.named_modules():
            if name in ckpt["ip_attn"]:
                module.load_state_dict(ckpt["ip_attn"][name])
        if accelerator.is_main_process:
            print(f"[Resume] Loaded image_proj_model & ip_attn from {resume_path}")
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

    # 合并 image_proj_model 和所有 to_k_ip / to_v_ip 参数
    params_to_opt = itertools.chain(
        cldm.image_proj_model.parameters(),
        to_kv_ip_params  # ✅ 现在是纯 list，无需展开 *
    )
    optimizer = torch.optim.AdamW(params_to_opt, lr=cfg.train.learning_rate)

    # ⚠️ 只对 image_proj_model 和 to_k_ip/to_v_ip 使用 EMA
    ema_params = itertools.chain(
        cldm.image_proj_model.parameters(),
        to_kv_ip_params  # 你之前已构建此参数列表
    )
    ema = ExponentialMovingAverage(ema_params, decay=cfg.train.ema_decay)
    ema.to(device)

    # Setup data:
    collate_fn = FaceAdapterDataset.collate_fn  # ✅ 注意：是类，不是实例
    train_dataset = instantiate_from_config(cfg.dataset.train)
    train_loader = DataLoader(train_dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=12,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn 
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
    cldm.train().to(device)       
    swinir.eval().to(device)     
    diffusion.to(device)
    
    cldm, optimizer, train_loader = accelerator.prepare(cldm, optimizer, train_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    noise_aug_timestep = cfg.train.noise_aug_timestep

    identity_loss = IdentityLoss(sqrt_alphas_cumprod=diffusion.sqrt_alphas_cumprod.clone().detach().to(device))

    # Setup training loop
    global_step = 0
    max_steps = cfg.train.train_steps  # e.g., 50000
    epoch = 0
    step_loss = []
    epoch_loss = []
    step_diffusion_loss = []
    step_id_loss = []

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
            loss_diffusion, loss_id = diffusion.p_losses(
                cldm,               # model (含 face-adapter、UNet、VAE)
                z_0,                # latent 表示的 x_start
                t,                  # 当前 timestep
                cond_aug,           # 条件输入
                image_prompt,       # prompt
                x_start_pixel=gt,   # 原图像，用于 ID Loss
                identity_loss_fn=identity_loss,
            )
            
            # 加权合成总损失
            lambda_id = cfg.train.lambda_id if hasattr(cfg.train, "lambda_id") else 1.0
            if loss_id is not None:
                total_loss = loss_diffusion + lambda_id * loss_id
            else:
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
            step_diffusion_loss.append(loss_diffusion.item())  # 若需单独可视化
            
            if loss_id is not None:
                step_id_loss.append(loss_id.item())
            

            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Step: {global_step:07d}, "
                f"Total Loss: {total_loss.item():.6f}, "
                f"Diffusion Loss: {loss_diffusion.item():.6f}, "
                #f"ID Loss: {loss_id.item():.6f}" if loss_id else ""
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # diffusion loss
                    avg_diff = accelerator.gather(
                        torch.tensor(step_diffusion_loss, device=device).unsqueeze(0)
                    ).mean().item()
                    writer.add_scalar("loss/diffusion", avg_diff, global_step)
                    step_diffusion_loss.clear()

                    
                    # identity loss
                    if loss_id is not None and step_id_loss:
                        avg_id = accelerator.gather(
                            torch.tensor(step_id_loss, device=device).unsqueeze(0)
                        ).mean().item()
                        writer.add_scalar("loss/identity", avg_id, global_step)
                        step_id_loss.clear()
                    else:
                        avg_id = 0.0  # 若未启用 ID Loss，则默认置 0
                    
                    # total loss = diffusion + lambda_id * identity
                    avg_total = accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    ).mean().item()
                    writer.add_scalar("loss/total", avg_total, global_step)
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
                        }
                    }
                    ckpt_path = os.path.join(ckpt_dir, f"{global_step:07d}.pt")
                    torch.save(ckpt, ckpt_path)

            #sample
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
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                    if accelerator.is_main_process:
                        writer.add_text(
                            "identity",                # 标签名
                            "\n".join(log_identity),   # 将 8 个 ID 用换行连接
                            global_step
                        )
                ema.restore()
                cldm.train()

            """
            if global_step % cfg.train.eval_every == 0 and global_step > 0:
                metrics = evaluate_on_val(
                    val_loader=val_loader,
                    model=cldm,
                    swinir=swinir,
                    diffusion=diffusion,
                    identity_loss_fn=identity_loss,
                    sampler=sampler,
                    global_step=global_step,
                    writer=writer,
                    device=device,
                    ema=ema,
                    max_batches=3,
                )
                if accelerator.is_main_process:
                    print(f"[Eval@{global_step}] IDS: {metrics['IDS']:.4f}, LPIPS: {metrics['LPIPS']:.4f}, "
                        f"NIQE: {metrics['NIQE']:.4f},"
                        #f"MUSIQ: {metrics['MUSIQ']:.4f}"
                        f"PSNR: {metrics['PSNR']:.4f}, SSIM: {metrics['SSIM']:.4f}")
                    pbar.refresh()  # ✅ 验证后刷新主进度条
            """
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
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


