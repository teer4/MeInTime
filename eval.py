import torch
import random
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from pyiqa import create_metric  
from itertools import islice


def to_unit_range(x):
    """将 [-1, 1] 映射到 [0, 1] 并限制范围，防止生成图溢出造成断言错误"""
    return ((x + 1) / 2).clamp(0, 1)


@torch.no_grad()
def evaluate_on_val(val_loader, model, diffusion, swinir, identity_loss_fn, sampler, global_step, writer, device, ema=None, log_image_num=8, max_batches=3):
    model.eval()
    if ema is not None:
        ema.store()
        ema.copy_to()

    # 初始化指标
    lpips_metric = create_metric("lpips-vgg", device=device)
    niqe_metric = create_metric("niqe", device=device)
    #musiq_metric = create_metric("musiq", device=device)
    psnr_metric = create_metric("psnr", device=device)
    ssim_metric = create_metric("ssim", device=device)


     # ✅ 动态跳过随机数目的 batch，再取 max_batches 个
    skip = random.randint(0, len(val_loader) - max_batches - 1)
    selected_batches = islice(val_loader, skip, skip + max_batches)


    ids_scores, lpips_scores, niqe_scores, musiq_scores, psnr_scores, ssim_scores = [], [], [], [], [], []
    sample_logged = False

    for batch in tqdm(selected_batches, desc=f"[Eval@{global_step}]", leave=False, position=1):


        gt = batch['gt'].to(device)
        lq = batch['lq'].to(device)
        emb = batch['embedding'].to(device)
        prompt = batch['prompt']
        identity = batch['identity']

        z_0 = model.vae_encode(gt)
        clean = swinir(lq)
        cond = model.prepare_condition(clean, prompt)
        image_prompt = model.encode_image_prompt(emb)
        t = torch.randint(0, diffusion.num_timesteps, (z_0.size(0),), device=device).long()

        log_clean = clean[:log_image_num]
        log_cond = {k: v[:log_image_num] for k, v in cond.items()}
        log_gt, log_lq = gt[:log_image_num], lq[:log_image_num]
        log_prompt = prompt[:log_image_num]
        log_image_prompt = image_prompt[:log_image_num]
        log_identity = batch["identity"][:log_image_num]

        z = sampler.sample(
            model=model,
            device=device,
            steps=50,
            x_size=(len(log_gt), *z_0.shape[1:]),
            cond=log_cond,
            uncond=None,
            image_prompt=log_image_prompt,
            cfg_scale=1.0,
            progress=False
        )

        pred_img = model.vae_decode(z).detach()

        if not sample_logged:
            for tag, image in [
                ("val/image/samples", to_unit_range(pred_img)),
                ("val/image/gt", to_unit_range(log_gt)),
                ("val/image/lq", log_lq),
            ]:
                writer.add_image(tag, make_grid(image, nrow=4), global_step)
            writer.add_text("val/identity", "\n".join(log_identity), global_step)
            sample_logged = True

        # 图像值域转换 [−1, 1] → [0, 1]
        pred = to_unit_range(pred_img)
        target = to_unit_range(log_gt)
        t_eval = torch.zeros((log_gt.shape[0],), device=device, dtype=torch.long)



        # Identity Loss
        try:
            ids_score = identity_loss_fn(pred_img, log_gt, t_eval).mean().item()
        except Exception as e:
            print("IDS error:", e)
            ids_score = 0.0

        # Quality Metrics
        def safe_metric(metric_func, *args):
            try:
                return metric_func(*args).mean().item()
            except Exception as e:
                print(f"{metric_func.__name__} error:", e)
                return 0.0

        lpips_score = safe_metric(lpips_metric, pred, target)
        niqe_score = safe_metric(niqe_metric, pred)
        #musiq_score = safe_metric(musiq_metric, pred)
        psnr_score = safe_metric(psnr_metric, pred, target)
        ssim_score = safe_metric(ssim_metric, pred, target)


        ids_scores.append(ids_score)
        lpips_scores.append(lpips_score)
        niqe_scores.append(niqe_score)
        #musiq_scores.append(musiq_score)
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)


    # 均值汇总
    mean_scores = {
        "IDS": float(np.mean(ids_scores)),
        "LPIPS": float(np.mean(lpips_scores)),
        "NIQE": float(np.mean(niqe_scores)),
        #"MUSIQ": float(np.mean(musiq_scores)),
        "PSNR": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores)),
    }

    for key, value in mean_scores.items():
        writer.add_scalar(f"val/{key}", value, global_step)

    if ema is not None:
        ema.restore()
    model.train()
    return mean_scores
