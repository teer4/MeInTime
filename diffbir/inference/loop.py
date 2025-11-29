import os
from typing import overload, Generator, List, Tuple
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd

from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
    VRAMPeakMonitor,
)
from .pretrained_models import MODELS
from ..pipeline import Pipeline
from ..utils.cond_fn import MSEGuidance, WeightedMSEGuidance
from ..model import ControlLDM, Diffusion
from ..utils.caption import (
    LLaVACaptioner,
    EmptyCaptioner,
    RAMCaptioner,
    LLAVA_AVAILABLE,
    RAM_AVAILABLE,
)


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        #还没有实例化Pipeline
        self.pipeline: Pipeline = None
        with VRAMPeakMonitor("loading cleaner model"):
            self.load_cleaner()
        with VRAMPeakMonitor("loading cldm model"):
            self.load_cldm()
        self.load_cond_fn()
        self.load_pipeline()
        with VRAMPeakMonitor("loading captioner"):
            self.load_captioner()

    @overload
    def load_cleaner(self) -> None: ...

    def load_cldm(self) -> None:
        # ------------------------
        # 0. 强制先在 CPU 上构建模型，避免 GPU 初始化峰值
        # ------------------------
        self.cldm: ControlLDM = instantiate_from_config(
            OmegaConf.load("configs/inference/cldm.yaml")
        ).cpu()

        # ==============================
        # 1. load Stable Diffusion weight
        # ==============================
        with torch.no_grad():   # ← 防止 load 时产生梯度占用显存
            if self.args.version == "v2.1":
                sd_weight = load_model_from_url(MODELS["sd_v2.1_zsnr"])
            else:
                cfg = OmegaConf.load("configs/inference/inference.yaml")
                sd_ckpt = torch.load(cfg.infer.sd_path, map_location="cpu")
                sd_weight = sd_ckpt["state_dict"]
                del sd_ckpt
                print("load sd v2 weight from", cfg.infer.sd_path)

            unused, missing = self.cldm.load_pretrained_sd(sd_weight)
            print(f"load pretrained stable diffusion, unused: {unused}, missing: {missing}")
            del sd_weight  # ← 立即释放 CPU 显存

        # ==============================
        # 2. load ControlNet
        # ==============================
        with torch.no_grad():
            if self.args.version == "v2":
                cfg = OmegaConf.load("configs/inference/inference.yaml")
                control_ckpt = torch.load(cfg.infer.controlnet_path, map_location="cpu")
                print("load controlnet v2 weight from", cfg.infer.controlnet_path)
            elif self.args.version == "v1":
                control_ckpt = load_model_from_url(
                    MODELS["v1_face" if self.args.task == "face" else "v1_general"]
                )
            else:
                control_ckpt = load_model_from_url(MODELS["v2.1"])

            # 关键修改：strict=False → 避免权重检查导致额外内存开销
            self.cldm.controlnet.load_state_dict(control_ckpt, strict=False)
            print("load controlnet weight")
            del control_ckpt

        # ==============================
        # 3. load FaceAdapter (image_proj_model + to_k_ip/to_v_ip)
        # ==============================
        with torch.no_grad():
            cfg = OmegaConf.load("configs/inference/inference.yaml")
            adapter_ckpt = torch.load(cfg.infer.adapter_path, map_location="cpu")
            self.cldm.load_adapter_from_ckpt(adapter_ckpt)
            print("Loaded Face-Adapter (image_proj_model & to_k_ip / to_v_ip, and control_fusions)")
            del adapter_ckpt

        # ==============================
        # 4. dtype 强制转换（避免巨大显存占用）
        # ==============================
        cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]
        self.cldm.cast_dtype(cast_type)

        # ==============================
        # 5. 最后再放到 GPU
        # ==============================
        self.cldm.eval().to(self.args.device)

        # ==============================
        # 6. load diffusion module (几乎不占内存)
        # ==============================
        if self.args.version in ["v1", "v2"]:
            config = "configs/inference/diffusion.yaml"
        else:
            config = "configs/inference/diffusion_v2.1.yaml"

        self.diffusion: Diffusion = instantiate_from_config(
            OmegaConf.load(config)
        )
        self.diffusion.to(self.args.device)


        #self.verify_loaded_modules()

    """
    def verify_loaded_modules(self):
        print("===== ✅ VERIFY WEIGHTS =====")
        # 1. image_proj_model
        for name, param in self.cldm.image_proj_model.named_parameters():
            print(f"[Resampler] {name}: mean={param.data.mean():.4f}, grad={param.requires_grad}")
        
        # 2. ControlNet 检查 hint adapter 层
        for name, module in self.cldm.controlnet.named_modules():

            print(f"[ControlNet] {name} exists.")

        # 3. UNet 中 attention 层的 to_k_ip, to_v_ip
        for name, mod in self.cldm.unet.named_modules():
            if hasattr(mod, "to_k_ip"):
                print(f"[UNet attn] {name}.to_k_ip loaded, mean={mod.to_k_ip.weight.data.mean():.4f}")
            if hasattr(mod, "to_v_ip"):
                print(f"[UNet attn] {name}.to_v_ip loaded, mean={mod.to_v_ip.weight.data.mean():.4f}")
        
        print("===== ✅ VERIFY DONE =====")
    """

    def load_cond_fn(self) -> None:
        if not self.args.guidance:
            self.cond_fn = None
            return
        if self.args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.args.g_loss)
        self.cond_fn = cond_fn_cls(
            self.args.g_scale,
            self.args.g_start,
            self.args.g_stop,
            self.args.g_space,
            self.args.g_repeat,
        )

    @overload
    def load_pipeline(self) -> None: ...

    def load_captioner(self) -> None:
        if self.args.captioner == "none":
            self.captioner = EmptyCaptioner(self.args.device)
        elif self.args.captioner == "llava":
            assert LLAVA_AVAILABLE, "llava is not available in your environment."
            self.captioner = LLaVACaptioner(self.args.device, self.args.llava_bit)
        elif self.args.captioner == "ram":
            assert RAM_AVAILABLE, "ram is not available in your environment."
            self.captioner = RAMCaptioner(self.args.device)
        else:
            raise ValueError(f"unsupported captioner: {self.args.captioner}")

    def setup(self) -> None:
        self.save_dir = self.args.output
        os.makedirs(self.save_dir, exist_ok=True)
        self.trg_prompt_dict = self._load_trg_prompt_list(self.args.trg_prompt_list)


    def load_lq_and_ref(self) -> Generator[Tuple[Image.Image, List[Image.Image]], None, None]:
        img_exts = [".png", ".jpg", ".jpeg"]
        assert os.path.isdir(self.args.input), "input folder not found"
        assert os.path.isdir(self.args.ref), "ref folder not found"

        lq_files = sorted(os.listdir(self.args.input))
        lq_dict = {os.path.splitext(f)[0]: f for f in lq_files if os.path.splitext(f)[1].lower() in img_exts}

        for stem, lq_file in lq_dict.items():
            lq_path = os.path.join(self.args.input, lq_file)
            ref_dir = os.path.join(self.args.ref, stem)
            assert os.path.isdir(ref_dir), f"Missing ref folder for: {stem}"

            ref_files = sorted([
                os.path.join(ref_dir, f) for f in os.listdir(ref_dir)
                if os.path.splitext(f)[1].lower() in img_exts
            ])
            ref_imgs = [Image.open(p).convert("RGB") for p in ref_files]

            print(f"[pair] lq: {lq_path}, refs: {len(ref_imgs)} images")
            lq = Image.open(lq_path).convert("RGB")
            self.loop_ctx["file_stem"] = stem
            yield lq, ref_imgs

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        return np.array(lq)

    def _load_trg_prompt_list(self, path):
        if not path or not os.path.exists(path):
            return {}
        prompt_dict = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    identity, prompt = parts
                    prompt_dict[identity] = prompt
        return prompt_dict


    @torch.no_grad()
    def run(self) -> None:
        self.setup()

        auto_cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]

        for lq_img, ref_imgs in self.load_lq_and_ref():
            # prompt 构建
            with VRAMPeakMonitor("applying captioner"):
                caption = self.captioner(lq_img)
            src_prompt = ", ".join(
                [text for text in [caption, self.args.src_prompt] if text]
            )  

            # 提取身份名
            identity = self.loop_ctx["file_stem"]

            # 判定使用批量prompt还是单一全局prompt
            if self.args.trg_prompt_list:
                # 有 prompt list 时必须强制使用它
                if identity not in self.trg_prompt_dict:
                    print(f"[Warning] Identity '{identity}' not found in trg_prompt_list. Skipping.")
                    continue  # 或 raise Exception() 取决于你容忍程度
                individual_prompt = self.trg_prompt_dict[identity]
            else:
                individual_prompt = self.args.trg_prompt

            # 构造 trg_prompt
            trg_prompt = ", ".join(
                [text for text in [caption, individual_prompt] if text]
            )


            neg_prompt = self.args.neg_prompt

            # lq 处理
            lq = self.after_load_lq(lq_img)

            # 提取 embedding
            ref_emb = self.get_ref_embedding(ref_imgs)  # torch.Size([1, 512])

            # 批处理（和原来一样）
            n_samples = self.args.n_samples        #总共需要生成的样本数量
            batch_size = self.args.batch_size       #每个批次中的图像数量
            num_batches = (n_samples + batch_size - 1) // batch_size   #模拟向上取整[n_samples / batch_size], 计算一共需要处理几个批次
            samples = []

            for i in range(num_batches):
                n_inputs = min((i + 1) * batch_size, n_samples) - i * batch_size
                with torch.autocast(self.args.device, auto_cast_type):
                    batch_samples = self.pipeline.run(
                        np.tile(lq[None], (n_inputs, 1, 1, 1)),
                        self.args.steps,
                        self.args.strength,
                        self.args.cleaner_tiled,
                        self.args.cleaner_tile_size,
                        self.args.cleaner_tile_stride,
                        self.args.vae_encoder_tiled,
                        self.args.vae_encoder_tile_size,
                        self.args.vae_decoder_tiled,
                        self.args.vae_decoder_tile_size,
                        self.args.cldm_tiled,
                        self.args.cldm_tile_size,
                        self.args.cldm_tile_stride,
                        src_prompt,
                        trg_prompt,
                        neg_prompt,
                        self.args.cfg_scale,
                        self.args.start_point_type,
                        self.args.sampler,
                        self.args.noise_aug,
                        self.args.rescale_cfg,
                        self.args.s_churn,
                        self.args.s_tmin,
                        self.args.s_tmax,
                        self.args.s_noise,
                        self.args.eta,
                        self.args.order,
                        ref_emb=ref_emb.repeat(n_inputs, 1),  # 注意：此处需要 pipeline.run 支持 ref_emb 参数
                    )
                samples.extend(list(batch_samples))
            self.save(samples, src_prompt, trg_prompt, neg_prompt)


    def save(self, samples: List[np.ndarray], src_prompt: str, trg_prompt: str, neg_prompt: str) -> None:
        file_stem = self.loop_ctx["file_stem"]
        assert len(samples) == self.args.n_samples
        for i, sample in enumerate(samples):
            file_name = (
                f"{file_stem}_{i}.png"
                if self.args.n_samples > 1
                else f"{file_stem}.png"
            )
            save_path = os.path.join(self.save_dir, file_name)
            Image.fromarray(sample).save(save_path)
            print(f"save result to {save_path}")
        csv_path = os.path.join(self.save_dir, "prompt.csv")
        df = pd.DataFrame(
            {
                "file_name": [file_stem],
                "src_prompt": [src_prompt],
                "trg_prompt": [trg_prompt],
                "neg_prompt": [neg_prompt],
            }
        )
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=None, mode="a", header=None)
        else:
            df.to_csv(csv_path, index=None)
