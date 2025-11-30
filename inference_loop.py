import os
from typing import Generator, List, Tuple
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd

from cldm.utils.common import (
    instantiate_from_config,
    VRAMPeakMonitor,
)
from cldm.pipeline import Pipeline, SwinIRPipeline
from cldm.model import ControlLDM, Diffusion, SwinIR
from insightface.app import FaceAnalysis
import cv2


class InferenceLoop:
    def __init__(
        self,
        args: Namespace,
        provider=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.infer_cfg = OmegaConf.load("configs/inference/inference.yaml")

        with VRAMPeakMonitor("loading cleaner model"):
            self.load_cleaner()
        with VRAMPeakMonitor("loading cldm model"):
            self.load_cldm()
        self.load_pipeline()

        self.app = FaceAnalysis(name="antelopev2", root="./", providers=provider)
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    def load_cleaner(self) -> None:
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load("configs/inference/swinir.yaml")
        )

        weight = torch.load(self.infer_cfg.infer.swinir_path, map_location="cpu")
        if "state_dict" in weight:
            weight = weight["state_dict"]
        weight = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in weight.items()
        }
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_cldm(self) -> None:
        self.cldm: ControlLDM = instantiate_from_config(
            OmegaConf.load("configs/inference/cldm.yaml")
        ).cpu()

        with torch.no_grad():
            sd_ckpt = torch.load(self.infer_cfg.infer.sd_path, map_location="cpu")
            sd_weight = sd_ckpt["state_dict"]
            del sd_ckpt
            unused, missing = self.cldm.load_pretrained_sd(sd_weight)
            print(f"load pretrained stable diffusion, unused: {unused}, missing: {missing}")
            del sd_weight

        with torch.no_grad():
            control_ckpt = torch.load(
                self.infer_cfg.infer.controlnet_path, map_location="cpu"
            )
            print(
                "load controlnet v2 weight from",
                self.infer_cfg.infer.controlnet_path,
            )
            self.cldm.controlnet.load_state_dict(control_ckpt, strict=False)
            print("load controlnet weight")
            del control_ckpt

        with torch.no_grad():
            adapter_ckpt = torch.load(
                self.infer_cfg.infer.adapter_path, map_location="cpu"
            )
            self.cldm.load_adapter_from_ckpt(adapter_ckpt)
            print(
                "Loaded Face-Adapter (image_proj_model & to_k_ip / to_v_ip, and control_fusions)"
            )
            del adapter_ckpt

        cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]
        self.cldm.cast_dtype(cast_type)

        self.cldm.eval().to(self.args.device)

        config = "configs/inference/diffusion.yaml"
        self.diffusion: Diffusion = instantiate_from_config(OmegaConf.load(config))
        self.diffusion.to(self.args.device)

    def load_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return np.array(lq)

    def get_ref_embedding(self, ref_imgs: List[Image.Image]) -> torch.Tensor:
        emb_list = []
        for img in ref_imgs:
            np_img = np.array(img)
            face = self.app.get(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
            if len(face) == 0:
                continue
            emb = face[0].normed_embedding
            emb_list.append(emb.astype(np.float32))

        if len(emb_list) == 0:
            raise ValueError("No face detected in any reference image")

        emb_avg = np.mean(emb_list, axis=0)
        emb_norm = emb_avg / np.linalg.norm(emb_avg)
        return torch.tensor(emb_norm).unsqueeze(0).to(self.device)

    def setup(self) -> None:
        self.save_dir = self.args.output
        os.makedirs(self.save_dir, exist_ok=True)
        self.trg_prompt_dict = self._load_trg_prompt_list(self.args.trg_prompt_list)

    def load_lq_and_ref(
        self,
    ) -> Generator[Tuple[Image.Image, List[Image.Image]], None, None]:
        img_exts = [".png", ".jpg", ".jpeg"]
        assert os.path.isdir(self.args.input), "input folder not found"
        assert os.path.isdir(self.args.ref), "ref folder not found"

        lq_files = sorted(os.listdir(self.args.input))
        lq_dict = {
            os.path.splitext(f)[0]: f
            for f in lq_files
            if os.path.splitext(f)[1].lower() in img_exts
        }

        for stem, lq_file in lq_dict.items():
            lq_path = os.path.join(self.args.input, lq_file)
            ref_dir = os.path.join(self.args.ref, stem)
            assert os.path.isdir(ref_dir), f"Missing ref folder for: {stem}"

            ref_files = sorted(
                [
                    os.path.join(ref_dir, f)
                    for f in os.listdir(ref_dir)
                    if os.path.splitext(f)[1].lower() in img_exts
                ]
            )
            ref_imgs = [Image.open(p).convert("RGB") for p in ref_files]

            print(f"[pair] lq: {lq_path}, refs: {len(ref_imgs)} images")
            lq = Image.open(lq_path).convert("RGB")
            self.loop_ctx["file_stem"] = stem
            yield lq, ref_imgs

    def _load_trg_prompt_list(self, path):
        if not path or not os.path.exists(path):
            return {}
        prompt_dict = {}
        with open(path, "r") as f:
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
            src_prompt = ", ".join(
                [text for text in [self.args.src_prompt] if text]
            )

            identity = self.loop_ctx["file_stem"]

            if self.args.trg_prompt_list:
                if identity not in self.trg_prompt_dict:
                    print(
                        f"[Warning] Identity '{identity}' not found in trg_prompt_list. Skipping."
                    )
                    continue
                individual_prompt = self.trg_prompt_dict[identity]
            else:
                individual_prompt = self.args.trg_prompt

            trg_prompt = ", ".join(
                [text for text in [individual_prompt] if text]
            )

            neg_prompt = self.args.neg_prompt

            lq = self.after_load_lq(lq_img)

            ref_emb = self.get_ref_embedding(ref_imgs)

            n_samples = self.args.n_samples
            batch_size = self.args.batch_size
            num_batches = (n_samples + batch_size - 1) // batch_size
            samples = []

            for i in range(num_batches):
                n_inputs = min((i + 1) * batch_size, n_samples) - i * batch_size
                with torch.autocast(self.args.device, auto_cast_type):
                    batch_samples = self.pipeline.run(
                        np.tile(lq[None], (n_inputs, 1, 1, 1)),
                        self.args.steps,
                        self.args.strength,
                        self.args.dds_steps,
                        self.args.age_guidance,
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
                        ref_emb=ref_emb.repeat(n_inputs, 1),
                    )
                samples.extend(list(batch_samples))
            self.save(samples, src_prompt, trg_prompt, neg_prompt)

    def save(
        self, samples: List[np.ndarray], src_prompt: str, trg_prompt: str, neg_prompt: str
    ) -> None:
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
