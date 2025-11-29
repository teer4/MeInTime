from typing import List
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
from torchvision import transforms
import torch.nn.functional as F
from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline
from ..model import SwinIR
from insightface.app import FaceAnalysis
import cv2


class BFRInferenceLoop(InferenceLoop):
    def __init__(self, args, provider=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        super().__init__(args)  # 传入 args 给父类
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.app = FaceAnalysis(name="antelopev2", root="./", providers=provider)
        self.app.prepare(ctx_id=0, det_size=(512, 512))  # 根据需要调整

    def load_cleaner(self) -> None:
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load("configs/inference/swinir.yaml")
        )
        #weight = load_model_from_url(MODELS["swinir_face"])
        cfg = OmegaConf.load("configs/inference/inference.yaml")
        weight = torch.load(cfg.infer.swinir_path, map_location="cpu")
        if "state_dict" in weight:
            weight = weight["state_dict"]
        weight = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in weight.items()}
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
    
    def get_ref_embedding(self, ref_imgs: List[Image.Image]) -> torch.Tensor:
        emb_list = []
        for img in ref_imgs:
            np_img = np.array(img)  # (H,W,3), RGB
            face = self.app.get(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
            if len(face) == 0:
                continue
            emb = face[0].normed_embedding  # numpy [512,]
            emb_list.append(emb.astype(np.float32))

        if len(emb_list) == 0:
            raise ValueError("No face detected in any reference image")

        emb_avg = np.mean(emb_list, axis=0)
        emb_norm = emb_avg / np.linalg.norm(emb_avg)  # L2 normalize
        return torch.tensor(emb_norm).unsqueeze(0).to(self.device)  # shape: [1, 512]