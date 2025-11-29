import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from insightface.app import FaceAnalysis

from line_profiler import LineProfiler

class IdentityEncoder(nn.Module):
    def __init__(self, provider = [('CUDAExecutionProvider', {
    'arena_extend_strategy': 'kSameAsRequested',
    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 2GB
})]): #, "CPUExecutionProvider"
        super().__init__()
        self.app = FaceAnalysis(name="antelopev2", root="./", providers=provider)
        self.app.prepare(ctx_id=0, det_size=(512, 512))  # 可根据需要调整

    @torch.no_grad()
    #@profile
    def forward(self, img_tensor):  # [B, 3, H, W], in [-1, 1]
        img_np = ((img_tensor + 1) * 127.5).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        img_np = img_np[..., ::-1]  # RGB → BGR for insightface
        feats = []
        device = img_tensor.device
        zero_padding = torch.zeros(512, device=device) 
        for i in range(img_np.shape[0]):
            faces = self.app.get(img_np[i])
            if faces and faces[0].normed_embedding is not None:
                #feats.append(torch.tensor(faces[0].normed_embedding))
                feats.append(torch.tensor(faces[0].normed_embedding, device=device))
            else:
                #feats.append(torch.zeros(512))  # fallback embedding
                feats.append(zero_padding)  # fallback embedding on the same device
        return torch.stack(feats).to(device)

class IdentityLoss(nn.Module):
    def __init__(self, sqrt_alphas_cumprod: torch.Tensor):
        super().__init__()
        self.encoder = IdentityEncoder()
        # 注册为 buffer 以确保会随模型迁移 device，参与保存与加载
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.clone().detach().float())
    
    def forward(self, x0_pred, x0_gt, t):  # [B, 3, 512, 512], t: LongTensor[B]
        #torch.cuda.synchronize() 
        feat_pred = self.encoder(x0_pred)  # [B, 512]
        #torch.cuda.synchronize() 
        feat_gt = self.encoder(x0_gt)
        #torch.cuda.synchronize()

        cosine_sim = F.cosine_similarity(feat_pred, feat_gt, dim=1)  # [B]
        #torch.cuda.synchronize()
        raw_loss = 1.0 - cosine_sim  # [B]
        #torch.cuda.synchronize()

        scale = self.sqrt_alphas_cumprod[t]  # [B]
        #torch.cuda.synchronize()
        loss = (scale * raw_loss).mean()     # scalar
        #torch.cuda.synchronize()
        return loss
