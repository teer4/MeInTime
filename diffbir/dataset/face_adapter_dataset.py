import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from einops import rearrange

class FaceAdapterDataset(Dataset):
    def __init__(self, jsonl_path, root_dir, prompt_text="photo of a person"):
        """
        参数:
            jsonl_path: JSONL 文件路径，每行是一个三元组信息
            root_dir: 数据根目录，包含 gt/、lq/、embedding/
            prompt_text: 默认的文本提示
        """
        self.root_dir = root_dir
        self.prompt_text = prompt_text
        self.data = []

        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]
        identity = entry["identity"]

        # ========== 图像读取 ==========
        gt_path = os.path.join(self.root_dir, entry["gt"])
        lq_path = os.path.join(self.root_dir, entry["lq"])

        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        lq = cv2.imread(lq_path)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0        # [0, 1]

        # ========== 多个 embedding 加载 + 融合 ==========
        emb_list = []
        for rel_path in entry["embedding_refs"]:
            emb_path = os.path.join(self.root_dir, rel_path)
            emb = np.load(emb_path).astype(np.float32)
            emb_list.append(emb)

        emb_avg = np.mean(emb_list, axis=0)
        emb_norm = emb_avg / np.linalg.norm(emb_avg)

        return {
            "identity": identity,
            "gt": gt,                       # (H, W, 3), float32, RGB, [-1,1]
            "lq": lq,                       # (H, W, 3), float32, RGB, [0,1]
            "embedding": emb_norm,         # (512,), float32
            "prompt": self.prompt_text     # str
        }

    @staticmethod
    def collate_fn(batch):
        """
        将 batch 转换为 PyTorch tensor
        输出结构：
            gt:        [B, 3, 512, 512], float32
            lq:        [B, 3, 512, 512], float32
            embedding: [B, 512], float32
            prompt:    List[str]
            identity:  List[str]
        """
        gt = torch.stack([torch.from_numpy(item["gt"]) for item in batch], dim=0)
        gt = rearrange(gt, "b h w c -> b c h w").float()

        lq = torch.stack([torch.from_numpy(item["lq"]) for item in batch], dim=0)
        lq = rearrange(lq, "b h w c -> b c h w").float()

        embedding = torch.stack([torch.from_numpy(item["embedding"]) for item in batch], dim=0)
        prompt = [item["prompt"] for item in batch]
        identity = [item["identity"] for item in batch]

        return {
            "gt": gt,                   # torch.FloatTensor [B, 3, 512, 512]
            "lq": lq,                   # torch.FloatTensor [B, 3, 512, 512]
            "embedding": embedding,     # torch.FloatTensor [B, 512]
            "prompt": prompt,           # List[str]
            "identity": identity        # List[str]
        }
