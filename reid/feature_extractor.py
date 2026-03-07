"""
reid/feature_extractor.py
--------------------------
MobileNetV3-Small backbone + 512-d embedding head.
Pure torchvision — no torchreid, no Cython, no build step.
Works on Python 3.10 CPU out of the box.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from typing import List, Optional


REID_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


class EmbeddingHead(nn.Module):
    def __init__(self, in_features: int = 576, embed_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=1)


class ReIDNet(nn.Module):
    BACKBONE_OUT = 576

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights  = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool     = backbone.avgpool
        self.head     = EmbeddingHead(self.BACKBONE_OUT, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class FeatureExtractor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 8,
        embed_dim: int = 512,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        self.model = ReIDNet(embed_dim=embed_dim, pretrained=(weights_path is None))

        if weights_path and Path(weights_path).exists():
            ckpt  = torch.load(weights_path, map_location=self.device)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[ReID] Loaded weights ← {weights_path}")
        else:
            print(f"[ReID] MobileNetV3 ImageNet pretrained  (device={self.device})")

        self.model.to(self.device).eval()

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(torch.zeros(3, 256, 128))
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(REID_TRANSFORM(rgb))
        all_feats = []
        for i in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[i:i+self.batch_size]).to(self.device)
            all_feats.append(self.model(batch).cpu().numpy())
        return np.vstack(all_feats).astype(np.float32)

    def extract_single(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        return self.extract([crop])[0]
