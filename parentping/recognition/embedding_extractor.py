from __future__ import annotations

import cv2
import numpy as np
import torch


class EmbeddingExtractor:
    def __init__(self, model: torch.nn.Module, device: torch.device, input_size: int = 160) -> None:
        self.model = model
        self.device = device
        self.input_size = input_size

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (self.input_size, self.input_size))
        face = face_rgb.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
        return tensor

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess(face_bgr)
        with torch.no_grad():
            embedding = self.model(input_tensor)[0]
        return embedding.detach().cpu().numpy().astype(np.float32)

