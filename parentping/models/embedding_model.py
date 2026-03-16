from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ArcFaceEmbeddingModel(nn.Module):
    """ResNet18 backbone + 512D embedding head for ArcFace inference."""

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint
    raise ValueError("Unsupported checkpoint format for model weights.")


def load_embedding_model(
    weights_path: str | Path,
    embedding_size: int = 512,
    device: str | None = None,
) -> Tuple[ArcFaceEmbeddingModel, torch.device]:
    model = ArcFaceEmbeddingModel(embedding_size=embedding_size)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_weights_path = Path(weights_path).expanduser().resolve()

    if not resolved_weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {resolved_weights_path}. "
            "Pass an absolute path to best_resnet18_arcface_parentping.pth."
        )

    checkpoint = torch.load(str(resolved_weights_path), map_location=selected_device)
    raw_state = _extract_state_dict(checkpoint)

    cleaned_state: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()
    for key, value in raw_state.items():
        normalized_key = key.replace("module.", "")
        if normalized_key == "embedding.weight":
            normalized_key = "embedding.0.weight"
        elif normalized_key == "embedding.bias":
            normalized_key = "embedding.0.bias"
        if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
            cleaned_state[normalized_key] = value

    if not cleaned_state:
        raise RuntimeError(
            "No compatible backbone/embedding weights found in checkpoint. "
            "Expected keys for ResNet18 backbone + embedding layer only."
        )

    model.load_state_dict(cleaned_state, strict=False)
    model.to(selected_device)
    model.eval()
    return model, selected_device
