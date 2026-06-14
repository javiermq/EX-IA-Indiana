from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121
from transformers import AutoModel, AutoTokenizer


class DenseNet121Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        self.out_dim = backbone.classifier.in_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.features(x)
        return F.relu(fmap, inplace=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.forward_features(x)
        pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        return pooled, fmap


class ResidualMLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.block(h)
        return self.out(h)


class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = DenseNet121Encoder(pretrained=pretrained)
        self.head = ResidualMLPHead(self.encoder.out_dim, num_classes, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled, fmap = self.encoder(x)
        logits = self.head(pooled)
        return logits, pooled, fmap


class AttentionDecoder(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 4, 1, kernel_size=1),
        )

    def forward(self, fmap: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        attn = self.net(fmap)
        attn = F.interpolate(attn, size=out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(attn)


class VisualProjector(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, gradcam_dim: int = 0, hidden_dim: int = 512) -> None:
        super().__init__()
        in_dim = visual_dim + gradcam_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, visual: torch.Tensor, gradcam: torch.Tensor | None = None) -> torch.Tensor:
        if gradcam is not None:
            visual = torch.cat([visual, gradcam], dim=-1)
        return F.normalize(self.net(visual), dim=-1)


@dataclass
class QwenTextEncoderConfig:
    model_id: str
    device: torch.device
    dtype: torch.dtype | None = None


class QwenTextEncoder(nn.Module):
    def __init__(self, cfg: QwenTextEncoderConfig) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype).to(cfg.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.out_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled.float(), dim=-1)


class GradCAM:
    def __init__(self, model: DenseNetClassifier) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target = self.model.encoder.features.denseblock4
        self.forward_handle = target.register_forward_hook(self._save_activation)
        self.backward_handle = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, _module: nn.Module, _grad_input: tuple[torch.Tensor], grad_output: tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0]

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, images: torch.Tensor, class_idx: torch.Tensor | None = None) -> torch.Tensor:
        logits, _, _ = self.model(images)
        if class_idx is None:
            class_idx = logits.sigmoid().argmax(dim=1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-6)


def clip_contrastive_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    scale = temperature.exp().clamp(max=100)
    logits = scale * image_emb @ text_emb.t()
    labels = torch.arange(image_emb.size(0), device=image_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
