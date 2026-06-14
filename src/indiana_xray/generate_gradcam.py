from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TARGET_CONCEPTS
from .dataset import IndianaConceptDataset
from .models import AttentionDecoder, DenseNetClassifier, GradCAM
from .utils import dump_json, ensure_dir, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM maps and train a light attention decoder.")
    parser.add_argument("--concepts-tsv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_classifier(checkpoint: Path, device: torch.device) -> DenseNetClassifier:
    model = DenseNetClassifier(num_classes=len(TARGET_CONCEPTS), pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def train_decoder(
    model: DenseNetClassifier,
    decoder: AttentionDecoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    cammer = GradCAM(model)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    decoder.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in tqdm(loader, desc=f"decoder epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            cams = cammer(images).detach()
            with torch.no_grad():
                _, _, fmap = model(images)
            refined = decoder(fmap.detach(), out_size=images.shape[-2:])
            loss = F.binary_cross_entropy(refined, cams)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        print(f"decoder_epoch={epoch} bce={history[-1]:.5f}")
    cammer.remove()
    return {"decoder_bce_last": history[-1], "decoder_bce_history": history}


def collect_maps_and_metrics(
    model: DenseNetClassifier,
    decoder: AttentionDecoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    cammer = GradCAM(model)
    decoder.eval()
    image_ids: list[str] = []
    cam_embeddings: list[np.ndarray] = []
    refined_embeddings: list[np.ndarray] = []
    deletion_drops: list[float] = []
    refined_mse: list[float] = []

    for batch in tqdm(loader, desc="Grad-CAM"):
        images = batch["image"].to(device)
        logits, _, fmap = model(images)
        class_idx = logits.sigmoid().argmax(dim=1)
        cams = cammer(images, class_idx=class_idx).detach()
        with torch.no_grad():
            refined = decoder(fmap.detach(), out_size=images.shape[-2:])
            top_mask = (cams >= cams.flatten(1).quantile(0.8, dim=1).view(-1, 1, 1, 1)).float()
            masked_images = images * (1.0 - top_mask)
            masked_logits, _, _ = model(masked_images)
            base_prob = logits.sigmoid().gather(1, class_idx.view(-1, 1))
            masked_prob = masked_logits.sigmoid().gather(1, class_idx.view(-1, 1))
            deletion_drops.extend((base_prob - masked_prob).squeeze(1).detach().cpu().tolist())
            refined_mse.extend(F.mse_loss(refined, cams, reduction="none").mean(dim=(1, 2, 3)).detach().cpu().tolist())
            cam_vec = F.adaptive_avg_pool2d(cams, (7, 7)).flatten(1)
            refined_vec = F.adaptive_avg_pool2d(refined, (7, 7)).flatten(1)
        image_ids.extend(list(batch["image_id"]))
        cam_embeddings.append(cam_vec.detach().cpu().numpy())
        refined_embeddings.append(refined_vec.detach().cpu().numpy())

    cammer.remove()
    arrays = {
        "image_ids": np.array(image_ids),
        "gradcam": np.concatenate(cam_embeddings),
        "refined_attention": np.concatenate(refined_embeddings),
    }
    metrics = {
        "mean_deletion_drop_top20": float(np.mean(deletion_drops)),
        "median_deletion_drop_top20": float(np.median(deletion_drops)),
        "mean_refined_attention_mse": float(np.mean(refined_mse)),
    }
    return arrays, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dataset = IndianaConceptDataset(args.concepts_tsv, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = load_classifier(args.checkpoint, device)
    decoder = AttentionDecoder(in_channels=model.encoder.out_dim).to(device)

    train_metrics = train_decoder(model, decoder, loader, device, args.decoder_epochs, args.lr)
    arrays, metrics = collect_maps_and_metrics(model, decoder, loader, device)
    np.savez_compressed(out_dir / "gradcam_embeddings.npz", **arrays)
    torch.save({"decoder_state": decoder.state_dict(), "metrics": metrics}, out_dir / "attention_decoder.pt")
    dump_json({**train_metrics, **metrics}, out_dir / "metrics.json")
    print(f"Saved Grad-CAM embeddings to {out_dir / 'gradcam_embeddings.npz'}")


if __name__ == "__main__":
    main()
