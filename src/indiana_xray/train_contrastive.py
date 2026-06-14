from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULT_MODEL_ID, TARGET_CONCEPTS
from .dataset import IndianaConceptDataset
from .models import (
    DenseNetClassifier,
    QwenTextEncoder,
    QwenTextEncoderConfig,
    VisualProjector,
    clip_contrastive_loss,
)
from .utils import dump_json, ensure_dir, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual-concept contrastive alignment with Qwen text embeddings.")
    parser.add_argument("--concepts-tsv", type=Path, required=True)
    parser.add_argument("--densenet-checkpoint", type=Path, required=True)
    parser.add_argument("--gradcam-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def torch_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_densenet(checkpoint: Path, device: torch.device) -> DenseNetClassifier:
    model = DenseNetClassifier(num_classes=len(TARGET_CONCEPTS), pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model


def load_gradcam_embeddings(gradcam_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    path = gradcam_dir / "gradcam_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    image_ids = data["image_ids"].astype(str)
    emb = np.concatenate([data["gradcam"], data["refined_attention"]], axis=1).astype("float32")
    mapping = {image_id: emb[i] for i, image_id in enumerate(image_ids)}
    return mapping, emb.shape[1]


def batch_gradcam(batch: dict[str, object], mapping: dict[str, np.ndarray], dim: int, device: torch.device) -> torch.Tensor:
    vectors = []
    for image_id in batch["image_id"]:
        vectors.append(mapping.get(str(image_id), np.zeros(dim, dtype="float32")))
    return torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)


def train_one_setting(
    name: str,
    densenet: DenseNetClassifier,
    text_encoder: QwenTextEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    use_gradcam: bool,
    out_dir: Path,
) -> dict[str, object]:
    projector = VisualProjector(
        visual_dim=densenet.encoder.out_dim,
        text_dim=text_encoder.out_dim,
        gradcam_dim=gradcam_dim if use_gradcam else 0,
    ).to(device)
    log_temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(list(projector.parameters()) + [log_temperature], lr=lr, weight_decay=1e-4)
    history = []

    for epoch in range(1, epochs + 1):
        projector.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{name} epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            texts = list(batch["text"])
            with torch.no_grad():
                _, visual, _ = densenet(images)
                text_emb = text_encoder(texts).to(device)
                grad = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device) if use_gradcam else None
            image_emb = projector(visual, grad)
            loss = clip_contrastive_loss(image_emb, text_emb, log_temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(projector, densenet, text_encoder, test_loader, gradcam_mapping, gradcam_dim, device, use_gradcam)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        print(f"{name} epoch={epoch} loss={record['loss']:.4f} r@1={metrics['text_retrieval_r1']:.4f}")

    ckpt_path = out_dir / f"{name}_projector.pt"
    torch.save(
        {
            "projector_state": projector.state_dict(),
            "log_temperature": float(log_temperature.detach().cpu()),
            "use_gradcam": use_gradcam,
            "gradcam_dim": gradcam_dim if use_gradcam else 0,
            "model_id": text_encoder.tokenizer.name_or_path,
        },
        ckpt_path,
    )
    return {"history": history, "final": history[-1], "checkpoint": str(ckpt_path)}


@torch.no_grad()
def evaluate(
    projector: VisualProjector,
    densenet: DenseNetClassifier,
    text_encoder: QwenTextEncoder,
    loader: DataLoader,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
    use_gradcam: bool,
) -> dict[str, float]:
    projector.eval()
    image_embs = []
    text_embs = []
    for batch in loader:
        images = batch["image"].to(device)
        texts = list(batch["text"])
        _, visual, _ = densenet(images)
        grad = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device) if use_gradcam else None
        image_embs.append(projector(visual, grad).cpu())
        text_embs.append(text_encoder(texts).cpu())

    image_emb = torch.cat(image_embs)
    text_emb = torch.cat(text_embs)
    sims = image_emb @ text_emb.t()
    ranks = []
    for i in range(sims.size(0)):
        order = torch.argsort(sims[i], descending=True)
        rank = int((order == i).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    ranks_np = np.array(ranks)
    return {
        "text_retrieval_r1": float(np.mean(ranks_np <= 1)),
        "text_retrieval_r5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np)),
        "mean_positive_cosine": float(torch.diag(sims).mean().item()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dtype = torch_dtype(args.dtype, device)

    base_ds = IndianaConceptDataset(args.concepts_tsv)
    indices = np.arange(len(base_ds))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, shuffle=True)
    train_ds = IndianaConceptDataset(args.concepts_tsv, indices=train_idx.tolist(), train=False)
    test_ds = IndianaConceptDataset(args.concepts_tsv, indices=test_idx.tolist(), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    densenet = load_densenet(args.densenet_checkpoint, device)
    gradcam_mapping, gradcam_dim = load_gradcam_embeddings(args.gradcam_dir)
    text_encoder = QwenTextEncoder(QwenTextEncoderConfig(model_id=args.model_id, device=device, dtype=dtype))

    visual_only = train_one_setting(
        "visual_only",
        densenet,
        text_encoder,
        train_loader,
        test_loader,
        gradcam_mapping,
        gradcam_dim,
        device,
        args.epochs,
        args.lr,
        use_gradcam=False,
        out_dir=out_dir,
    )
    visual_gradcam = train_one_setting(
        "visual_gradcam",
        densenet,
        text_encoder,
        train_loader,
        test_loader,
        gradcam_mapping,
        gradcam_dim,
        device,
        args.epochs,
        args.lr,
        use_gradcam=True,
        out_dir=out_dir,
    )
    metrics = {
        "visual_only": visual_only,
        "visual_gradcam": visual_gradcam,
        "gradcam_delta_r1": visual_gradcam["final"]["text_retrieval_r1"] - visual_only["final"]["text_retrieval_r1"],
        "gradcam_delta_positive_cosine": visual_gradcam["final"]["mean_positive_cosine"]
        - visual_only["final"]["mean_positive_cosine"],
    }
    dump_json(metrics, out_dir / "metrics.json")
    print(f"Saved contrastive metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
