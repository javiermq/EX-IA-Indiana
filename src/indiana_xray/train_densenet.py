from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TARGET_CONCEPTS
from .dataset import IndianaConceptDataset
from .models import DenseNetClassifier
from .utils import dump_json, ensure_dir, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet121 + residual MLP on IU-Xray weak labels.")
    parser.add_argument("--concepts-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | dict[str, float]]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, float | dict[str, float]] = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_auc: dict[str, float] = {}
    per_ap: dict[str, float] = {}
    for i, concept in enumerate(TARGET_CONCEPTS):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        per_auc[concept] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        per_ap[concept] = float(average_precision_score(y_true[:, i], y_prob[:, i]))
    metrics["macro_auc"] = float(np.mean(list(per_auc.values()))) if per_auc else float("nan")
    metrics["macro_ap"] = float(np.mean(list(per_ap.values()))) if per_ap else float("nan")
    metrics["per_class_auc"] = per_auc
    metrics["per_class_ap"] = per_ap
    return metrics


def run_epoch(
    model: DenseNetClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        with torch.set_grad_enabled(train):
            logits, _, _ = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(logits.sigmoid().detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(y_true), np.concatenate(y_prob)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)

    base_ds = IndianaConceptDataset(args.concepts_tsv)
    indices = np.arange(len(base_ds))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, shuffle=True)
    train_ds = IndianaConceptDataset(args.concepts_tsv, indices=train_idx.tolist(), train=True)
    test_ds = IndianaConceptDataset(args.concepts_tsv, indices=test_idx.tolist(), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DenseNetClassifier(num_classes=len(TARGET_CONCEPTS), pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_auc = -1.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_y, train_p = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_y, test_p = run_epoch(model, test_loader, criterion, device)
        train_metrics = multilabel_metrics(train_y, train_p)
        test_metrics = multilabel_metrics(test_y, test_p)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"test_macro_auc={test_metrics['macro_auc']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}"
        )
        score = float(test_metrics["macro_auc"])
        if np.isfinite(score) and score > best_auc:
            best_auc = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_concepts": TARGET_CONCEPTS,
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                    "epoch": epoch,
                    "metrics": test_metrics,
                },
                out_dir / "best.pt",
            )

    dump_json({"history": history, "best_macro_auc": best_auc}, out_dir / "metrics.json")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
