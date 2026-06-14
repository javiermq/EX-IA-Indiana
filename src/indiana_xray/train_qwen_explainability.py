from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_MODEL_ID, TARGET_CONCEPTS
from .dataset import image_transforms
from .models import DenseNetClassifier, VisualProjector, clip_contrastive_loss
from .utils import dump_json, ensure_dir, pick_device, set_seed


class SyntheticIndianaDataset(Dataset):
    def __init__(self, tsv_path: Path, indices: list[int], train: bool, image_size: int = 224) -> None:
        self.df = pd.read_csv(tsv_path, sep="\t").fillna("")
        self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = image_transforms(train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = str(row.get("clip_text", "") or row.get("synthetic_anomaly_text", "")).strip()
        next_text = str(row.get("next_token_text", "") or row.get("synthetic_anomaly_text", "")).strip()
        if not text:
            text = "Chest xray: No acute cardiopulmonary abnormality."
        if not next_text:
            next_text = "No acute cardiopulmonary abnormality."
        return {
            "image": self.transform(image),
            "image_id": str(row.get("image_id", idx)),
            "clip_text": text,
            "next_token_text": next_text,
        }


class VisualPrefixProjector(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        qwen_dim: int,
        gradcam_dim: int,
        prefix_len: int = 8,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.net = nn.Sequential(
            nn.Linear(visual_dim + gradcam_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prefix_len * qwen_dim),
        )

    def forward(self, visual: torch.Tensor, gradcam: torch.Tensor) -> torch.Tensor:
        prefix = self.net(torch.cat([visual, gradcam], dim=-1))
        return prefix.view(visual.size(0), self.prefix_len, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen explainability alignment with CLIP + next-token losses.")
    parser.add_argument("--synthetic-tsv", type=Path, required=True)
    parser.add_argument("--densenet-checkpoint", type=Path, required=True)
    parser.add_argument("--gradcam-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--clip-weight", type=float, default=1.0)
    parser.add_argument("--next-token-weight", type=float, default=0.25)
    parser.add_argument("--eval-examples", type=int, default=3)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-random-examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def clean_prediction(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()
    if not text:
        return "<empty>"
    if "." in text:
        text = text.split(".")[0].strip() + "."
    return text


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def load_densenet(checkpoint: Path, device: torch.device) -> DenseNetClassifier:
    model = DenseNetClassifier(num_classes=len(TARGET_CONCEPTS), pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model


def load_gradcam_embeddings(gradcam_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    data = np.load(gradcam_dir / "gradcam_embeddings.npz", allow_pickle=True)
    image_ids = data["image_ids"].astype(str)
    emb = np.concatenate([data["gradcam"], data["refined_attention"]], axis=1).astype("float32")
    return {image_id: emb[i] for i, image_id in enumerate(image_ids)}, emb.shape[1]


def batch_gradcam(batch: dict[str, object], mapping: dict[str, np.ndarray], dim: int, device: torch.device) -> torch.Tensor:
    vectors = [mapping.get(str(image_id), np.zeros(dim, dtype="float32")) for image_id in batch["image_id"]]
    return torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)


@torch.no_grad()
def qwen_text_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    outputs = model.model(**encoded, use_cache=False)
    hidden = outputs.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.float(), dim=-1)


def next_token_loss(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: torch.Tensor,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    token_emb = qwen.get_input_embeddings()(encoded["input_ids"])
    prefix = prefix.to(dtype=token_emb.dtype)
    inputs_embeds = torch.cat([prefix, token_emb], dim=1)
    prefix_mask = torch.ones(prefix.shape[:2], dtype=encoded["attention_mask"].dtype, device=device)
    attention_mask = torch.cat([prefix_mask, encoded["attention_mask"]], dim=1)
    prefix_labels = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=device)
    labels = torch.cat([prefix_labels, encoded["input_ids"]], dim=1)
    labels[:, prefix.shape[1]] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)
    outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return outputs.loss


def run_epoch(
    train: bool,
    loader: DataLoader,
    densenet: DenseNetClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    visual_projector: VisualProjector,
    prefix_projector: VisualPrefixProjector,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    log_temperature: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    visual_projector.train(train)
    prefix_projector.train(train)
    losses: list[float] = []
    clip_losses: list[float] = []
    nt_losses: list[float] = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        clip_texts = list(batch["clip_text"])
        next_texts = list(batch["next_token_text"])
        with torch.no_grad():
            _, visual, _ = densenet(images)
            gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
            text_emb = qwen_text_embeddings(qwen, tokenizer, clip_texts, device)

        with torch.set_grad_enabled(train):
            image_emb = visual_projector(visual.float(), gradcam)
            clip_loss = clip_contrastive_loss(image_emb, text_emb, log_temperature)
            prefix = prefix_projector(visual.float(), gradcam)
            nt_loss = next_token_loss(qwen, tokenizer, prefix, next_texts, device)
            loss = args.clip_weight * clip_loss + args.next_token_weight * nt_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        clip_losses.append(float(clip_loss.detach().cpu()))
        nt_losses.append(float(nt_loss.detach().cpu()))

    return {
        "loss": float(np.mean(losses)),
        "clip_loss": float(np.mean(clip_losses)),
        "next_token_loss": float(np.mean(nt_losses)),
    }


@torch.no_grad()
def retrieval_metrics(
    loader: DataLoader,
    densenet: DenseNetClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    visual_projector: VisualProjector,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
) -> dict[str, float]:
    visual_projector.eval()
    image_embs = []
    text_embs = []
    for batch in loader:
        images = batch["image"].to(device)
        texts = list(batch["clip_text"])
        _, visual, _ = densenet(images)
        gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
        image_embs.append(visual_projector(visual.float(), gradcam).cpu())
        text_embs.append(qwen_text_embeddings(qwen, tokenizer, texts, device).cpu())
    image_emb = torch.cat(image_embs)
    text_emb = torch.cat(text_embs)
    sims = image_emb @ text_emb.t()
    ranks = []
    for i in range(sims.size(0)):
        order = torch.argsort(sims[i], descending=True)
        ranks.append(int((order == i).nonzero(as_tuple=False)[0].item()) + 1)
    ranks_np = np.array(ranks)
    return {
        "retrieval_r1": float(np.mean(ranks_np <= 1)),
        "retrieval_r5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np)),
        "positive_cosine": float(torch.diag(sims).mean().item()),
    }


def make_eval_example_loader(
    dataset: SyntheticIndianaDataset,
    max_examples: int,
    batch_size: int,
    seed: int,
) -> DataLoader | None:
    if max_examples <= 0:
        return None
    df = dataset.df.reset_index(drop=True)
    normal_mask = df["next_token_text"].astype(str).eq("No acute cardiopulmonary abnormality.")
    normal_idx = np.flatnonzero(normal_mask.to_numpy())
    abnormal_idx = np.flatnonzero(~normal_mask.to_numpy())
    rng = np.random.default_rng(seed)
    n_abnormal = min(len(abnormal_idx), max_examples // 2 + max_examples % 2)
    n_normal = min(len(normal_idx), max_examples - n_abnormal)
    chosen: list[int] = []
    if n_abnormal:
        chosen.extend(rng.choice(abnormal_idx, size=n_abnormal, replace=False).tolist())
    if n_normal:
        chosen.extend(rng.choice(normal_idx, size=n_normal, replace=False).tolist())
    if len(chosen) < max_examples:
        pool = np.setdiff1d(np.arange(len(df)), np.array(chosen, dtype=int), assume_unique=False)
        if len(pool):
            chosen.extend(rng.choice(pool, size=min(len(pool), max_examples - len(chosen)), replace=False).tolist())
    rng.shuffle(chosen)
    subset = torch.utils.data.Subset(dataset, chosen)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def print_eval_examples(
    loader: DataLoader | None,
    densenet: DenseNetClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_projector: VisualPrefixProjector,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
    max_examples: int,
    max_new_tokens: int,
) -> None:
    if max_examples <= 0 or loader is None:
        return
    prefix_projector.eval()
    shown = 0
    print("eval_examples:")
    for batch in loader:
        images = batch["image"].to(device)
        _, visual, _ = densenet(images)
        gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
        prefix = prefix_projector(visual.float(), gradcam).to(dtype=qwen.get_input_embeddings().weight.dtype)
        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=device)
        generated = qwen.generate(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for image_id, gt, pred in zip(batch["image_id"], batch["next_token_text"], decoded):
            print(f"  image_id: {image_id}")
            print(f"    gt:   {gt}")
            print(f"    pred: {clean_prediction(pred)}")
            shown += 1
            if shown >= max_examples:
                return


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dtype = dtype_from_name(args.dtype, device)

    df = pd.read_csv(args.synthetic_tsv, sep="\t")
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, shuffle=True)
    train_ds = SyntheticIndianaDataset(args.synthetic_tsv, train_idx.tolist(), train=True)
    test_ds = SyntheticIndianaDataset(args.synthetic_tsv, test_idx.tolist(), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    eval_example_loader = make_eval_example_loader(
        test_ds,
        max_examples=args.eval_examples,
        batch_size=min(args.batch_size, max(1, args.eval_examples)),
        seed=args.seed,
    )

    densenet = load_densenet(args.densenet_checkpoint, device)
    gradcam_mapping, gradcam_dim = load_gradcam_embeddings(args.gradcam_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    qwen = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    qwen.eval()
    qwen.requires_grad_(False)
    qwen.config.use_cache = False

    visual_projector = VisualProjector(
        visual_dim=densenet.encoder.out_dim,
        text_dim=int(qwen.config.hidden_size),
        gradcam_dim=gradcam_dim,
    ).to(device)
    prefix_projector = VisualPrefixProjector(
        visual_dim=densenet.encoder.out_dim,
        qwen_dim=int(qwen.config.hidden_size),
        gradcam_dim=gradcam_dim,
        prefix_len=args.prefix_len,
    ).to(device)
    log_temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        list(visual_projector.parameters()) + list(prefix_projector.parameters()) + [log_temperature],
        lr=args.lr,
        weight_decay=1e-4,
    )

    history = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            True,
            train_loader,
            densenet,
            qwen,
            tokenizer,
            visual_projector,
            prefix_projector,
            gradcam_mapping,
            gradcam_dim,
            log_temperature,
            optimizer,
            args,
            device,
        )
        test_losses = run_epoch(
            False,
            test_loader,
            densenet,
            qwen,
            tokenizer,
            visual_projector,
            prefix_projector,
            gradcam_mapping,
            gradcam_dim,
            log_temperature,
            None,
            args,
            device,
        )
        retrieval = retrieval_metrics(
            test_loader,
            densenet,
            qwen,
            tokenizer,
            visual_projector,
            gradcam_mapping,
            gradcam_dim,
            device,
        )
        record = {"epoch": epoch, "train": train_metrics, "test": test_losses, "retrieval": retrieval}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"test_loss={test_losses['loss']:.4f} r@1={retrieval['retrieval_r1']:.4f} "
            f"r@5={retrieval['retrieval_r5']:.4f} nt={test_losses['next_token_loss']:.4f}"
        )
        print_eval_examples(
            eval_example_loader if args.eval_random_examples else test_loader,
            densenet,
            qwen,
            tokenizer,
            prefix_projector,
            gradcam_mapping,
            gradcam_dim,
            device,
            args.eval_examples,
            args.eval_max_new_tokens,
        )
        score = retrieval["retrieval_r5"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "visual_projector_state": visual_projector.state_dict(),
                    "prefix_projector_state": prefix_projector.state_dict(),
                    "log_temperature": float(log_temperature.detach().cpu()),
                    "epoch": epoch,
                    "metrics": record,
                    "model_id": args.model_id,
                    "prefix_len": args.prefix_len,
                    "gradcam_dim": gradcam_dim,
                },
                out_dir / "best.pt",
            )

    dump_json({"history": history, "best_retrieval_r5": best_score}, out_dir / "metrics.json")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
