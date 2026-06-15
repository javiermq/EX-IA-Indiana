from __future__ import annotations

import argparse
import re
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
from .models import DenseNetClassifier, clip_contrastive_loss
from .utils import dump_json, ensure_dir, pick_device, set_seed


DEFAULT_DECODER_PROMPT = "Chest xray finding in English:"


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


class RegionalVisualAdapter(nn.Module):
    def __init__(
        self,
        fmap_channels: int,
        qwen_dim: int,
        gradcam_dim: int,
        prefix_len: int = 8,
        hidden_dim: int = 1024,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.gradcam_dim = gradcam_dim
        self.region_proj = nn.Sequential(
            nn.Linear(fmap_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, qwen_dim),
        )
        self.query_tokens = nn.Parameter(torch.randn(prefix_len, qwen_dim) * 0.02)
        self.reducer = nn.MultiheadAttention(qwen_dim, num_heads=num_heads, batch_first=True)
        self.prefix_norm = nn.LayerNorm(qwen_dim)
        self.clip_norm = nn.LayerNorm(qwen_dim)

    def gradcam_maps(self, gradcam: torch.Tensor, fmap_size: tuple[int, int]) -> torch.Tensor:
        half = self.gradcam_dim // 2
        side = int(np.sqrt(half))
        if self.gradcam_dim % 2 != 0 or side * side != half:
            raise ValueError(f"Expected gradcam_dim to be 2*square, got {self.gradcam_dim}")
        maps = gradcam.view(gradcam.size(0), 2, side, side)
        if maps.shape[-2:] != fmap_size:
            maps = F.interpolate(maps, size=fmap_size, mode="bilinear", align_corners=False)
        return maps

    def region_tokens(self, fmap: torch.Tensor, gradcam: torch.Tensor) -> torch.Tensor:
        maps = self.gradcam_maps(gradcam, fmap.shape[-2:]).to(dtype=fmap.dtype)
        regions = torch.cat([fmap, maps], dim=1).flatten(2).transpose(1, 2)
        return self.region_proj(regions)

    def clip_embedding(self, fmap: torch.Tensor, gradcam: torch.Tensor) -> torch.Tensor:
        tokens = self.region_tokens(fmap, gradcam)
        return F.normalize(self.clip_norm(tokens.mean(dim=1)).float(), dim=-1)

    def prefix_tokens(self, fmap: torch.Tensor, gradcam: torch.Tensor) -> torch.Tensor:
        tokens = self.region_tokens(fmap, gradcam)
        queries = self.query_tokens.unsqueeze(0).expand(tokens.size(0), -1, -1)
        prefix, _ = self.reducer(queries, tokens, tokens, need_weights=False)
        return self.prefix_norm(prefix)


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
    parser.add_argument("--decoder-prompt", default=DEFAULT_DECODER_PROMPT)
    parser.add_argument("--eval-examples", type=int, default=3)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-constrained-candidates", type=int, default=40)
    parser.add_argument("--no-eval-prior-correction", action="store_true")
    parser.add_argument("--eval-free-generation", action="store_true")
    parser.add_argument("--eval-random-examples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def clean_prediction(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split()).strip()
    text = re.sub(r"^[\s:;,.!?-]+", "", text)
    text = re.sub(r"^icalcified\b", "calcified", text, flags=re.IGNORECASE)
    text = re.sub(r"^icarcinoma\b", "carcinoma", text, flags=re.IGNORECASE)
    text = re.sub(r"^icortical\b", "cortical", text, flags=re.IGNORECASE)
    text = re.sub(r"^ilar\b", "hilar", text, flags=re.IGNORECASE)
    if not text:
        return "<empty>"
    if re.search(r"[^\x00-\x7F]", text):
        return "<non-english>"
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
    decoder_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    prompt = decoder_prompt.strip() + " "
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        full_texts = [prompt + text for text in texts]
        encoded = tokenizer(full_texts, padding=True, truncation=True, max_length=80, return_tensors="pt").to(device)
        prompt_encoded = tokenizer([prompt] * len(texts), padding=False, truncation=True, max_length=80)
    finally:
        tokenizer.padding_side = old_padding_side

    token_emb = qwen.get_input_embeddings()(encoded["input_ids"])
    prefix = prefix.to(dtype=token_emb.dtype)
    inputs_embeds = torch.cat([prefix, token_emb], dim=1)
    prefix_mask = torch.ones(prefix.shape[:2], dtype=encoded["attention_mask"].dtype, device=device)
    attention_mask = torch.cat([prefix_mask, encoded["attention_mask"]], dim=1)
    prefix_labels = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=device)
    token_labels = encoded["input_ids"].clone()
    for i, prompt_ids in enumerate(prompt_encoded["input_ids"]):
        token_labels[i, : min(len(prompt_ids), token_labels.size(1))] = -100
    token_labels = token_labels.masked_fill(encoded["attention_mask"] == 0, -100)
    labels = torch.cat([prefix_labels, token_labels], dim=1)
    labels = labels.masked_fill(attention_mask == 0, -100)
    outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
    return outputs.loss


def build_prompted_inputs(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: torch.Tensor,
    texts: list[str],
    decoder_prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = decoder_prompt.strip() + " "
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        full_texts = [prompt + text for text in texts]
        encoded = tokenizer(full_texts, padding=True, truncation=True, max_length=80, return_tensors="pt").to(device)
        prompt_encoded = tokenizer([prompt] * len(texts), padding=False, truncation=True, max_length=80)
    finally:
        tokenizer.padding_side = old_padding_side

    token_emb = qwen.get_input_embeddings()(encoded["input_ids"])
    prefix = prefix.to(dtype=token_emb.dtype)
    inputs_embeds = torch.cat([prefix, token_emb], dim=1)
    prefix_mask = torch.ones(prefix.shape[:2], dtype=encoded["attention_mask"].dtype, device=device)
    attention_mask = torch.cat([prefix_mask, encoded["attention_mask"]], dim=1)
    prefix_labels = torch.full(prefix.shape[:2], -100, dtype=torch.long, device=device)
    token_labels = encoded["input_ids"].clone()
    for i, prompt_ids in enumerate(prompt_encoded["input_ids"]):
        token_labels[i, : min(len(prompt_ids), token_labels.size(1))] = -100
    token_labels = token_labels.masked_fill(encoded["attention_mask"] == 0, -100)
    labels = torch.cat([prefix_labels, token_labels], dim=1)
    labels = labels.masked_fill(attention_mask == 0, -100)
    return inputs_embeds, attention_mask, labels


@torch.no_grad()
def next_token_nll_per_sample(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: torch.Tensor,
    texts: list[str],
    decoder_prompt: str,
    device: torch.device,
) -> torch.Tensor:
    inputs_embeds, attention_mask, labels = build_prompted_inputs(
        qwen, tokenizer, prefix, texts, decoder_prompt, device
    )
    logits = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    return flat_loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def run_epoch(
    train: bool,
    loader: DataLoader,
    densenet: DenseNetClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    visual_adapter: RegionalVisualAdapter,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    log_temperature: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    visual_adapter.train(train)
    losses: list[float] = []
    clip_losses: list[float] = []
    nt_losses: list[float] = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        clip_texts = list(batch["clip_text"])
        next_texts = list(batch["next_token_text"])
        with torch.no_grad():
            _, _, fmap = densenet(images)
            gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
            text_emb = qwen_text_embeddings(qwen, tokenizer, clip_texts, device)

        with torch.set_grad_enabled(train):
            image_emb = visual_adapter.clip_embedding(fmap.float(), gradcam)
            clip_loss = clip_contrastive_loss(image_emb, text_emb, log_temperature)
            prefix = visual_adapter.prefix_tokens(fmap.float(), gradcam)
            nt_loss = next_token_loss(qwen, tokenizer, prefix, next_texts, args.decoder_prompt, device)
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
    visual_adapter: RegionalVisualAdapter,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
) -> dict[str, float]:
    visual_adapter.eval()
    image_embs = []
    text_embs = []
    for batch in loader:
        images = batch["image"].to(device)
        texts = list(batch["clip_text"])
        _, _, fmap = densenet(images)
        gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
        image_embs.append(visual_adapter.clip_embedding(fmap.float(), gradcam).cpu())
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


def candidate_texts_from_df(df: pd.DataFrame, max_candidates: int) -> list[str]:
    if max_candidates <= 0 or "next_token_text" not in df.columns:
        return []
    texts = df["next_token_text"].astype(str).str.strip()
    texts = texts[texts.ne("")]
    texts = texts[~texts.str.contains(r"[^\x00-\x7F]", regex=True)]
    ranked = texts.value_counts().head(max_candidates).index.tolist()
    normal = "No acute cardiopulmonary abnormality."
    if normal in set(texts) and normal not in ranked:
        ranked = [normal] + ranked[: max_candidates - 1]
    return ranked


@torch.no_grad()
def constrained_predictions(
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: torch.Tensor,
    candidates: list[str],
    decoder_prompt: str,
    device: torch.device,
    chunk_size: int = 32,
    prior_correction: bool = True,
) -> list[str]:
    if not candidates:
        return []
    predictions: list[str] = []
    for i in range(prefix.size(0)):
        row_prefix = prefix[i : i + 1]
        losses = []
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            repeated_prefix = row_prefix.expand(len(chunk), -1, -1).contiguous()
            conditional = next_token_nll_per_sample(qwen, tokenizer, repeated_prefix, chunk, decoder_prompt, device)
            if prior_correction:
                null_prefix = torch.zeros_like(repeated_prefix)
                prior = next_token_nll_per_sample(qwen, tokenizer, null_prefix, chunk, decoder_prompt, device)
                conditional = conditional - prior
            losses.append(conditional)
        scores = torch.cat(losses)
        predictions.append(candidates[int(torch.argmin(scores).item())])
    return predictions


@torch.no_grad()
def print_eval_examples(
    loader: DataLoader | None,
    densenet: DenseNetClassifier,
    qwen: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    visual_adapter: RegionalVisualAdapter,
    gradcam_mapping: dict[str, np.ndarray],
    gradcam_dim: int,
    device: torch.device,
    max_examples: int,
    max_new_tokens: int,
    decoder_prompt: str,
    candidates: list[str],
    free_generation: bool,
    prior_correction: bool,
) -> None:
    if max_examples <= 0 or loader is None:
        return
    visual_adapter.eval()
    shown = 0
    print("eval_examples:")
    for batch in loader:
        images = batch["image"].to(device)
        _, _, fmap = densenet(images)
        gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
        prefix = visual_adapter.prefix_tokens(fmap.float(), gradcam).to(dtype=qwen.get_input_embeddings().weight.dtype)
        constrained = constrained_predictions(
            qwen,
            tokenizer,
            prefix,
            candidates,
            decoder_prompt,
            device,
            prior_correction=prior_correction,
        )
        decoded = [""] * len(batch["image_id"])
        if free_generation:
            prompt_texts = [decoder_prompt.strip() + " "] * prefix.size(0)
            encoded_prompt = tokenizer(prompt_texts, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
            prompt_emb = qwen.get_input_embeddings()(encoded_prompt["input_ids"])
            prompt_emb = prompt_emb.to(dtype=prefix.dtype)
            inputs_embeds = torch.cat([prefix, prompt_emb], dim=1)
            prefix_mask = torch.ones(prefix.shape[:2], dtype=encoded_prompt["attention_mask"].dtype, device=device)
            attention_mask = torch.cat([prefix_mask, encoded_prompt["attention_mask"]], dim=1)
            generated = qwen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, (image_id, gt) in enumerate(zip(batch["image_id"], batch["next_token_text"])):
            print(f"  image_id: {image_id}")
            print(f"    gt:   {gt}")
            if constrained:
                print(f"    pred: {constrained[i]}")
            if free_generation:
                print(f"    free: {clean_prediction(decoded[i])}")
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
    eval_candidates = candidate_texts_from_df(train_ds.df, args.eval_constrained_candidates)

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

    visual_adapter = RegionalVisualAdapter(
        fmap_channels=densenet.encoder.out_dim,
        qwen_dim=int(qwen.config.hidden_size),
        gradcam_dim=gradcam_dim,
        prefix_len=args.prefix_len,
    ).to(device)
    log_temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        list(visual_adapter.parameters()) + [log_temperature],
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
            visual_adapter,
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
            visual_adapter,
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
            visual_adapter,
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
            visual_adapter,
            gradcam_mapping,
            gradcam_dim,
            device,
            args.eval_examples,
            args.eval_max_new_tokens,
            args.decoder_prompt,
            eval_candidates,
            args.eval_free_generation,
            not args.no_eval_prior_correction,
        )
        score = retrieval["retrieval_r5"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "visual_adapter_state": visual_adapter.state_dict(),
                    "log_temperature": float(log_temperature.detach().cpu()),
                    "epoch": epoch,
                    "metrics": record,
                    "model_id": args.model_id,
                    "prefix_len": args.prefix_len,
                    "gradcam_dim": gradcam_dim,
                    "adapter": "regional_7x7_dense_gradcam_refined",
                },
                out_dir / "best.pt",
            )

    dump_json({"history": history, "best_retrieval_r5": best_score}, out_dir / "metrics.json")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
