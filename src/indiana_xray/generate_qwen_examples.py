from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TARGET_CONCEPTS
from .models import DenseNetClassifier
from .train_qwen_explainability import (
    SyntheticIndianaDataset,
    VisualPrefixProjector,
    batch_gradcam,
    dtype_from_name,
    load_gradcam_embeddings,
)
from .utils import pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen test examples from visual prefixes.")
    parser.add_argument("--synthetic-tsv", type=Path, required=True)
    parser.add_argument("--densenet-checkpoint", type=Path, required=True)
    parser.add_argument("--gradcam-dir", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    return parser.parse_args()


def load_densenet(checkpoint: Path, device: torch.device) -> DenseNetClassifier:
    model = DenseNetClassifier(num_classes=len(TARGET_CONCEPTS), pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.requires_grad_(False)
    return model


def clean_prediction(text: str) -> str:
    text = " ".join(text.replace("\n", " ").split())
    if not text:
        return "<empty>"
    for stop in [".", "\n"]:
        if stop in text:
            text = text.split(stop)[0].strip() + "."
            break
    return text


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    dtype = dtype_from_name(args.dtype, device)

    df = pd.read_csv(args.synthetic_tsv, sep="\t")
    indices = np.arange(len(df))
    _, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, shuffle=True)
    test_idx = test_idx[: args.num_examples]
    dataset = SyntheticIndianaDataset(args.synthetic_tsv, test_idx.tolist(), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    densenet = load_densenet(args.densenet_checkpoint, device)
    gradcam_mapping, gradcam_dim = load_gradcam_embeddings(args.gradcam_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    qwen = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    qwen.eval()
    qwen.requires_grad_(False)

    qwen_ckpt = torch.load(args.qwen_checkpoint, map_location=device)
    prefix_projector = VisualPrefixProjector(
        visual_dim=densenet.encoder.out_dim,
        qwen_dim=int(qwen.config.hidden_size),
        gradcam_dim=int(qwen_ckpt["gradcam_dim"]),
        prefix_len=int(qwen_ckpt["prefix_len"]),
    ).to(device)
    prefix_projector.load_state_dict(qwen_ckpt["prefix_projector_state"])
    prefix_projector.eval()

    for batch in loader:
        images = batch["image"].to(device)
        _, visual, _ = densenet(images)
        gradcam = batch_gradcam(batch, gradcam_mapping, gradcam_dim, device)
        prefix = prefix_projector(visual.float(), gradcam).to(dtype=qwen.get_input_embeddings().weight.dtype)
        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=device)
        generated = qwen.generate(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for image_id, gt, pred in zip(batch["image_id"], batch["next_token_text"], decoded):
            print("=" * 80)
            print(f"image_id: {image_id}")
            print(f"ground_truth: {gt}")
            print(f"predicted:    {clean_prediction(pred)}")


if __name__ == "__main__":
    main()
