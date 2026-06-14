from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import ensure_dir, pick_device


PROMPT_VERSION = "synthetic_anomaly_v5_keyword_guided"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NORMAL_SENTENCE = "No acute cardiopulmonary abnormality."
GENERIC_CONCEPTS = {"normal", "chest_xray", "lungs", "heart", "pleura", "mediastinum", "thorax"}
CONCEPT_TEXT = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "pleural_effusion": "pleural effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "emphysema": "emphysema",
    "nodule": "pulmonary nodule",
    "opacity": "pulmonary opacity",
    "infiltrate": "pulmonary infiltrate",
}
FORBIDDEN_PATTERNS = [
    r"\brecommend(?:ed|s|ation)?\b.*",
    r"\bfollow[- ]?up\b.*",
    r"\bcorrelate clinically\b.*",
    r"\bclinical correlation\b.*",
    r"\bif clinically indicated\b.*",
    r"\bshould be considered\b.*",
    r"\b(?:tube|catheter|line|clip|clips|hardware|port|pacemaker|sternotomy wire|surgical suture)\b.*",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "the",
    "there",
    "to",
    "was",
    "were",
    "with",
    "without",
}
RADIOLOGY_KEYWORDS = {
    "acute",
    "airspace",
    "apical",
    "atelectasis",
    "bibasilar",
    "bilateral",
    "blunting",
    "calcified",
    "cardiac",
    "cardiomegaly",
    "chronic",
    "clear",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "enlarged",
    "granuloma",
    "granulomatous",
    "heart",
    "hilar",
    "hyperinflation",
    "hyperinflated",
    "infiltrate",
    "interstitial",
    "left",
    "lobe",
    "lower",
    "lung",
    "lungs",
    "mild",
    "moderate",
    "nodule",
    "nodular",
    "opacity",
    "opacities",
    "perihilar",
    "pleural",
    "pneumonia",
    "pneumothorax",
    "pulmonary",
    "right",
    "scarring",
    "small",
    "subsegmental",
    "upper",
}
NEGATION_TERMS = {"no", "without", "negative", "absent", "absence", "free"}
NEGATION_WINDOW = 7
CLINICAL_PHRASES = [
    "airspace opacity",
    "airspace consolidation",
    "apical pneumothorax",
    "bibasilar opacities",
    "calcified granuloma",
    "cardiac enlargement",
    "chronic scarring",
    "enlarged heart",
    "granulomatous changes",
    "heart size enlarged",
    "interstitial edema",
    "interstitial infiltrate",
    "left lower lobe",
    "left upper lobe",
    "nodular opacity",
    "pleural effusion",
    "pulmonary edema",
    "pulmonary infiltrate",
    "pulmonary nodule",
    "right lower lobe",
    "right upper lobe",
    "subsegmental atelectasis",
    "vascular congestion",
]
CONCEPT_PHRASES = {
    "atelectasis": ["atelectasis", "atelectatic", "subsegmental atelectasis"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart size enlarged", "cardiac enlargement"],
    "consolidation": ["consolidation", "airspace consolidation"],
    "edema": ["edema", "pulmonary edema", "vascular congestion", "interstitial edema"],
    "pleural_effusion": ["pleural effusion", "effusion", "blunting"],
    "pneumonia": ["pneumonia"],
    "pneumothorax": ["pneumothorax", "apical pneumothorax"],
    "emphysema": ["emphysema", "hyperinflation", "hyperinflated", "hyperexpanded"],
    "nodule": ["nodule", "nodular opacity", "pulmonary nodule"],
    "opacity": ["opacity", "opacities", "airspace opacity", "opacification"],
    "infiltrate": ["infiltrate", "infiltrates", "interstitial infiltrate"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short synthetic abnormality sentences from IU-Xray reports with Qwen."
    )
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--keyword-count", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Optional small debug subset.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing rows in out-tsv by image_id.")
    return parser.parse_args()


def quantization_config(args: argparse.Namespace, dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" else torch.float32


def clean_report_text(*parts: object) -> str:
    text = " ".join(str(part or "") for part in parts)
    text = re.sub(r"\bX{2,}\b", " ", text)
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def abnormal_concepts(row: dict[str, object]) -> list[str]:
    concepts = [c for c in str(row.get("concepts", "")).split("|") if c]
    concepts = [c for c in concepts if c not in GENERIC_CONCEPTS]
    ordered = []
    for concept in concepts:
        if concept not in ordered:
            ordered.append(concept)
    return ordered


def is_negated(text: str, start_idx: int) -> bool:
    prefix = tokenize_keywords(text[:start_idx])
    if any(token in NEGATION_TERMS for token in prefix[-NEGATION_WINDOW:]):
        return True
    clause_start = max(text.rfind(".", 0, start_idx), text.rfind(";", 0, start_idx), text.rfind(":", 0, start_idx))
    clause = text[clause_start + 1 : start_idx].lower()
    return bool(re.search(r"\b(?:without|no)\s+(?:evidence\s+of|signs?\s+of|definite\s+)?", clause))


def concept_is_only_negated(report_text: str, concept: str) -> bool:
    phrases = CONCEPT_PHRASES.get(concept, [CONCEPT_TEXT.get(concept, concept.replace("_", " "))])
    matches: list[bool] = []
    lowered = report_text.lower()
    for phrase in phrases:
        phrase_norm = phrase.lower()
        for match in re.finditer(rf"\b{re.escape(phrase_norm)}\b", lowered):
            matches.append(is_negated(lowered, match.start()))
    return bool(matches) and all(matches)


def filter_negated_concepts(report_text: str, concepts: list[str]) -> list[str]:
    return [concept for concept in concepts if not concept_is_only_negated(report_text, concept)]


def concepts_to_text(concepts: list[str], max_items: int = 4) -> str:
    terms = [CONCEPT_TEXT.get(c, c.replace("_", " ")) for c in concepts[:max_items]]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0].capitalize() + "."
    return ", ".join(terms[:-1]).capitalize() + f", and {terms[-1]}."


def tokenize_keywords(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z/-]*", text.lower())


def extract_prompt_keywords(report_text: str, concepts: list[str], max_keywords: int) -> list[str]:
    lowered = report_text.lower()
    scores: dict[str, float] = {}

    for concept in concepts:
        phrase = CONCEPT_TEXT.get(concept, concept.replace("_", " "))
        for token in tokenize_keywords(phrase):
            if token not in STOPWORDS:
                scores[token] = max(scores.get(token, 0.0), 5.0)

    for phrase in CLINICAL_PHRASES:
        match_positions = [match.start() for match in re.finditer(rf"\b{re.escape(phrase)}\b", lowered)]
        if match_positions and any(not is_negated(lowered, pos) for pos in match_positions):
            scores[phrase] = max(scores.get(phrase, 0.0), 4.0)
            for token in tokenize_keywords(phrase):
                if token not in STOPWORDS:
                    scores[token] = max(scores.get(token, 0.0), 3.0)

    for match in re.finditer(r"\b[a-z][a-z/-]*\b", lowered):
        token = match.group(0)
        if token in STOPWORDS or len(token) < 3:
            continue
        if is_negated(lowered, match.start()):
            continue
        if token in RADIOLOGY_KEYWORDS:
            scores[token] = scores.get(token, 0.0) + 2.0
        elif token in {concept.replace("_", "-") for concept in concepts}:
            scores[token] = scores.get(token, 0.0) + 1.5

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    keywords: list[str] = []
    covered_tokens: set[str] = set()
    for keyword, _ in ranked:
        parts = tokenize_keywords(keyword)
        if len(parts) > 1:
            keywords.append(keyword)
            covered_tokens.update(parts)
        elif keyword not in covered_tokens:
            keywords.append(keyword)
        if len(keywords) >= max_keywords:
            break
    return keywords


def build_prompt(report_text: str, candidate_text: str, prompt_keywords: list[str]) -> str:
    hints = candidate_text or "none"
    keywords = ", ".join(prompt_keywords) if prompt_keywords else "none"
    return (
        "Extract only radiographic chest xray findings from the report.\n"
        "Write one complete noun phrase in English with a maximum of 12 words.\n"
        f"Candidate abnormality hints from weak labels: {hints}.\n"
        f"Precomputed report keywords, without stopwords: {keywords}.\n"
        f"If the report and hints contain no abnormality, write exactly: {NORMAL_SENTENCE}\n"
        "Prefer words from the precomputed keywords when they are clinically relevant.\n"
        "Mention only visible abnormalities and locations.\n"
        "If weak labels suggest an abnormality and the report supports it, include it.\n"
        "Do not introduce diseases, measurements, or locations absent from the report keywords or report text.\n"
        "Do not mention tubes, catheters, lines, clips, ports, pacemakers, or hardware unless they are the primary finding.\n"
        "Do not mention recommendations, follow-up, clinical correlation, uncertainty management, patient data, dates, comparisons, placeholders, or report metadata.\n"
        "Do not invent findings.\n\n"
        f"Report:\n{report_text}\n\n"
        "Short abnormality sentence:"
    )


def normalize_generation(text: str) -> str:
    text = re.sub(r"\bX{2,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -:\t\n\"'")
    if not text:
        return NORMAL_SENTENCE
    lines = [line.strip(" -:\t\"'") for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text
    text = re.split(r"(?i)\b(report|short abnormality sentence)\s*:", text)[0].strip()
    for pattern in FORBIDDEN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,;:-")
    if not text:
        return NORMAL_SENTENCE
    words = text.split()
    if len(words) > 12:
        text = " ".join(words[:12]).rstrip(" ,;:-")
    if text[-1] not in ".!?":
        text += "."
    return text


def apply_abnormal_fallback(sentence: str, concepts: list[str]) -> str:
    if sentence == NORMAL_SENTENCE and concepts:
        fallback = concepts_to_text(concepts)
        return fallback or sentence
    return sentence


def load_existing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    existing = pd.read_csv(path, sep="\t").fillna("")
    if "image_id" not in existing.columns or "synthetic_anomaly_text" not in existing.columns:
        return {}
    return dict(zip(existing["image_id"].astype(str), existing["synthetic_anomaly_text"].astype(str)))


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages
        ]
    else:
        prompt_texts = prompts

    encoded = tokenizer(prompt_texts, padding=True, truncation=True, max_length=768, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[:, encoded["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [normalize_generation(text) for text in decoded]


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = dtype_from_name(args.dtype, device)

    df = pd.read_csv(args.input_tsv, sep="\t").fillna("")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    existing = load_existing(args.out_tsv) if args.resume else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    qconfig = quantization_config(args, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        quantization_config=qconfig,
        device_map="auto" if qconfig is not None else None,
    )
    if qconfig is None:
        model = model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    pending: list[tuple[int, dict[str, object], str, list[str]]] = []

    for idx, row in enumerate(df.to_dict("records")):
        image_id = str(row.get("image_id", idx))
        report_text_clean = clean_report_text(
            row.get("findings", ""),
            row.get("impression", ""),
            row.get("indication", ""),
        )
        out = dict(row)
        concepts = filter_negated_concepts(report_text_clean, abnormal_concepts(row))
        candidate_text = ", ".join(CONCEPT_TEXT.get(c, c.replace("_", " ")) for c in concepts)
        prompt_keywords = extract_prompt_keywords(report_text_clean, concepts, args.keyword_count)
        out["report_text_clean"] = report_text_clean
        out["prompt_keywords"] = "|".join(prompt_keywords)
        if image_id in existing and existing[image_id]:
            sentence = apply_abnormal_fallback(normalize_generation(existing[image_id]), concepts)
            out["synthetic_anomaly_text"] = sentence
            out["clip_text"] = f"Chest xray: {sentence}"
            out["next_token_text"] = sentence
            out["synthetic_model"] = args.model_id
            out["synthetic_prompt_version"] = PROMPT_VERSION
            rows.append(out)
            continue
        pending.append((idx, out, build_prompt(report_text_clean, candidate_text, prompt_keywords), concepts))

    for start in tqdm(range(0, len(pending), args.batch_size), desc="Generating synthetic text"):
        batch = pending[start : start + args.batch_size]
        prompts = [item[2] for item in batch]
        sentences = generate_batch(model, tokenizer, prompts, device, args.max_new_tokens)
        for (_, out, _, concepts), sentence in zip(batch, sentences):
            sentence = apply_abnormal_fallback(sentence, concepts)
            out["synthetic_anomaly_text"] = sentence
            out["clip_text"] = f"Chest xray: {sentence}"
            out["next_token_text"] = sentence
            out["synthetic_model"] = args.model_id
            out["synthetic_prompt_version"] = PROMPT_VERSION
            rows.append(out)

    out_df = pd.DataFrame(rows)
    ensure_dir(args.out_tsv.parent)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_tsv}")


if __name__ == "__main__":
    main()
