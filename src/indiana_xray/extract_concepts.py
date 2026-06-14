from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from .config import TARGET_CONCEPTS
from .utils import ensure_dir

NEGATION_WINDOW = 7
NEGATION_TERMS = {"no", "without", "negative", "absence", "absent", "free"}
GENERIC_CONCEPTS = ["chest_xray", "lungs", "heart", "pleura", "mediastinum", "thorax"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalized clinical concepts from IU-Xray reports.")
    parser.add_argument("--dataset-tsv", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, default=Path(__file__).with_name("clinical_dictionary.yml"))
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--min-concepts", type=int, default=5)
    parser.add_argument("--max-concepts", type=int, default=10)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s_/.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_/-]+", normalize_text(text))


def load_dictionary(path: Path) -> dict[str, dict[str, list[str] | str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return raw


def is_negated(text: str, start_idx: int) -> bool:
    prefix = tokenize(text[:start_idx])
    window = prefix[-NEGATION_WINDOW:]
    return any(tok in NEGATION_TERMS for tok in window)


def score_concepts(text: str, labels: str, dictionary: dict[str, dict[str, list[str] | str]]) -> dict[str, float]:
    norm_text = normalize_text(text)
    label_text = normalize_text(labels)
    scores = {concept: 0.0 for concept in TARGET_CONCEPTS}

    for key, entry in dictionary.items():
        canonical = str(entry.get("canonical", key))
        phrases = [key.replace("_", " ")] + list(entry.get("synonyms", []))
        for phrase in phrases:
            phrase_norm = normalize_text(phrase)
            if not phrase_norm:
                continue
            for match in re.finditer(rf"\b{re.escape(phrase_norm)}\b", norm_text):
                if not is_negated(norm_text, match.start()):
                    scores[canonical] = scores.get(canonical, 0.0) + 1.0
            if phrase_norm in label_text:
                scores[canonical] = scores.get(canonical, 0.0) + 1.5

    if scores.get("normal", 0.0) > 0 and sum(v for k, v in scores.items() if k != "normal") > 0:
        scores["normal"] *= 0.3
    return scores


def select_concepts(scores: dict[str, float], min_concepts: int, max_concepts: int) -> list[str]:
    ranked = [k for k, v in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if v > 0]
    if not ranked:
        ranked = ["normal"]
    if len(ranked) < min_concepts:
        ranked.extend([c for c in GENERIC_CONCEPTS if c not in ranked])
    return ranked[:max_concepts]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset_tsv, sep="\t").fillna("")
    dictionary = load_dictionary(args.dictionary)

    rows = []
    for row in df.to_dict("records"):
        text = " ".join([row.get("findings", ""), row.get("impression", ""), row.get("indication", "")])
        labels = " ".join([row.get("labels_available", ""), row.get("major_mesh", ""), row.get("minor_mesh", "")])
        scores = score_concepts(text, labels, dictionary)
        concepts = select_concepts(scores, args.min_concepts, args.max_concepts)
        out = dict(row)
        out["concepts"] = "|".join(concepts)
        out["concept_scores"] = "|".join(f"{k}:{scores.get(k, 0.0):.3f}" for k in TARGET_CONCEPTS)
        for concept in TARGET_CONCEPTS:
            out[f"label_{concept}"] = int(concept in concepts and scores.get(concept, 0.0) > 0)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    ensure_dir(args.out_tsv.parent)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_tsv}")


if __name__ == "__main__":
    main()
