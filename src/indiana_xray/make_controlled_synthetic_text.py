from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import TARGET_CONCEPTS
from .utils import ensure_dir


NORMAL_SENTENCE = "No acute cardiopulmonary abnormality."
GENERIC_CONCEPTS = {"chest_xray", "lungs", "heart", "pleura", "mediastinum", "thorax"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a closed-vocabulary synthetic text TSV from normalized clinical concepts."
    )
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10, help="Maximum abnormal concepts kept in the controlled vocabulary.")
    parser.add_argument("--max-terms-per-image", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=1)
    return parser.parse_args()


def row_positive_concepts(row: dict[str, object]) -> list[str]:
    positives: list[str] = []
    for concept in TARGET_CONCEPTS:
        if concept == "normal":
            continue
        value = row.get(f"label_{concept}", 0)
        try:
            is_positive = float(value) > 0
        except (TypeError, ValueError):
            is_positive = str(value).strip().lower() in {"true", "yes", "y"}
        if is_positive:
            positives.append(concept)

    concepts = str(row.get("concepts", "")).split("|")
    for concept in concepts:
        concept = concept.strip()
        if concept and concept in TARGET_CONCEPTS and concept != "normal" and concept not in positives:
            positives.append(concept)
    return positives


def controlled_vocabulary(df: pd.DataFrame, top_k: int, min_count: int) -> tuple[list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for row in df.to_dict("records"):
        counts.update(row_positive_concepts(row))
    ranked = [
        concept
        for concept, count in counts.most_common()
        if concept not in GENERIC_CONCEPTS and count >= min_count
    ]
    return ranked[:top_k], counts


def sentence_from_terms(terms: list[str]) -> str:
    if not terms:
        return NORMAL_SENTENCE
    names = [CONCEPT_TEXT.get(term, term.replace("_", " ")) for term in terms]
    if len(names) == 1:
        return names[0].capitalize() + "."
    if len(names) == 2:
        return f"{names[0].capitalize()} and {names[1]}."
    return ", ".join(names[:-1]).capitalize() + f", and {names[-1]}."


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_tsv, sep="\t").fillna("")
    vocab, counts = controlled_vocabulary(df, args.top_k, args.min_count)
    allowed = set(vocab)

    rows: list[dict[str, object]] = []
    for row in df.to_dict("records"):
        positives = [concept for concept in row_positive_concepts(row) if concept in allowed]
        positives = positives[: args.max_terms_per_image]
        sentence = sentence_from_terms(positives)
        out = dict(row)
        out["controlled_terms"] = "|".join(positives if positives else ["normal"])
        out["synthetic_anomaly_text"] = sentence
        out["clip_text"] = f"Chest xray: {sentence}"
        out["next_token_text"] = sentence
        out["synthetic_model"] = "controlled_concept_vocabulary"
        out["synthetic_prompt_version"] = f"controlled_top{len(vocab)}_max{args.max_terms_per_image}"
        rows.append(out)

    out_df = pd.DataFrame(rows)
    ensure_dir(args.out_tsv.parent)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)

    print(f"Controlled vocabulary: {', '.join(vocab)}")
    print(f"Wrote {len(out_df)} rows to {args.out_tsv}")

    if args.summary_out:
        ensure_dir(args.summary_out.parent)
        summary = pd.DataFrame(
            [
                {
                    "concept": concept,
                    "count": int(counts.get(concept, 0)),
                    "kept": concept in allowed,
                    "term": CONCEPT_TEXT.get(concept, concept.replace("_", " ")),
                }
                for concept in TARGET_CONCEPTS
                if concept != "normal"
            ]
        ).sort_values(["kept", "count", "concept"], ascending=[False, False, True])
        summary.to_csv(args.summary_out, sep="\t", index=False)
        print(f"Wrote vocabulary summary to {args.summary_out}")


if __name__ == "__main__":
    main()
