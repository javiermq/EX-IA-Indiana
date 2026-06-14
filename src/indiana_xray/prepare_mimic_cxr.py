from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import TARGET_CONCEPTS
from .utils import ensure_dir


CHEXPERT_TO_LOCAL = {
    "No Finding": "normal",
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Pleural Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Emphysema": "emphysema",
    "Lung Lesion": "nodule",
    "Lung Opacity": "opacity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train-compatible TSV from local MIMIC-CXR-JPG files.")
    parser.add_argument("--mimic-jpg-root", type=Path, required=True, help="Root containing MIMIC-CXR-JPG files/ and CSVs.")
    parser.add_argument("--mimic-reports-root", type=Path, default=None, help="Root containing MIMIC-CXR text reports files/.")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--chexpert-csv", type=Path, default=None)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["PA", "AP"], help="ViewPosition values to keep.")
    parser.add_argument("--split", choices=["train", "validate", "test", "all"], default="all")
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--synthetic-out-tsv", type=Path, default=None)
    return parser.parse_args()


def default_path(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    return direct


def clean_text(text: object) -> str:
    text = str(text or "")
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_report_sections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"findings": "", "impression": "", "indication": "", "comparison": ""}
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = {"findings": "", "impression": "", "indication": "", "comparison": ""}
    pattern = re.compile(
        r"(?im)^\s*(FINDINGS|IMPRESSION|INDICATION|HISTORY|COMPARISON)\s*:\s*"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        sections["findings"] = clean_text(text)
        return sections
    for i, match in enumerate(matches):
        label = match.group(1).lower()
        if label == "history":
            label = "indication"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if label in sections:
            sections[label] = clean_text(text[start:end])
    return sections


def report_path(reports_root: Path | None, subject_id: int, study_id: int) -> Path | None:
    if reports_root is None:
        return None
    subject = f"p{int(subject_id)}"
    parent = f"p{int(subject_id) // 1000000:02d}"
    return reports_root / "files" / parent / subject / f"s{int(study_id)}.txt"


def image_path(jpg_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    subject = f"p{int(subject_id)}"
    parent = f"p{int(subject_id) // 1000000:02d}"
    return jpg_root / "files" / parent / subject / f"s{int(study_id)}" / f"{dicom_id}.jpg"


def labels_from_chexpert(row: pd.Series | None) -> tuple[list[str], dict[str, int]]:
    labels: list[str] = []
    binary = {f"label_{v}": 0 for v in TARGET_CONCEPTS}
    if row is None:
        return labels, binary
    for source, target in CHEXPERT_TO_LOCAL.items():
        value = row.get(source, 0)
        positive = pd.notna(value) and float(value) == 1.0
        if positive:
            labels.append(target)
            binary[f"label_{target}"] = 1
    return labels, binary


def concise_impression(impression: str, findings: str, labels: list[str]) -> str:
    text = clean_text(impression) or clean_text(findings)
    if not text:
        if labels and labels != ["normal"]:
            return ", ".join(labels).replace("_", " ").capitalize() + "."
        return "No acute cardiopulmonary abnormality."
    text = re.sub(r"\b(No comparison|Comparison:?).*", "", text, flags=re.IGNORECASE).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentence = clean_text(sentences[0] if sentences else text)
    words = sentence.split()
    if len(words) > 24:
        sentence = " ".join(words[:24]).rstrip(" ,;:-") + "."
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence or "No acute cardiopulmonary abnormality."


def main() -> None:
    args = parse_args()
    jpg_root = args.mimic_jpg_root
    reports_root = args.mimic_reports_root
    metadata_csv = args.metadata_csv or default_path(jpg_root, "mimic-cxr-2.0.0-metadata.csv")
    split_csv = args.split_csv or default_path(jpg_root, "mimic-cxr-2.0.0-split.csv")
    chexpert_csv = args.chexpert_csv or default_path(jpg_root, "mimic-cxr-2.0.0-chexpert.csv")

    metadata = pd.read_csv(metadata_csv)
    if split_csv.exists():
        split = pd.read_csv(split_csv)
        metadata = metadata.merge(split, on=["dicom_id", "study_id", "subject_id"], how="left")
    else:
        metadata["split"] = "all"
    if args.split != "all":
        metadata = metadata[metadata["split"].eq(args.split)].copy()
    metadata = metadata[metadata["ViewPosition"].fillna("").isin(args.views)].copy()

    chexpert = pd.read_csv(chexpert_csv) if chexpert_csv.exists() else pd.DataFrame()
    if not chexpert.empty:
        chexpert = chexpert.set_index(["subject_id", "study_id"], drop=False)

    rows: list[dict[str, object]] = []
    seen_studies: set[tuple[int, int]] = set()
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Indexing MIMIC-CXR"):
        subject_id = int(row.subject_id)
        study_id = int(row.study_id)
        dicom_id = str(row.dicom_id)
        study_key = (subject_id, study_id)
        if args.max_studies is not None and study_key not in seen_studies and len(seen_studies) >= args.max_studies:
            continue
        seen_studies.add(study_key)
        img = image_path(jpg_root, subject_id, study_id, dicom_id)
        if not img.exists():
            continue
        rpt = report_path(reports_root, subject_id, study_id)
        sections = parse_report_sections(rpt) if rpt else {"findings": "", "impression": "", "indication": "", "comparison": ""}
        cxr_series = None
        if not chexpert.empty and study_key in chexpert.index:
            cxr_series = chexpert.loc[study_key]
            if isinstance(cxr_series, pd.DataFrame):
                cxr_series = cxr_series.iloc[0]
        labels, binary = labels_from_chexpert(cxr_series)
        report_text = clean_text(" ".join([sections["findings"], sections["impression"], sections["indication"]]))
        out = {
            "report_id": f"s{study_id}",
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "image_id": dicom_id,
            "image_path": str(img.resolve()),
            "projection": str(row.ViewPosition).lower(),
            "split": getattr(row, "split", "all"),
            "caption": str(row.ViewPosition),
            "findings": sections["findings"],
            "impression": sections["impression"],
            "indication": sections["indication"],
            "comparison": sections["comparison"],
            "report_text": report_text,
            "major_mesh": "",
            "minor_mesh": "",
            "labels_available": "|".join(labels),
            "grounding_json": "[]",
            "has_grounding": False,
        }
        out.update(binary)
        rows.append(out)

    df = pd.DataFrame(rows)
    ensure_dir(args.out_tsv.parent)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(df)} image rows to {args.out_tsv}")

    if args.synthetic_out_tsv:
        syn = df.copy()
        syn["report_text_clean"] = syn["report_text"].map(clean_text)
        syn["synthetic_anomaly_text"] = [
            concise_impression(row.impression, row.findings, str(row.labels_available).split("|"))
            for row in syn.itertuples(index=False)
        ]
        syn["clip_text"] = "Chest xray: " + syn["synthetic_anomaly_text"]
        syn["next_token_text"] = syn["synthetic_anomaly_text"]
        syn["synthetic_model"] = "mimic_impression"
        syn["synthetic_prompt_version"] = "mimic_impression_v1"
        ensure_dir(args.synthetic_out_tsv.parent)
        syn.to_csv(args.synthetic_out_tsv, sep="\t", index=False)
        print(f"Wrote {len(syn)} synthetic rows to {args.synthetic_out_tsv}")


if __name__ == "__main__":
    main()
