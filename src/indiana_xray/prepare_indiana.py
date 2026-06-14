from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import DEFAULT_DATA_ROOT, OPENI_IMAGES_URL, OPENI_REPORTS_URL
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and index Indiana / IU-Xray.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source", choices=["openi", "hf"], default="openi")
    parser.add_argument("--hf-repo", default="ykumards/open-i")
    parser.add_argument("--hf-image-size", type=int, default=384)
    parser.add_argument("--hf-image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--max-reports", type=int, default=None)
    parser.add_argument("--images-url", default=OPENI_IMAGES_URL)
    parser.add_argument("--reports-url", default=OPENI_REPORTS_URL)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--out-tsv", type=Path, default=None)
    return parser.parse_args()


def download(url: str, dst: Path, retries: int = 5) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Already downloaded: {dst}")
        return
    ensure_dir(dst.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, dst.open("wb") as handle:
                total = int(response.headers.get("Content-Length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bar.update(len(chunk))
            return
        except Exception as exc:
            last_error = exc
            if dst.exists():
                dst.unlink()
            wait = 2 * attempt
            print(f"Download failed ({attempt}/{retries}): {exc}. Retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {url}") from last_error


def extract_tgz(archive: Path, dst: Path) -> None:
    marker = dst / f".extracted_{archive.stem}"
    if marker.exists():
        print(f"Already extracted: {archive}")
        return
    ensure_dir(dst)
    with tarfile.open(archive, "r:gz") as tar:
        dst_resolved = dst.resolve()
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            if not str(target).startswith(str(dst_resolved)):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(dst)
    marker.write_text("ok", encoding="utf-8")


def text_or_empty(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return re.sub(r"\s+", " ", elem.text).strip()


def section_text(root: ET.Element, section_name: str) -> str:
    for abstract_text in root.findall(".//AbstractText"):
        if abstract_text.attrib.get("Label", "").lower() == section_name.lower():
            return text_or_empty(abstract_text)
    return ""


def collect_terms(root: ET.Element, tag: str) -> list[str]:
    terms: list[str] = []
    for node in root.findall(f".//{tag}"):
        value = text_or_empty(node)
        if value:
            terms.append(value)
    return sorted(set(terms))


def collect_grounding(root: ET.Element) -> list[dict[str, object]]:
    grounding: list[dict[str, object]] = []
    coord_keys = {"x", "y", "w", "h", "width", "height", "xmin", "ymin", "xmax", "ymax"}
    for node in root.iter():
        attrs = {k.lower(): v for k, v in node.attrib.items()}
        if coord_keys.intersection(attrs):
            grounding.append(
                {
                    "tag": node.tag,
                    "text": text_or_empty(node),
                    "attrs": attrs,
                }
            )
    return grounding


def image_projection(parent_image: ET.Element) -> str:
    caption = parent_image.find("caption")
    text = text_or_empty(caption).lower()
    if "frontal" in text or "pa and lateral" in text or "pa" in text or "ap" in text:
        return "frontal"
    if "lateral" in text:
        return "lateral"
    return "unknown"


def find_images(images_dir: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    for path in images_dir.rglob("*.png"):
        image_map[path.stem] = path
    return image_map


def iter_reports(reports_dir: Path) -> Iterable[Path]:
    yield from sorted(reports_dir.rglob("*.xml"))


def build_master_tsv(data_root: Path, out_tsv: Path) -> pd.DataFrame:
    images_dir = data_root / "images"
    reports_dir = data_root / "reports"
    image_map = find_images(images_dir)
    rows: list[dict[str, object]] = []

    for report_path in tqdm(list(iter_reports(reports_dir)), desc="Parsing XML reports"):
        root = ET.parse(report_path).getroot()
        report_id = report_path.stem
        findings = section_text(root, "FINDINGS")
        impression = section_text(root, "IMPRESSION")
        indication = section_text(root, "INDICATION")
        comparison = section_text(root, "COMPARISON")
        full_report = " ".join(x for x in [findings, impression, indication] if x)
        major_mesh = collect_terms(root, "major")
        minor_mesh = collect_terms(root, "minor")
        automatic_terms = sorted(set(major_mesh + minor_mesh))
        grounding = collect_grounding(root)

        parent_images = root.findall("parentImage")
        for parent_image in parent_images:
            image_id = parent_image.attrib.get("id", "")
            image_path = image_map.get(image_id)
            if image_path is None:
                continue
            rows.append(
                {
                    "report_id": report_id,
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "projection": image_projection(parent_image),
                    "caption": text_or_empty(parent_image.find("caption")),
                    "findings": findings,
                    "impression": impression,
                    "indication": indication,
                    "comparison": comparison,
                    "report_text": full_report,
                    "major_mesh": "|".join(major_mesh),
                    "minor_mesh": "|".join(minor_mesh),
                    "labels_available": "|".join(automatic_terms),
                    "grounding_json": json.dumps(grounding, ensure_ascii=False),
                    "has_grounding": bool(grounding),
                }
            )

    df = pd.DataFrame(rows).sort_values(["report_id", "image_id"])
    ensure_dir(out_tsv.parent)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {len(df)} image rows to {out_tsv}")
    return df


def hf_file_list(repo_id: str) -> list[str]:
    api = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/data?recursive=false"
    with urllib.request.urlopen(api, timeout=60) as response:
        payload = pd.read_json(BytesIO(response.read()))
    return [str(path) for path in payload["path"].tolist() if str(path).endswith(".parquet")]


def download_hf_parquets(repo_id: str, data_root: Path) -> list[Path]:
    parquet_dir = ensure_dir(data_root / "hf_parquet")
    paths = []
    for repo_path in hf_file_list(repo_id):
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"
        dst = parquet_dir / Path(repo_path).name
        download(url, dst)
        paths.append(dst)
    return paths


def save_image_bytes(image_bytes: bytes, path: Path, max_size: int, image_format: str) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        ensure_dir(path.parent)
        if image_format == "jpg":
            image.save(path, quality=88, optimize=True)
        else:
            image.save(path, optimize=True)
        return True
    except Exception:
        return False


def build_master_from_hf(
    repo_id: str,
    data_root: Path,
    out_tsv: Path,
    skip_download: bool,
    image_size: int,
    image_format: str,
    max_reports: int | None,
) -> pd.DataFrame:
    parquet_paths = sorted((data_root / "hf_parquet").glob("*.parquet")) if skip_download else []
    if not parquet_paths:
        parquet_paths = download_hf_parquets(repo_id, data_root)

    images_dir = ensure_dir(data_root / "images")
    rows: list[dict[str, object]] = []
    for parquet_path in tqdm(parquet_paths, desc="Reading HF parquet shards"):
        df = pd.read_parquet(parquet_path)
        for row in tqdm(df.to_dict("records"), leave=False):
            if max_reports is not None and len({r["report_id"] for r in rows}) >= max_reports:
                break
            uid = str(row.get("uid", ""))
            report_text = " ".join(
                str(row.get(k, "") or "") for k in ["findings", "impression", "indication"] if row.get(k, "")
            )
            labels = str(row.get("Problems", "") or row.get("MeSH", "") or "")
            for projection, column in [("frontal", "img_frontal"), ("lateral", "img_lateral")]:
                blob = row.get(column)
                if blob is None:
                    continue
                image_id = f"{uid}_{projection}"
                image_path = images_dir / f"{image_id}.{image_format}"
                if not save_image_bytes(blob, image_path, image_size, image_format):
                    continue
                rows.append(
                    {
                        "report_id": uid,
                        "image_id": image_id,
                        "image_path": str(image_path.resolve()),
                        "projection": projection,
                        "caption": projection,
                        "findings": str(row.get("findings", "") or ""),
                        "impression": str(row.get("impression", "") or ""),
                        "indication": str(row.get("indication", "") or ""),
                        "comparison": str(row.get("comparison", "") or ""),
                        "report_text": report_text,
                        "major_mesh": str(row.get("MeSH", "") or ""),
                        "minor_mesh": "",
                        "labels_available": labels,
                        "grounding_json": "[]",
                        "has_grounding": False,
                    }
                )
        if max_reports is not None and len({r["report_id"] for r in rows}) >= max_reports:
            break

    out_df = pd.DataFrame(rows).sort_values(["report_id", "projection"])
    ensure_dir(out_tsv.parent)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out_df)} image rows to {out_tsv}")
    return out_df


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    archives_dir = ensure_dir(data_root / "archives")
    images_dir = ensure_dir(data_root / "images")
    reports_dir = ensure_dir(data_root / "reports")

    out_tsv = args.out_tsv or data_root / "indiana_master.tsv"
    if args.source == "hf":
        build_master_from_hf(
            args.hf_repo,
            data_root,
            out_tsv,
            args.skip_download,
            args.hf_image_size,
            args.hf_image_format,
            args.max_reports,
        )
        return

    if not args.skip_download:
        images_archive = archives_dir / "NLMCXR_png.tgz"
        reports_archive = archives_dir / "NLMCXR_reports.tgz"
        download(args.images_url, images_archive)
        download(args.reports_url, reports_archive)
        extract_tgz(images_archive, images_dir)
        extract_tgz(reports_archive, reports_dir)

    build_master_tsv(data_root, out_tsv)


if __name__ == "__main__":
    main()
