from __future__ import annotations

from pathlib import Path


OPENI_IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
OPENI_REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

DEFAULT_DATA_ROOT = Path("data/indiana")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"

TARGET_CONCEPTS = [
    "normal",
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
    "emphysema",
    "nodule",
    "opacity",
    "infiltrate",
]

