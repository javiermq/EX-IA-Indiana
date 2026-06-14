from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DEFAULT_IMAGE_SIZE, TARGET_CONCEPTS


def image_transforms(train: bool, image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    aug = []
    if train:
        aug.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        aug.extend([transforms.Resize((image_size, image_size))])
    aug.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(aug)


class IndianaConceptDataset(Dataset):
    def __init__(
        self,
        tsv_path: str | Path,
        indices: list[int] | None = None,
        train: bool = False,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.df = pd.read_csv(tsv_path, sep="\t").fillna("")
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = image_transforms(train=train, image_size=image_size)
        self.label_cols = [f"label_{name}" for name in TARGET_CONCEPTS]
        missing = [col for col in self.label_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing concept label columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.label_cols].astype(float).to_numpy(), dtype=torch.float32)
        concepts = [c for c in str(row.get("concepts", "")).split("|") if c]
        text = ", ".join(concepts) if concepts else str(row.get("impression", "") or row.get("findings", ""))
        return {
            "image": self.transform(image),
            "labels": labels,
            "text": text,
            "image_id": str(row.get("image_id", idx)),
            "image_path": str(image_path),
            "concepts": "|".join(concepts),
        }
