"""
Custom PyTorch Dataset for multi-label construction safety violation classification.

Expects a CSV with columns: filename, no_hard_hat, no_high_vis_vest, ..., ladder_violation
Each label column is 0 or 1. Images are loaded from a flat directory.
"""

import os
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.taxonomy import CLASS_NAMES


class ViolationDataset(Dataset):
    """Multi-label dataset for construction safety violations."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        class_names: Optional[list] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES

        # Validate that all expected label columns exist
        missing = [c for c in self.class_names if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Label columns missing from CSV: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = row["filename"]
        image_path = os.path.join(self.image_dir, filename)

        # Load image as RGB numpy array
        image = np.array(Image.open(image_path).convert("RGB"))

        # Extract multi-label target vector
        labels = torch.tensor(
            [float(row[c]) for c in self.class_names], dtype=torch.float32
        )

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {"image": image, "labels": labels, "filename": filename}

    def get_class_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

        pos_weight[i] = num_negatives[i] / num_positives[i]
        Higher weight for rare classes pushes the model to catch them.
        """
        label_matrix = self.df[self.class_names].values
        pos_counts = label_matrix.sum(axis=0)
        neg_counts = len(self.df) - pos_counts

        # Avoid division by zero for classes with no positive samples
        pos_counts = np.clip(pos_counts, a_min=1, a_max=None)
        weights = neg_counts / pos_counts

        return torch.tensor(weights, dtype=torch.float32)

    def get_label_distribution(self) -> dict:
        """Return count of positive samples per class for inspection."""
        label_matrix = self.df[self.class_names].values
        pos_counts = label_matrix.sum(axis=0).astype(int)
        return dict(zip(self.class_names, pos_counts))


def get_train_transforms(config: dict) -> A.Compose:
    """Build training augmentation pipeline from config."""
    aug = config["augmentation"]
    t = aug["train"]
    norm = t["normalize"]

    return A.Compose([
        A.RandomResizedCrop(
            height=aug["image_size"],
            width=aug["image_size"],
            scale=tuple(t["random_resized_crop"]["scale"]),
            ratio=tuple(t["random_resized_crop"]["ratio"]),
        ),
        A.HorizontalFlip(p=t["horizontal_flip"]),
        A.Rotate(limit=t["random_rotation"], p=0.5),
        A.ColorJitter(
            brightness=t["color_jitter"]["brightness"],
            contrast=t["color_jitter"]["contrast"],
            saturation=t["color_jitter"]["saturation"],
            hue=t["color_jitter"]["hue"],
            p=0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=t["gaussian_blur"]),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


def get_val_transforms(config: dict) -> A.Compose:
    """Build validation/test transform pipeline from config."""
    aug = config["augmentation"]
    v = aug["val"]
    norm = v["normalize"]

    return A.Compose([
        A.Resize(height=v["resize"], width=v["resize"]),
        A.CenterCrop(height=v["center_crop"], width=v["center_crop"]),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])
