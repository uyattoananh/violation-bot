"""
Data preparation script.

Usage:
    python scripts/prepare_data.py --config configs/train_config.yaml

This script:
1. Scans raw image directory for supported image files
2. Validates images can be opened
3. Creates a blank labels CSV template (if none exists) for manual annotation
4. Splits an existing labeled CSV into train/val/test sets (stratified)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxonomy import CLASS_NAMES


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def scan_images(raw_dir: str) -> list[str]:
    """Find all valid image files in the raw directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for fname in sorted(os.listdir(raw_dir)):
        if Path(fname).suffix.lower() in extensions:
            files.append(fname)
    return files


def validate_images(raw_dir: str, filenames: list[str]) -> tuple[list, list]:
    """Check that images can be opened. Return (valid, corrupt) lists."""
    valid, corrupt = [], []
    for fname in filenames:
        try:
            img = Image.open(os.path.join(raw_dir, fname))
            img.verify()
            valid.append(fname)
        except Exception as e:
            print(f"  CORRUPT: {fname} — {e}")
            corrupt.append(fname)
    return valid, corrupt


def create_label_template(filenames: list[str], output_path: str):
    """Create a blank CSV with filename + one column per violation class."""
    df = pd.DataFrame({"filename": filenames})
    for cls in CLASS_NAMES:
        df[cls] = 0  # default: no violation
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Label template created: {output_path}")
    print(f"  {len(filenames)} images, {len(CLASS_NAMES)} classes")
    print(f"  Fill in 0/1 values for each violation class, then re-run with --split")


def split_labels(
    labels_csv: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """Split labeled CSV into train/val/test CSVs using stratified sampling."""
    df = pd.read_csv(labels_csv)
    n = len(df)
    print(f"Loaded {n} labeled samples from {labels_csv}")

    # Print class distribution
    print("\nClass distribution:")
    for cls in CLASS_NAMES:
        if cls in df.columns:
            count = df[cls].sum()
            print(f"  {cls}: {count} ({count/n*100:.1f}%)")

    # For stratification, create a composite label string
    # This helps maintain similar class ratios across splits
    label_cols = [c for c in CLASS_NAMES if c in df.columns]
    df["_stratify_key"] = df[label_cols].astype(str).agg("-".join, axis=1)

    # Check if any stratify group has < 2 samples (can't split)
    key_counts = df["_stratify_key"].value_counts()
    small_groups = key_counts[key_counts < 2]
    if len(small_groups) > 0:
        print(f"\nWarning: {len(small_groups)} label combinations have <2 samples.")
        print("Falling back to random split (no stratification).")
        stratify_col = None
    else:
        stratify_col = df["_stratify_key"]

    # Split: train / (val + test)
    test_val_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df, test_size=test_val_ratio, random_state=seed, stratify=stratify_col
    )

    # Split remaining into val / test
    val_fraction = val_ratio / test_val_ratio
    if stratify_col is not None:
        stratify_temp = temp_df["_stratify_key"]
        temp_key_counts = stratify_temp.value_counts()
        small_temp = temp_key_counts[temp_key_counts < 2]
        if len(small_temp) > 0:
            stratify_temp = None
    else:
        stratify_temp = None

    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_fraction), random_state=seed, stratify=stratify_temp
    )

    # Drop helper column and save
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.drop(columns=["_stratify_key"])
        out_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"\n{split_name}: {len(split_df)} samples → {out_path}")

    print("\nDone. Split complete.")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--split", action="store_true",
        help="Split existing labels.csv into train/val/test"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    labels_csv = data_cfg["labels_csv"]

    if args.split:
        # Split mode: requires a filled-in labels CSV
        if not os.path.exists(labels_csv):
            print(f"Error: {labels_csv} not found. Run without --split first to create the template.")
            sys.exit(1)
        split_labels(
            labels_csv=labels_csv,
            output_dir=os.path.dirname(labels_csv),
            train_ratio=data_cfg["split_ratios"]["train"],
            val_ratio=data_cfg["split_ratios"]["val"],
            seed=data_cfg["random_seed"],
        )
    else:
        # Scan and template mode
        if not os.path.isdir(raw_dir):
            print(f"Error: Raw image directory not found: {raw_dir}")
            print("Place your construction site photos in this directory and re-run.")
            sys.exit(1)

        print(f"Scanning {raw_dir} for images...")
        filenames = scan_images(raw_dir)
        print(f"Found {len(filenames)} image files")

        if len(filenames) == 0:
            print("No images found. Add .jpg/.png files to data/raw/ and re-run.")
            sys.exit(1)

        print("\nValidating images...")
        valid, corrupt = validate_images(raw_dir, filenames)
        print(f"Valid: {len(valid)}, Corrupt: {len(corrupt)}")

        if corrupt:
            print(f"\nRemove or fix corrupt files before labeling.")

        create_label_template(valid, labels_csv)


if __name__ == "__main__":
    main()
