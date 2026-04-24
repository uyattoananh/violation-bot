"""
Evaluation script — generates detailed metrics, confusion matrix, and per-class report.

Usage:
    python -m src.evaluate --config configs/train_config.yaml --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelConfusionMatrix,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import ViolationDataset, get_val_transforms
from src.model import build_model
from src.taxonomy import CLASS_LABELS_EN, CLASS_NAMES, NUM_CLASSES, VIOLATION_CLASSES


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run inference on entire dataset. Returns (all_probs, all_labels, filenames)."""
    model.eval()
    all_probs, all_labels, all_fnames = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["labels"]
        logits = model(images)
        probs = torch.sigmoid(logits).cpu()

        all_probs.append(probs)
        all_labels.append(labels)
        all_fnames.extend(batch["filename"])

    return torch.cat(all_probs), torch.cat(all_labels), all_fnames


def per_class_report(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> list[dict]:
    """Compute precision, recall, F1 per class."""
    preds = (probs >= threshold).int()
    labels_int = labels.int()
    report = []

    for i in range(NUM_CLASSES):
        tp = ((preds[:, i] == 1) & (labels_int[:, i] == 1)).sum().item()
        fp = ((preds[:, i] == 1) & (labels_int[:, i] == 0)).sum().item()
        fn = ((preds[:, i] == 0) & (labels_int[:, i] == 1)).sum().item()
        tn = ((preds[:, i] == 0) & (labels_int[:, i] == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report.append({
            "class_index": i,
            "name": CLASS_NAMES[i],
            "label": CLASS_LABELS_EN[i],
            "support": int(labels_int[:, i].sum().item()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return report


def plot_confusion_matrices(probs: torch.Tensor, labels: torch.Tensor, threshold: float, output_dir: str):
    """Plot per-class binary confusion matrices in a grid."""
    preds = (probs >= threshold).int()
    cm_metric = MultilabelConfusionMatrix(num_labels=NUM_CLASSES, threshold=threshold)
    cm = cm_metric(probs, labels.int()).numpy()  # shape: (num_classes, 2, 2)

    cols = 5
    rows = (NUM_CLASSES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i in range(NUM_CLASSES):
        sns.heatmap(
            cm[i], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred -", "Pred +"],
            yticklabels=["True -", "True +"],
            ax=axes[i],
        )
        axes[i].set_title(CLASS_NAMES[i], fontsize=8)

    # Hide unused subplots
    for j in range(NUM_CLASSES, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Per-Class Confusion Matrices", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_f1_by_class(report: list[dict], output_dir: str):
    """Bar chart of F1 scores per class."""
    names = [r["name"] for r in report]
    f1s = [r["f1"] for r in report]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if f < 0.5 else "#f39c12" if f < 0.7 else "#2ecc71" for f in f1s]
    ax.barh(names, f1s, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 Score")
    ax.set_title("F1 Score by Violation Class")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Minimum threshold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "f1_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def find_hard_examples(probs: torch.Tensor, labels: torch.Tensor, filenames: list[str], threshold: float, top_n: int = 20) -> dict:
    """Find images the model is most confused about."""
    preds = (probs >= threshold).float()
    errors = (preds != labels).float()

    # Images with the most misclassified labels
    error_counts = errors.sum(dim=1)
    worst_indices = error_counts.argsort(descending=True)[:top_n]

    hard_examples = []
    for idx in worst_indices:
        i = idx.item()
        wrong_classes = torch.where(errors[i] == 1)[0].tolist()
        hard_examples.append({
            "filename": filenames[i],
            "num_errors": int(error_counts[i].item()),
            "wrong_classes": [CLASS_NAMES[c] for c in wrong_classes],
        })

    return hard_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = config["evaluation"]["threshold"]

    # Load model
    model_cfg = config["model"]
    model = build_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    # Load data
    labels_dir = os.path.dirname(config["data"]["labels_csv"])
    csv_path = os.path.join(labels_dir, f"{args.split}.csv")
    image_dir = config["data"]["raw_dir"]
    transforms = get_val_transforms(config)
    dataset = ViolationDataset(csv_path, image_dir, transform=transforms)
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["training"]["num_workers"],
    )
    print(f"Evaluating on {args.split} set: {len(dataset)} samples")

    # Collect predictions
    probs, labels, filenames = collect_predictions(model, loader, device)

    # Metrics
    report = per_class_report(probs, labels, threshold)
    macro_f1 = np.mean([r["f1"] for r in report])
    macro_p = np.mean([r["precision"] for r in report])
    macro_r = np.mean([r["recall"] for r in report])

    print(f"\n{'='*60}")
    print(f"Results on {args.split} set (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_p:.4f}")
    print(f"  Macro Recall:    {macro_r:.4f}")
    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<35} {'Support':>8} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-'*67}")
    for r in report:
        print(f"  {r['name']:<35} {r['support']:>8} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

    # Hard examples
    hard = find_hard_examples(probs, labels, filenames, threshold)

    # Save everything
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "split": args.split,
        "threshold": threshold,
        "checkpoint": args.checkpoint,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "per_class": report,
        "hard_examples": hard,
    }
    json_path = os.path.join(output_dir, f"eval_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    # Plots
    plot_confusion_matrices(probs, labels, threshold, output_dir)
    plot_f1_by_class(report, output_dir)


if __name__ == "__main__":
    main()
