"""
Single-image and batch prediction script.

Usage:
    # Single image
    python -m src.predict --image path/to/photo.jpg

    # Batch (folder)
    python -m src.predict --image_dir path/to/folder --output results.json

    # With custom threshold
    python -m src.predict --image path/to/photo.jpg --threshold 0.4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_val_transforms
from src.model import build_model
from src.taxonomy import NUM_CLASSES, VIOLATION_CLASSES, format_prediction


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    model = build_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_single(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run prediction on a single image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    violations = []
    for i in range(NUM_CLASSES):
        if probs[i] >= threshold:
            violations.append(format_prediction(i, float(probs[i])))

    # Sort by confidence descending
    violations.sort(key=lambda v: v["confidence"], reverse=True)

    return {
        "image": os.path.basename(image_path),
        "path": image_path,
        "num_violations": len(violations),
        "compliant": len(violations) == 0,
        "violations": violations,
        "all_scores": {
            VIOLATION_CLASSES[i]["name"]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        },
    }


def predict_batch(
    model: torch.nn.Module,
    image_dir: str,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> list[dict]:
    """Run prediction on all images in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []

    files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in extensions
    ])
    print(f"Found {len(files)} images in {image_dir}")

    for fname in files:
        path = os.path.join(image_dir, fname)
        try:
            result = predict_single(model, path, transform, device, threshold)
            results.append(result)
            status = "COMPLIANT" if result["compliant"] else f"{result['num_violations']} violation(s)"
            print(f"  {fname}: {status}")
        except Exception as e:
            print(f"  {fname}: ERROR — {e}")

    # Summary
    total = len(results)
    compliant = sum(1 for r in results if r["compliant"])
    print(f"\nSummary: {compliant}/{total} compliant, {total - compliant}/{total} with violations")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a single image")
    parser.add_argument("--image_dir", default=None, help="Path to folder of images")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image_dir")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold or config["evaluation"]["threshold"]

    print(f"Device: {device}")
    print(f"Threshold: {threshold}")

    model = load_model(config, args.checkpoint, device)
    transform = get_val_transforms(config)

    if args.image:
        result = predict_single(model, args.image, transform, device, threshold)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved: {args.output}")
    else:
        results = predict_batch(model, args.image_dir, transform, device, threshold)

        output_path = args.output or "reports/batch_predictions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
