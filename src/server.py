"""
FastAPI server for construction safety violation classification.

Usage:
    python -m src.server
    python -m src.server --checkpoint checkpoints/best_model.pt --port 8000

API docs available at http://localhost:8000/docs
"""

import argparse
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_val_transforms
from src.model import build_model, count_parameters
from src.taxonomy import (
    CATEGORIES,
    CATEGORY_EN,
    CLASS_LABELS,
    CLASS_LABELS_EN,
    CLASS_NAMES,
    NUM_CLASSES,
    VIOLATION_CLASSES,
    format_prediction,
    get_classes_by_category,
)

# ---------------------------------------------------------------------------
# Globals — populated on startup
# ---------------------------------------------------------------------------
model: Optional[torch.nn.Module] = None
transform = None
device: Optional[torch.device] = None
config: Optional[dict] = None
model_info: dict = {}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_on_startup(config_path: str, checkpoint_path: str):
    """Load model and transforms into global state."""
    global model, transform, device, config, model_info

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"]
    model = build_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
    )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model_info = {
            "checkpoint": checkpoint_path,
            "epoch": ckpt.get("epoch", "unknown"),
            "metrics": ckpt.get("metrics", {}),
        }
    else:
        model_info = {
            "checkpoint": None,
            "warning": f"Checkpoint not found: {checkpoint_path}. Model has random weights.",
        }

    model = model.to(device)
    model.eval()

    params = count_parameters(model)
    model_info.update({
        "architecture": model_cfg["architecture"],
        "num_classes": model_cfg["num_classes"],
        "device": str(device),
        "parameters": params,
    })

    transform = get_val_transforms(config)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
# Parse args before app creation so lifespan can use them
_parser = argparse.ArgumentParser()
_parser.add_argument("--config", default="configs/train_config.yaml")
_parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
_parser.add_argument("--host", default="0.0.0.0")
_parser.add_argument("--port", type=int, default=8000)
_args, _ = _parser.parse_known_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    print(f"Loading model from {_args.checkpoint}...")
    load_model_on_startup(_args.config, _args.checkpoint)
    print(f"Model loaded on {device}. Server ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Construction Safety Violation Classifier",
    description="Identifies safety violations in construction site photos per Vietnamese regulations (QCVN 18:2021/BXD)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


async def read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an RGB numpy array."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, BMP, or WebP.",
        )
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return np.array(image)


def classify_image(image: np.ndarray, threshold: float) -> dict:
    """Run inference on a single image array."""
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    violations = []
    for i in range(NUM_CLASSES):
        if probs[i] >= threshold:
            violations.append(format_prediction(i, float(probs[i])))

    violations.sort(key=lambda v: v["confidence"], reverse=True)

    all_scores = {
        VIOLATION_CLASSES[i]["name"]: round(float(probs[i]), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "num_violations": len(violations),
        "compliant": len(violations) == 0,
        "violations": violations,
        "all_scores": all_scores,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Check server status, model info, and GPU availability."""
    return {
        "status": "ok",
        "model": model_info,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/taxonomy")
async def taxonomy():
    """Return the full violation taxonomy with Vietnamese regulation codes."""
    classes = []
    for idx in range(NUM_CLASSES):
        info = VIOLATION_CLASSES[idx]
        classes.append({
            "index": idx,
            "name": info["name"],
            "label_vi": info["label"],
            "label_en": info["label_en"],
            "regulation": info["regulation"],
            "penalty": info["penalty"],
            "fine_vnd": info["fine_vnd"],
            "severity": info["severity"],
            "category": info["category"],
        })

    return {
        "num_classes": NUM_CLASSES,
        "categories": {k: {"vi": CATEGORIES[k], "en": CATEGORY_EN[k]} for k in CATEGORIES},
        "classes": classes,
    }


@app.post("/classify")
async def classify(
    file: UploadFile = File(..., description="Construction site photo (JPEG, PNG, BMP, WebP)"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Confidence threshold for positive prediction"),
):
    """Classify a single construction site photo for safety violations."""
    start = time.time()
    image = await read_image(file)
    result = classify_image(image, threshold)
    elapsed = round(time.time() - start, 3)

    return {
        "image": file.filename,
        "threshold": threshold,
        "inference_time_s": elapsed,
        **result,
    }


@app.post("/classify/batch")
async def classify_batch(
    files: list[UploadFile] = File(..., description="Multiple construction site photos"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    """Classify multiple construction site photos in one request."""
    start = time.time()
    results = []

    for file in files:
        try:
            image = await read_image(file)
            result = classify_image(image, threshold)
            results.append({"image": file.filename, "status": "ok", **result})
        except HTTPException as e:
            results.append({"image": file.filename, "status": "error", "detail": e.detail})

    elapsed = round(time.time() - start, 3)

    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    compliant = sum(1 for r in results if r.get("compliant", False))
    total_violations = sum(r.get("num_violations", 0) for r in results)

    return {
        "total_images": total,
        "processed": ok,
        "compliant": compliant,
        "non_compliant": ok - compliant,
        "total_violations_found": total_violations,
        "threshold": threshold,
        "total_time_s": elapsed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=_args.host,
        port=_args.port,
        reload=False,
    )
