"""CLIP image embeddings for photo-RAG.

Uses sentence-transformers' `clip-ViT-B-32` (512-dim, CPU-friendly).
First call downloads the model (~600 MB); cached on disk thereafter.

API:
    embed_image(Path) -> np.ndarray   # 512 floats, L2-normalized
    embed_images([Path]) -> np.ndarray of shape (N, 512)

The embedding space is normalized so cosine distance ~ 1 - dot product.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_NAME = "clip-ViT-B-32"


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        log.info("Loading CLIP model %s (first run downloads ~600MB)...", _MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
        log.info("CLIP model ready.")
    return _MODEL


def embed_image(image_path: Path) -> np.ndarray:
    """Return a 512-dim L2-normalized embedding for one image."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    model = _load_model()
    vec = model.encode([img], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32)


def embed_images(image_paths: Iterable[Path], batch_size: int = 16) -> np.ndarray:
    """Batch embed a list of image paths. Returns (N, 512) float32 array."""
    from PIL import Image
    model = _load_model()
    paths = list(image_paths)
    out = np.zeros((len(paths), 512), dtype=np.float32)
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                log.warning("skipping %s: %s", p, e)
                imgs.append(None)
        valid = [(j, im) for j, im in enumerate(imgs) if im is not None]
        if not valid:
            continue
        valid_imgs = [im for _, im in valid]
        vecs = model.encode(valid_imgs, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        for (j, _), v in zip(valid, vecs):
            out[i + j] = v.astype(np.float32)
    return out
