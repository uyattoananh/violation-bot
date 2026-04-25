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
_ADAPTER = None
_MODEL_LOCK = threading.Lock()
_MODEL_NAME = "clip-ViT-B-32"


def _load_model():
    global _MODEL, _ADAPTER
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        log.info("Loading CLIP model %s (first run downloads ~600MB)...", _MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
        # Optional adapter — specializes CLIP to construction-violation domain.
        try:
            from src.clip_adapter import load_adapter
            _ADAPTER = load_adapter()
            if _ADAPTER is not None:
                log.info("Domain adapter applied on top of CLIP.")
        except Exception as e:  # noqa: BLE001
            log.debug("No adapter applied: %s", e)
            _ADAPTER = None
        log.info("CLIP model ready.")
    return _MODEL


def _apply_adapter(vecs: np.ndarray) -> np.ndarray:
    """If an adapter is loaded, pass vectors through it. Input/output are
    both (N, 512) L2-normalized float32 arrays."""
    if _ADAPTER is None:
        return vecs
    import torch
    with torch.no_grad():
        t = torch.from_numpy(vecs.astype(np.float32))
        out = _ADAPTER(t).cpu().numpy().astype(np.float32)
    return out


def embed_image(image_path: Path) -> np.ndarray:
    """Return a 512-dim L2-normalized embedding for one image.
    If an adapter is loaded, the vector passes through it."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    model = _load_model()
    vec = model.encode([img], normalize_embeddings=True, convert_to_numpy=True)[0]
    vec = vec.astype(np.float32)
    return _apply_adapter(vec.reshape(1, -1))[0]


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
        vecs = _apply_adapter(vecs.astype(np.float32))
        for (j, _), v in zip(valid, vecs):
            out[i + j] = v
    return out
