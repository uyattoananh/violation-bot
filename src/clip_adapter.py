"""CLIP embedding adapter — small MLP that specializes the generic CLIP
ViT-B/32 space to Vietnamese construction safety violations.

Input/output are both 512-dim L2-normalized vectors (same as raw CLIP), so
the adapter is a drop-in replacement everywhere CLIP embeddings are used:
pgvector rows, k-NN retrieval, RAG prompt hints.

Architecture:
    512 -> Linear(512, 256) -> ReLU -> Linear(256, 512) -> residual add -> L2-normalize

Residual connection means the adapter starts near-identity at random init,
so early training doesn't destroy the already-useful CLIP signal. Total
trainable params: ~150K — trainable on CPU in under 30 minutes on our
1,108-photo dataset.

Weights file lives at `src/clip_adapter.pt` (small, ~600KB — fine to commit).
If present, `src/embeddings.py` loads and applies it automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

ADAPTER_PATH = Path(__file__).parent / "clip_adapter.pt"


class CLIPAdapter(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 512) L2-normalized CLIP embedding. Returns (B, 512) L2-normalized."""
        h = F.relu(self.fc1(x))
        delta = self.fc2(h)
        out = x + delta  # residual
        return F.normalize(out, dim=-1)


def load_adapter(path: Path = ADAPTER_PATH, device: str = "cpu") -> CLIPAdapter | None:
    """Load the adapter from disk. Returns None if the weights file doesn't exist."""
    if not path.exists():
        return None
    model = CLIPAdapter()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        log.info("CLIP adapter loaded from %s", path)
        return model
    except Exception as e:  # noqa: BLE001
        log.warning("CLIP adapter at %s failed to load: %s — falling back to raw CLIP", path, e)
        return None
