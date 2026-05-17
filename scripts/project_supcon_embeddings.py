"""Project every photo_embeddings row through the trained SupCon
head and save to tmp/clip_supcon_embeddings.npz for in-process
kNN at inference time.

This sidesteps the DB schema change (no new vector column, no new
RPC) while still letting us A/B SupCon vs original CLIP. The file
holds:
  sha256:     (N,) string array
  embedding:  (N, 512) float32, L2-normalized in projected space
  hse:        (N,) string array
  loc:        (N,) string array
  source:     (N,) string array (label_source)

Loaded by src/zero_shot.py:_retrieve_similar_labels when env
SUPCON_RAG=1.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from supabase import create_client

REPO = Path(__file__).resolve().parents[1]
# Head ships committed in src/ (~1MB); fall back to tmp/ for local
# dev where the trainer wrote it.
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--head", type=str, default="")
_ap.add_argument("--out",  type=str, default="")
_args, _ = _ap.parse_known_args()

if _args.head:
    HEAD_PATH = Path(_args.head).expanduser().resolve()
else:
    _HEAD_SRC = REPO / "src" / "clip_supcon_head.pt"
    _HEAD_TMP = REPO / "tmp" / "clip_supcon_head.pt"
    HEAD_PATH = _HEAD_SRC if _HEAD_SRC.exists() else _HEAD_TMP

if _args.out:
    OUT_PATH = Path(_args.out).expanduser().resolve()
else:
    OUT_PATH = REPO / "tmp" / "clip_supcon_embeddings.npz"


class ProjHead(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return F.normalize(x + self.fc2(h), dim=-1)


def parse_vec(v) -> np.ndarray:
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    if isinstance(v, str):
        try:
            return np.array(json.loads(v), dtype=np.float32)
        except Exception:  # noqa: BLE001
            return np.array([float(x) for x in v.strip("[]").split(",")],
                            dtype=np.float32)
    return np.zeros(512, dtype=np.float32)


def main() -> int:
    db = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    print("loading all photo_embeddings…")
    rows: list[dict] = []
    off = 0
    while True:
        page = (db.table("photo_embeddings")
            .select("sha256, hse_type_slug, location_slug, label_source, "
                    "is_holdout, embedding")
            .range(off, off + 999).execute().data or [])
        if not page: break
        rows.extend(page)
        if len(page) < 1000: break
        off += 1000
    print(f"  loaded {len(rows)} rows")

    # Parse + normalize
    X = np.stack([parse_vec(r["embedding"]) for r in rows]).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    # Load head
    ckpt = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
    head = ProjHead(dim=ckpt["input_dim"], hidden=ckpt["hidden_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    # Project in batches (CPU is fine for one-shot)
    print("projecting…")
    out = []
    BATCH = 256
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            chunk = torch.from_numpy(X[i:i+BATCH]).float()
            out.append(head(chunk).numpy())
    X_proj = np.concatenate(out, axis=0).astype(np.float32)
    print(f"  done: shape={X_proj.shape}")

    np.savez_compressed(
        OUT_PATH,
        sha256=np.array([r["sha256"] for r in rows]),
        embedding=X_proj,
        hse=np.array([(r.get("hse_type_slug") or "") for r in rows]),
        loc=np.array([(r.get("location_slug") or "") for r in rows]),
        source=np.array([(r.get("label_source") or "") for r in rows]),
        is_holdout=np.array([bool(r.get("is_holdout")) for r in rows]),
    )
    print(f"saved: {OUT_PATH} ({OUT_PATH.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
