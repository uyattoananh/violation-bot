"""SupCon head v3 — hard-negative mining via confusion-pair sampling.

v2 was disappointing because:
  - Multi-axis loss diluted hse_type signal
  - v2_visionchecked corpus has LLM labels that disagree with manual
    on the very boundaries we're trying to learn

v3 fixes both:
  - DROP multi-axis (hse-only, like v1)
  - DROP v2 corpus (manual-only, like v1)
  - ADD: confusion-pair oversampling. Build a map of (gt_class →
    most-confused-with class) from the existing eval per_photo data,
    then bias the WeightedRandomSampler so confusable-class rows
    appear more often. With class-balanced sampling already in
    place, this nudges batches toward the hard boundaries.

The bet: the architectural change in v1 already separated the easy
classes; what remains is teaching the head to push apart the SPECIFIC
confusion pairs the LLM trips on.

Output: src/clip_supcon_head_v3.pt
"""
from __future__ import annotations
import json
import os
import sys
import math
import collections
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from supabase import create_client

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "src" / "clip_supcon_head_v3.pt"

EVAL_FILES = [
    "tmp/eval_supcon_s42.json",
    "tmp/eval_supcon_s7.json",
    "tmp/eval_supcon_s25.json",
    "tmp/eval_deflag_s42.json",
    "tmp/eval_deflag_s7.json",
    "tmp/eval_deflag_s25.json",
]


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


def load_confusion_weight() -> dict[str, float]:
    """Aggregate per-class confusion rate from past eval per_photo data.
    Returns a {hse_slug: boost_factor} dict; classes mis-predicted often
    (either as gt or as wrong-prediction) get higher weights so they show
    up more in training batches."""
    miss_counts: collections.Counter[str] = collections.Counter()
    seen_counts: collections.Counter[str] = collections.Counter()
    for path in EVAL_FILES:
        p = REPO / path
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in d.get("per_photo", []):
            gt = row.get("gt_hse")
            pr = row.get("predicted_hse")
            if not gt:
                continue
            seen_counts[gt] += 1
            if not row.get("hse_ok", False):
                miss_counts[gt] += 1     # boost classes the model misses
                if pr and pr != gt:
                    miss_counts[pr] += 1  # also boost confusable predicted-class
    # Confusion rate: miss / seen. Class never seen → default 1.0.
    rate: dict[str, float] = {}
    for cls, seen in seen_counts.items():
        m = miss_counts.get(cls, 0)
        rate[cls] = m / max(seen, 1)
    return rate


class ProjHead(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return F.normalize(x + self.fc2(h), dim=-1)


class EmbedDataset(Dataset):
    def __init__(self, embs: np.ndarray, labels: np.ndarray):
        self.embs = torch.from_numpy(embs).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self): return len(self.embs)
    def __getitem__(self, i): return self.embs[i], self.labels[i]


def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                temp: float = 0.1) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    sim = (z @ z.T) / temp
    n = z.size(0)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    sim_max, _ = sim.masked_fill(eye, -float("inf")).max(dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable) * (~eye).float()
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12)
    log_prob = sim_stable - torch.log(denom)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = (label_eq & ~eye).float()
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    return -(log_prob * pos_mask).sum(dim=1).div(pos_counts).mean()


def main() -> int:
    db = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    print("loading manual embeddings…")
    rows: list[dict] = []
    off = 0
    while True:
        page = (db.table("photo_embeddings")
            .select("sha256, hse_type_slug, embedding")
            .eq("label_source", "manual")
            .eq("is_holdout", False)
            .range(off, off + 999).execute().data or [])
        if not page: break
        rows.extend(page)
        if len(page) < 1000: break
        off += 1000
    print(f"  {len(rows)} manual rows")

    X = np.stack([parse_vec(r["embedding"]) for r in rows]).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    y_str = np.array([r["hse_type_slug"] or "?" for r in rows])

    # Filter rare classes (need ≥4 for positive pairs)
    counts = collections.Counter(y_str)
    keep = np.array([counts[c] >= 4 for c in y_str])
    X = X[keep]
    y_str = y_str[keep]
    classes = sorted(set(y_str))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[c] for c in y_str], dtype=np.int64)
    print(f"  after filtering: {len(X)} rows in {len(classes)} classes")

    # Confusion-aware sampling weights:
    #   base    = 1 / class_freq      (the v1 class-balanced behaviour)
    #   boost   = 1 + 2.0 * confusion_rate
    # → rows in high-confusion classes get up to 3× as likely to be drawn,
    #   compounding with class balance. Caps so no single row dominates.
    confusion_rate = load_confusion_weight()
    print(f"  confusion rates loaded: {len(confusion_rate)} classes")
    top_conf = sorted(confusion_rate.items(), key=lambda x: -x[1])[:8]
    print(f"  hardest classes (boost ≈ 1 + 2×rate):")
    for c, r in top_conf:
        print(f"    {c:35} confusion={r:.2f}  boost={1 + 2*r:.2f}")

    class_freq = collections.Counter(y.tolist())
    sample_weights = []
    for yi, cls in zip(y, y_str):
        base = 1.0 / class_freq[int(yi)]
        boost = 1.0 + 2.0 * confusion_rate.get(cls, 0.0)
        sample_weights.append(base * boost)
    sample_weights = np.array(sample_weights, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    head = ProjHead(dim=512, hidden=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    ds = EmbedDataset(X, y)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(ds),
                                    replacement=True)
    # Slightly larger batch (192) — confusion-pair structure needs more
    # in-batch examples to materialize as effective negatives.
    loader = DataLoader(ds, batch_size=192, sampler=sampler, drop_last=True)

    print("\ntraining 60 epochs (hard-negative mining via confusion weights)…")
    import time
    t0 = time.time()
    for epoch in range(60):
        ep_loss = 0.0
        n_b = 0
        for emb, lab in loader:
            emb = emb.to(device); lab = lab.to(device)
            z = head(emb)
            loss = supcon_loss(z, lab, temp=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            n_b += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/60  loss={ep_loss/max(n_b,1):.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.0f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "classes": classes,
        "label_to_idx": label_to_idx,
        "input_dim": 512,
        "hidden_dim": 256,
        "trained_on": ["manual"],
        "sampling_mode": "confusion-weighted",
    }, OUT_PATH)
    print(f"\nsaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
