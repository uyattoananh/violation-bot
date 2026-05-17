"""SupCon head v5 — train on manual + ALL rare variants.

v4 trained on (manual + curated_rare_v1 first-128 only) and was flat
vs v1 at 5-seed N=100. v5 includes ALL rare-class supervised data
the project has accumulated:

  manual                          ~2,966 rows  (is_holdout=False only)
  aecis_curated_rare_v1               485 rows (regardless of is_holdout —
                                                we want their labels for
                                                training even if they're
                                                pool-suppressed at retrieval)
  aecis_curated_rare_v2_nbr           101 rows
  Total                            ~3,552 rows in ~27-29 classes

The is_holdout flag was set on 458 rare rows by Plan E (2026-05-17)
to suppress them from KNN retrieval — but their labels are still
valid training data. v5 lets SupCon see all 587 rare-class
supervised pairs, even though only 128 remain active at retrieval.

Hypothesis: the projection head benefits from MORE rare-class
supervision regardless of whether those rows participate in kNN
candidate selection. The QUERY embedding gets mapped to a better
space, and the (smaller) trusted candidate pool still wins kNN.

Output: src/clip_supcon_head_v5.pt
"""
from __future__ import annotations
import json
import os
import sys
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
OUT_PATH = REPO / "src" / "clip_supcon_head_v5.pt"


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

    print("loading training corpus:")
    rows: list[dict] = []
    per_source = collections.Counter()

    # manual: skip is_holdout (eval leak guards + the 698 HK suppressions —
    # wait, those were re-deflagged on 2026-05-16, so manual is_holdout=FALSE
    # is the full ~2,966 rows).
    off = 0
    while True:
        page = (db.table("photo_embeddings")
            .select("sha256, hse_type_slug, embedding")
            .eq("label_source", "manual")
            .eq("is_holdout", False)
            .range(off, off + 999).execute().data or [])
        if not page: break
        rows.extend(page)
        per_source["manual"] += len(page)
        if len(page) < 1000: break
        off += 1000

    # rare_v1 + rare_v2_nbr: TAKE ALL (regardless of is_holdout). Plan E
    # flagged some as out-of-pool for retrieval, but their labels are
    # still valid supervised pairs.
    for ls in ["aecis_curated_rare_v1", "aecis_curated_rare_v2_nbr"]:
        off = 0
        while True:
            page = (db.table("photo_embeddings")
                .select("sha256, hse_type_slug, embedding")
                .eq("label_source", ls)
                .range(off, off + 999).execute().data or [])
            if not page: break
            rows.extend(page)
            per_source[ls] += len(page)
            if len(page) < 1000: break
            off += 1000

    print(f"  per source: {dict(per_source)}")
    print(f"  total: {len(rows)}")

    X = np.stack([parse_vec(r["embedding"]) for r in rows]).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    y_str = np.array([r["hse_type_slug"] or "?" for r in rows])

    counts = collections.Counter(y_str)
    keep = np.array([counts[c] >= 4 for c in y_str])
    X = X[keep]
    y_str = y_str[keep]
    classes = sorted(set(y_str))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[c] for c in y_str], dtype=np.int64)
    print(f"  after filtering <4-per-class: {len(X)} rows in {len(classes)} classes")

    print("\nper-class counts in v5 training pool:")
    for cls, n in collections.Counter(y_str).most_common():
        bar = "#" * (n // 20)
        print(f"  {cls:35} {n:4}  {bar}")

    class_freq = collections.Counter(y.tolist())
    sample_w = np.array([1.0 / class_freq[int(yi)] for yi in y], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  device: {device}")

    head = ProjHead(dim=512, hidden=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

    ds = EmbedDataset(X, y)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=128, sampler=sampler, drop_last=True)

    print("\ntraining 50 epochs (single-axis SupCon, full rare corpus)…")
    import time
    t0 = time.time()
    for epoch in range(50):
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/50  loss={ep_loss/max(n_b,1):.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.0f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "classes": classes,
        "label_to_idx": label_to_idx,
        "input_dim": 512,
        "hidden_dim": 256,
        "trained_on": ["manual", "aecis_curated_rare_v1", "aecis_curated_rare_v2_nbr"],
        "per_source": dict(per_source),
    }, OUT_PATH)
    print(f"\nsaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
