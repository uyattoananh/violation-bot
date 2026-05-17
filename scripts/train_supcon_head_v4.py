"""SupCon head v4 — train on manual + aecis_curated_rare_v1.

Diagnoses why v2/v3 didn't help: the head was trained only on
manual rows, so any new k-NN candidates (rare-mine, v2_visionchecked)
get projected through a head that has never seen them as
supervised examples. They land in poorly-fit regions of the
projection space.

v4 fix: include the LLM-curated rare-class rows in training. These
went through TARGETED per-class vision-verify (much stricter than
the multi-class v2 pipeline) — quality is higher than v2_visionchecked.

Architecture + loss identical to v1 (single-axis SupCon, frozen
CLIP, 512→256→512 head). Only the training corpus differs:
v1: manual                              (3,015 rows, 27 classes)
v4: manual + aecis_curated_rare_v1      (~3,500 rows, ~29 classes)

Output: src/clip_supcon_head_v4.pt
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
OUT_PATH = REPO / "src" / "clip_supcon_head_v4.pt"
LABEL_SOURCES = ["manual", "aecis_curated_rare_v1"]


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

    print(f"loading embeddings from: {LABEL_SOURCES}")
    rows: list[dict] = []
    per_source = collections.Counter()
    for ls in LABEL_SOURCES:
        off = 0
        n_pulled = 0
        while True:
            page = (db.table("photo_embeddings")
                .select("sha256, hse_type_slug, embedding")
                .eq("label_source", ls)
                .eq("is_holdout", False)
                .range(off, off + 999).execute().data or [])
            if not page: break
            rows.extend(page)
            n_pulled += len(page)
            if len(page) < 1000: break
            off += 1000
        per_source[ls] = n_pulled
        print(f"  {ls}: {n_pulled}")
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
    print(f"  after filtering rare (<4 per class): {len(X)} rows in {len(classes)} classes")

    print("\nper-class counts in training pool:")
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

    print("\ntraining 50 epochs (single-axis SupCon, manual + rare_v1 corpus)…")
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
        "trained_on": LABEL_SOURCES,
        "per_source": dict(per_source),
    }, OUT_PATH)
    print(f"\nsaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
