"""SupCon head v2 — three upgrades over v1 (`train_supcon_head.py`):

  1. EXPANDED CORPUS: manual + aecis_labelled_v2_visionchecked rows
     (the v2 rows already passed both text-classify and vision-verify
     gates, so they're at least as reliable as the AECIS-labelled
     seeds we're targeting). Roughly doubles the supervision pool.

  2. MULTI-AXIS LOSS: adds a location_slug SupCon term with weight
     0.3. The shared 512-d embedding now has to separate BOTH axes
     simultaneously — auxiliary signal that should help location
     retrieval without disturbing the hse_type structure too much.

  3. TEMPERATURE ANNEAL: cosine schedule from 0.3 (broad pull) at
     epoch 0 down to 0.05 (sharp discriminator) at the end. Standard
     in modern contrastive papers.

Output: src/clip_supcon_head_v2.pt — same shape as v1 so the
projection script + zero_shot.py can load it interchangeably.
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
OUT_PATH = REPO / "src" / "clip_supcon_head_v2.pt"


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
    """Same architecture as v1 so weights are swap-compatible."""
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return F.normalize(x + self.fc2(h), dim=-1)


class MultiAxisDataset(Dataset):
    def __init__(self, embs: np.ndarray, hse: np.ndarray, loc: np.ndarray):
        self.embs = torch.from_numpy(embs).float()
        self.hse = torch.from_numpy(hse).long()
        self.loc = torch.from_numpy(loc).long()

    def __len__(self): return len(self.embs)
    def __getitem__(self, i): return self.embs[i], self.hse[i], self.loc[i]


def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                temp: float) -> torch.Tensor:
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

    print("loading manual + v2_visionchecked embeddings…")
    rows: list[dict] = []
    for label_source in ["manual", "aecis_labelled_v2_visionchecked"]:
        off = 0
        n_added = 0
        while True:
            page = (db.table("photo_embeddings")
                .select("sha256, hse_type_slug, location_slug, embedding")
                .eq("label_source", label_source)
                .eq("is_holdout", False)
                .range(off, off + 999).execute().data or [])
            if not page: break
            rows.extend(page)
            n_added += len(page)
            if len(page) < 1000: break
            off += 1000
        print(f"  {label_source}: +{n_added} rows")
    print(f"  total: {len(rows)} rows")

    X = np.stack([parse_vec(r["embedding"]) for r in rows]).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    y_hse = np.array([r.get("hse_type_slug") or "?" for r in rows])
    y_loc = np.array([r.get("location_slug") or "?" for r in rows])

    # Filter rare hse classes (need ≥4 for positive pairs in batch)
    hse_counts = collections.Counter(y_hse)
    keep = np.array([hse_counts[c] >= 4 for c in y_hse])
    X = X[keep]
    y_hse = y_hse[keep]
    y_loc = y_loc[keep]

    hse_classes = sorted(set(y_hse))
    loc_classes = sorted(set(y_loc))
    hse_to_idx = {c: i for i, c in enumerate(hse_classes)}
    loc_to_idx = {c: i for i, c in enumerate(loc_classes)}
    y_hse_idx = np.array([hse_to_idx[c] for c in y_hse], dtype=np.int64)
    y_loc_idx = np.array([loc_to_idx[c] for c in y_loc], dtype=np.int64)
    print(f"  after filtering: {len(X)} rows, "
          f"{len(hse_classes)} hse_classes, {len(loc_classes)} loc_classes")

    # Class-balanced sampler on hse_type (the harder axis to separate).
    class_freq = collections.Counter(y_hse_idx.tolist())
    sample_w = np.array([1.0 / class_freq[int(y)] for y in y_hse_idx],
                        dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    head = ProjHead(dim=512, hidden=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    n_epochs = 80
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    ds = MultiAxisDataset(X, y_hse_idx, y_loc_idx)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(ds),
                                    replacement=True)
    loader = DataLoader(ds, batch_size=128, sampler=sampler, drop_last=True)

    # Temperature cosine schedule (broad → sharp)
    temp_start, temp_end = 0.3, 0.05

    print(f"\ntraining {n_epochs} epochs (multi-axis SupCon + temp anneal)…")
    import time
    t0 = time.time()
    for epoch in range(n_epochs):
        # Cosine anneal temperature
        progress = epoch / max(n_epochs - 1, 1)
        temp = temp_end + 0.5 * (temp_start - temp_end) * (1 + math.cos(math.pi * progress))
        ep_loss_h = ep_loss_l = 0.0
        n_batches = 0
        for emb, hse, loc in loader:
            emb = emb.to(device); hse = hse.to(device); loc = loc.to(device)
            z = head(emb)
            loss_h = supcon_loss(z, hse, temp=temp)
            loss_l = supcon_loss(z, loc, temp=temp)
            loss = loss_h + 0.3 * loss_l
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss_h += float(loss_h)
            ep_loss_l += float(loss_l)
            n_batches += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  temp={temp:.3f}  "
                  f"loss_hse={ep_loss_h/max(n_batches,1):.3f}  "
                  f"loss_loc={ep_loss_l/max(n_batches,1):.3f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.0f}s")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "hse_classes": hse_classes,
        "loc_classes": loc_classes,
        "hse_to_idx": hse_to_idx,
        "loc_to_idx": loc_to_idx,
        "input_dim": 512,
        "hidden_dim": 256,
        "trained_on": ["manual", "aecis_labelled_v2_visionchecked"],
        "n_rows": len(X),
    }, OUT_PATH)
    print(f"\nsaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
