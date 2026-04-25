"""Fine-tune a small adapter on top of frozen CLIP ViT-B/32.

Input: all pgvector rows (1,108 labeled photos from SVN+MJNT currently).
Output: src/clip_adapter.pt (~600 KB) — a small MLP that maps CLIP embeddings
to a construction-safety-violation-aware space where same-class photos
cluster tightly regardless of project domain.

Loss: Supervised contrastive (Khosla 2020). For each anchor photo, pull
photos with the same hse_type_slug closer, push different-class photos
away. Normalized embeddings + temperature-scaled cosine similarity.

Training splits:
  - 90% train / 10% validation (stratified by class)
  - Balanced class sampler — each batch has examples from as many classes
    as possible to make the contrast informative.

Validation metric: k=5 retrieval accuracy on held-out set (fraction of
queries whose majority-vote among top-5 neighbours matches the ground
truth class).

Usage:
  python scripts/finetune_clip.py
  python scripts/finetune_clip.py --epochs 200 --lr 5e-4 --tau 0.07
  python scripts/finetune_clip.py --dry-run     # download data + exit
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("finetune")
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass


# ---------- data ----------

def _fetch_pgvector_embeddings() -> tuple[np.ndarray, list[str]]:
    """Fetch all pgvector rows: return (N, 512) embeddings and list of N hse labels.
    Only returns rows whose hse_type_slug is in the current taxonomy."""
    import json
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    db = create_client(url, key)
    tax = json.loads((REPO_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    valid = {h["slug"] for h in tax["hse_types"]}

    rows: list[dict] = []
    offset = 0
    while True:
        batch = (
            db.table("photo_embeddings")
              .select("sha256, hse_type_slug, embedding")
              .range(offset, offset + 499).execute().data or []
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 500:
            break
        offset += 500

    keep = [r for r in rows if r.get("hse_type_slug") in valid and r.get("embedding")]
    if not keep:
        raise RuntimeError("No usable pgvector rows found.")

    # embedding column is either a list of floats or a string repr — handle both
    def _parse(e):
        if isinstance(e, (list, tuple)):
            return np.asarray(e, dtype=np.float32)
        if isinstance(e, str):
            # "[0.1,0.2,...]" format
            vec = [float(x) for x in e.strip("[]").split(",")]
            return np.asarray(vec, dtype=np.float32)
        raise TypeError(f"unknown embedding type: {type(e)}")

    embs = np.stack([_parse(r["embedding"]) for r in keep], axis=0)
    labels = [r["hse_type_slug"] for r in keep]
    # Safety: re-normalize (should already be but be defensive)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
    log.info("Fetched %d usable pgvector rows across %d classes",
             len(keep), len(set(labels)))
    return embs.astype(np.float32), labels


# ---------- training ----------

def _supcon_loss(features, labels, tau: float = 0.1):
    """Supervised contrastive loss.
    features: (B, D) L2-normalized. labels: (B,) int64.
    Returns scalar loss (lower = tighter same-class, looser different-class)."""
    import torch
    sim = features @ features.T / tau  # (B, B)
    B = features.size(0)
    mask_diag = torch.eye(B, dtype=torch.bool, device=features.device)

    # Subtract per-row max for numerical stability (doesn't change softmax),
    # then exp. Zero out the diagonal so self-similarity contributes nothing.
    sim_stable = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim_stable) * (~mask_diag).float()  # (B, B), diagonal=0
    # log-softmax denominator
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))
    log_prob = sim_stable - log_denom  # finite everywhere

    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row) & ~mask_diag
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, requires_grad=True, device=features.device)
    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
    mean_pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_count
    loss = -mean_pos_log_prob[has_pos].mean()
    return loss


def _balanced_batch(emb: np.ndarray, labels_int: np.ndarray,
                    batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample one batch with roughly balanced class representation.
    Takes min(available, ceil(batch/n_classes)) from each class, fills remainder randomly."""
    classes = np.unique(labels_int)
    per_class = max(2, batch_size // len(classes))
    indices: list[int] = []
    for c in classes:
        where = np.where(labels_int == c)[0]
        if len(where) == 0:
            continue
        take = min(per_class, len(where))
        picked = rng.choice(where, size=take, replace=False)
        indices.extend(picked.tolist())
    # Fill to batch_size with random
    if len(indices) < batch_size:
        extra = rng.choice(len(labels_int), size=batch_size - len(indices), replace=True)
        indices.extend(extra.tolist())
    else:
        indices = indices[:batch_size]
    indices = np.array(indices)
    rng.shuffle(indices)
    return emb[indices], labels_int[indices]


def _knn_accuracy(query_emb: np.ndarray, query_labels: np.ndarray,
                  index_emb: np.ndarray, index_labels: np.ndarray, k: int = 5) -> float:
    """Leave-one-out k-NN with majority voting. Returns top-1 retrieval accuracy."""
    if len(query_emb) == 0:
        return 0.0
    sim = query_emb @ index_emb.T  # (Q, N)
    topk_idx = np.argsort(-sim, axis=1)[:, :k]  # (Q, k)
    votes = index_labels[topk_idx]  # (Q, k)
    # Majority vote
    correct = 0
    for i in range(len(query_labels)):
        counts = Counter(votes[i].tolist())
        pred = counts.most_common(1)[0][0]
        if pred == query_labels[i]:
            correct += 1
    return correct / len(query_labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1,
                    help="SupCon temperature (lower = tighter clusters)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop patience in epochs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    from src.clip_adapter import CLIPAdapter, ADAPTER_PATH

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Fetch data
    emb, labels_str = _fetch_pgvector_embeddings()
    n_total = len(emb)
    label_list = sorted(set(labels_str))
    lbl2int = {s: i for i, s in enumerate(label_list)}
    labels_int = np.array([lbl2int[s] for s in labels_str], dtype=np.int64)
    log.info("Classes (%d):", len(label_list))
    class_counts = Counter(labels_str)
    for s, n in class_counts.most_common():
        log.info("  %4d  %s", n, s)

    if args.dry_run:
        log.info("dry-run: data fetched, exiting.")
        return 0

    # Stratified train/val split
    train_idx, val_idx = [], []
    for c in np.unique(labels_int):
        where = np.where(labels_int == c)[0]
        rng.shuffle(where)
        n_val = max(1, int(round(len(where) * args.val_frac)))
        val_idx.extend(where[:n_val].tolist())
        train_idx.extend(where[n_val:].tolist())
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    log.info("Train: %d  Val: %d", len(train_idx), len(val_idx))

    X_train = emb[train_idx]
    y_train = labels_int[train_idx]
    X_val = emb[val_idx]
    y_val = labels_int[val_idx]

    # Baseline: k-NN on raw CLIP (no adapter) as sanity check
    baseline = _knn_accuracy(X_val, y_val, X_train, y_train, k=5)
    log.info("Baseline k=5 retrieval acc on val (no adapter): %.3f", baseline)

    # Train
    device = "cpu"
    model = CLIPAdapter().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = baseline
    best_state = None
    bad_epochs = 0

    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Balanced batch
        bx, by = _balanced_batch(X_train, y_train, args.batch_size, rng)
        bx_t = torch.from_numpy(bx).to(device)
        by_t = torch.from_numpy(by).to(device)

        opt.zero_grad()
        out = model(bx_t)
        loss = _supcon_loss(out, by_t, tau=args.tau)
        loss.backward()
        opt.step()

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr_emb = model(X_train_t).cpu().numpy()
                val_emb = model(X_val_t).cpu().numpy()
            val_acc = _knn_accuracy(val_emb, y_val, tr_emb, y_train, k=5)
            marker = ""
            if val_acc > best_val + 1e-4:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                marker = "  *"
            else:
                bad_epochs += 1
            log.info("[%3d] loss=%.4f  val_acc=%.3f%s", epoch, loss.item(), val_acc, marker)
            if bad_epochs >= args.patience:
                log.info("Early stop: no improvement for %d checkpoints", args.patience)
                break

    # Save best
    if best_state is None:
        log.warning("No improvement over baseline — saving current state anyway")
        best_state = model.state_dict()
    torch.save(best_state, ADAPTER_PATH)
    log.info("Best val k=5 acc: %.3f  (baseline %.3f, +%.3f)",
             best_val, baseline, best_val - baseline)
    log.info("Adapter saved to %s", ADAPTER_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
