"""Investigate why cross-domain scrapes hurt accuracy even when DTag-labeled.

Samples N photos from each project, CLIP-embeds them, and measures pairwise
cosine similarity. If cross-project distance is similar to or smaller than
within-project distance, the domain-shift hypothesis is confirmed — cross-
domain photos would surface as nearest neighbours for eval queries and
carry incorrect labels (because the violation in a university classroom may
look similar in CLIP space to an unrelated industrial scene).

Also analyzes: for a simulated "add RUVN" scenario, how many of the top-5
nearest neighbours for SVN/MJNT eval photos would be RUVN? If ≥ 40%, the
RUVN labels dominate retrieval.

Usage:
  python scripts/investigate_domain_shift.py --samples 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("domain")
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _sample_paths(root: Path, project_code: str, n: int, rng: random.Random) -> list[Path]:
    """Get N random photo paths for one project from on-disk metadata."""
    pool: list[Path] = []
    for m in root.glob("*/*/metadata.json"):
        try: d = json.loads(m.read_text(encoding="utf-8"))
        except Exception: continue
        if d.get("project_code") != project_code: continue
        for ph in (d.get("photos") or []):
            p = m.parent / (ph.get("file") or "")
            if p.exists(): pool.append(p)
    rng.shuffle(pool)
    return pool[:n]


def _mean_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity across all pairs between two embedding arrays.
    a: (n, d), b: (m, d), L2-normalized. Returns scalar in [-1, 1]."""
    return float((a @ b.T).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = DEFAULT_ROOT
    projects = ["SVN", "MJNT", "SLPXA", "AR", "RUVN", "H9"]
    samples: dict[str, list[Path]] = {}
    for p in projects:
        paths = _sample_paths(root, p, args.samples, rng)
        if not paths:
            log.info("no photos for %s (skipping)", p)
            continue
        samples[p] = paths
        log.info("sampled %d photos from %s", len(paths), p)

    # Embed
    from src.embeddings import embed_images
    embeddings: dict[str, np.ndarray] = {}
    for p, paths in samples.items():
        log.info("embedding %s...", p)
        vecs = embed_images(paths)
        # drop zero rows (failed reads)
        mask = np.any(vecs != 0, axis=1)
        embeddings[p] = vecs[mask]
        log.info("  %d valid embeddings for %s", mask.sum(), p)

    # Pairwise mean cosine similarity
    log.info("")
    log.info("=" * 70)
    log.info("Pairwise mean cosine similarity (higher = closer in CLIP space)")
    log.info("=" * 70)
    keys = list(embeddings.keys())
    header = "       " + " ".join(f"{k:>7}" for k in keys)
    log.info(header)
    for i, ki in enumerate(keys):
        row = [f"{ki:<7}"]
        for kj in keys:
            sim = _mean_sim(embeddings[ki], embeddings[kj])
            row.append(f"{sim:>7.3f}")
        log.info(" ".join(row))

    # For each project, mean within-project vs. mean cross-project distance
    log.info("")
    log.info("=" * 70)
    log.info("Within-project vs cross-project cohesion")
    log.info("=" * 70)
    log.info(f"{'project':<10} {'within':>8} {'cross(avg)':>12} {'shift':>8}")
    for ki in keys:
        within = _mean_sim(embeddings[ki], embeddings[ki])
        crosses = [_mean_sim(embeddings[ki], embeddings[kj])
                   for kj in keys if kj != ki]
        cross_avg = float(np.mean(crosses))
        shift = within - cross_avg
        log.info(f"{ki:<10} {within:>8.3f} {cross_avg:>12.3f} {shift:>+8.3f}")

    # Retrieval simulation: if we added RUVN/H9/AR to pgvector (which is
    # currently SVN+MJNT), what fraction of top-5 neighbours for a typical
    # SVN+MJNT eval query would be cross-domain?
    log.info("")
    log.info("=" * 70)
    log.info("Simulated top-5 neighbour domains for eval queries")
    log.info("=" * 70)
    # Merge SVN+MJNT as the current pgvector reference (query photos are
    # drawn from here too in the eval script — we need the domain source
    # to tell us "would RUVN/H9/AR neighbours DISPLACE SVN/MJNT ones?")
    for injecting in ["RUVN", "H9", "AR", "SLPXA"]:
        if injecting not in embeddings: continue
        # For each eval-candidate query (SVN+MJNT), compute nearest 5 among
        # the union {SVN, MJNT, injecting}, tag each neighbour with its
        # source project, and count how many are the injected one.
        index_keys = []
        index_vecs_list = []
        for k in ["SVN", "MJNT", injecting]:
            if k in embeddings:
                for v in embeddings[k]:
                    index_keys.append(k)
                    index_vecs_list.append(v)
        index_vecs = np.stack(index_vecs_list, axis=0)

        injected_counts = []
        for query_proj in ["SVN", "MJNT"]:
            if query_proj not in embeddings: continue
            for qv in embeddings[query_proj]:
                sims = index_vecs @ qv
                # exclude the query itself (cos=1)
                top5_idx = np.argsort(-sims)[1:6]
                src = [index_keys[i] for i in top5_idx]
                injected_counts.append(sum(1 for s in src if s == injecting))
        if injected_counts:
            avg_injected = np.mean(injected_counts)
            pct_dom = np.mean([c >= 2 for c in injected_counts]) * 100
            log.info(f"  inject {injecting:<6}  avg {avg_injected:.2f}/5 "
                     f"cross-domain neighbours, {pct_dom:.0f}% of queries "
                     f"get ≥2 from {injecting}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
