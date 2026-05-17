"""Approach A: visual neighbour expansion from manual exemplars.

For each rare HSE class:
  1. Pull manual photos labelled with that class (3-10 exemplars).
  2. Find unembedded manifest photos whose text mentions the class
     keywords (existing rare-mine candidate pool).
  3. Compute CLIP embedding for those candidates.
  4. Rank by MIN cosine-distance to any manual exemplar — the
     candidates that look most like a confirmed example.
  5. Run targeted vision-verify on the TOP-K nearest.
  6. Embed confirmed ones as label_source='aecis_curated_rare_v2_nbr'.

Different from mine_rare_class_seeds.py which checks candidates in
manifest order. This script ranks candidates by visual similarity
to a known good example first, then vision-verifies only the most
promising ones. Much higher hit rate per vision call for classes
where keyword filter is noisy (Welding, Ladder, Truck_vehicle).

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/mine_rare_via_visual_neighbours.py \\
        --class Welding_unsafe --top-k 50
    ./.venv-webapp/Scripts/python.exe scripts/mine_rare_via_visual_neighbours.py \\
        --all --top-k 30
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from supabase import create_client
from openai import OpenAI

# Reuse the keyword + class-description bank from the prior script
from scripts.mine_rare_class_seeds import (
    RARE_CLASSES, CLASS_DESCRIPTIONS, vision_verify, _safe, _sha256,
)

LABEL_SOURCE = "aecis_curated_rare_v2_nbr"
MANIFEST_PATH = REPO / "Issue_Gen" / "photos" / "manifest.jsonl"
PHOTOS_ROOT = REPO / "Issue_Gen" / "photos"
RARE_CACHE_PATH = REPO / "scripts" / ".rare_mine_cache.json"


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


def load_manual_exemplars(db, slug: str) -> np.ndarray | None:
    """Return (M, 512) L2-normalized exemplar embeddings for the class."""
    rows = (db.table("photo_embeddings")
            .select("embedding")
            .eq("label_source", "manual")
            .eq("hse_type_slug", slug)
            .eq("is_holdout", False)
            .execute().data or [])
    if not rows:
        return None
    X = np.stack([parse_vec(r["embedding"]) for r in rows]).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return X


def find_candidates_for_class(slug: str, exclude_shas: set[str],
                               exclude_pair_cache: set[str]) -> list[dict]:
    """Manifest entries whose text matches class keywords AND aren't
    already in DB (by sha later checked) AND haven't been vision-
    rejected for this slug in a prior run."""
    keywords = [k.lower() for k in RARE_CLASSES.get(slug, [])]
    seen_fp: set[str] = set()
    out: list[dict] = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except: continue
            fp = rec.get("filepath", "")
            if not fp or fp in seen_fp: continue
            seen_fp.add(fp)
            text = ((rec.get("issue_name", "") or "") + " " +
                    (rec.get("description", "") or "")).lower()
            if not any(kw in text for kw in keywords):
                continue
            local = PHOTOS_ROOT / _safe(fp)
            if not local.exists() or local.stat().st_size < 1024:
                continue
            out.append({**rec, "_local": str(local)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", type=str, default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top-k", type=int, default=30,
                    help="how many visually-closest candidates to vision-verify per class")
    ap.add_argument("--max-clip-embeds", type=int, default=400,
                    help="cap CLIP-embed work per class (cost control)")
    ap.add_argument("--min-vision-confidence", type=float, default=0.55)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.cls:
        ap.error("specify --class <slug> or --all")
    targets = list(RARE_CLASSES.keys()) if args.all else [args.cls]

    db = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write("OPENROUTER_API_KEY not set\n")
        return 2
    or_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # tenant
    try:
        tenant = (db.table("tenants").select("id")
                  .eq("name", "Public Demo").limit(1).execute().data or [])
        tenant_id = tenant[0]["id"] if tenant else None
    except Exception:  # noqa: BLE001
        tenant_id = None

    # vision-verify cache (sha+slug keyed) — skip already-judged pairs
    rare_cache: dict[str, dict] = {}
    if RARE_CACHE_PATH.exists():
        try:
            rare_cache = json.loads(RARE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rare_cache = {}

    # All shas currently in DB (any label_source) — exclude from re-embedding
    existing_shas: set[str] = set()
    off = 0
    while True:
        page = (db.table("photo_embeddings").select("sha256")
                .range(off, off + 999).execute().data or [])
        if not page: break
        for r in page:
            if r.get("sha256"):
                existing_shas.add(r["sha256"])
        if len(page) < 1000: break
        off += 1000
    print(f"loaded {len(existing_shas)} existing DB shas")

    # Lazy CLIP load on first use
    embed_image = None
    grand = {"candidates": 0, "embedded_clip": 0, "vision_checked": 0,
             "vision_confirmed": 0, "embedded_final": 0}

    for slug in targets:
        print(f"\n=== {slug} ===")
        exemplars = load_manual_exemplars(db, slug)
        if exemplars is None or len(exemplars) < 1:
            print(f"  no manual exemplars — skipping")
            continue
        print(f"  manual exemplars: {len(exemplars)}")

        candidates = find_candidates_for_class(slug, existing_shas, set())
        # Filter out already-judged-by-prior-rare-mine for this slug
        new_cands = []
        for rec in candidates:
            sha = _sha256(Path(rec["_local"]))
            if sha in existing_shas:
                continue
            if rare_cache.get(f"{sha}|{slug}") is not None:
                continue   # already vision-judged
            rec["_sha"] = sha
            new_cands.append(rec)
        # Cap CLIP-embed work
        if len(new_cands) > args.max_clip_embeds:
            new_cands = new_cands[: args.max_clip_embeds]
        print(f"  candidates after filter: {len(new_cands)} (clip-embedding…)")
        if not new_cands:
            print(f"  no new candidates to consider")
            continue

        if embed_image is None:
            from src.embeddings import embed_image as _embed
            embed_image = _embed

        # CLIP-embed candidates
        cand_embs = []
        cand_keep = []
        for rec in new_cands:
            try:
                e = embed_image(Path(rec["_local"]))
                en = e / max(float(np.linalg.norm(e)), 1e-9)
                cand_embs.append(en)
                cand_keep.append(rec)
            except Exception as e:  # noqa: BLE001
                continue
        if not cand_embs:
            print(f"  all candidates failed CLIP-embed")
            continue
        cand_mat = np.stack(cand_embs).astype(np.float32)
        grand["candidates"] += len(cand_keep)
        grand["embedded_clip"] += len(cand_keep)

        # Rank by MAX similarity to any exemplar (= MIN distance)
        sims = cand_mat @ exemplars.T   # (C, M)
        best_per_cand = sims.max(axis=1)
        ranked_idx = np.argsort(-best_per_cand)
        top_idx = ranked_idx[: args.top_k]
        print(f"  top-{len(top_idx)} candidates by visual similarity (max sim):")
        for i in top_idx[:5]:
            print(f"    sim={best_per_cand[i]:.3f}  {cand_keep[i].get('filepath','')[:80]}")

        # Vision-verify each top candidate
        n_conf = n_emb = n_check = 0
        for ci in top_idx:
            rec = cand_keep[ci]
            sha = rec["_sha"]
            n_check += 1
            grand["vision_checked"] += 1
            v = vision_verify(or_client, Path(rec["_local"]), slug,
                              CLASS_DESCRIPTIONS.get(slug, slug), args.model)
            rare_cache[f"{sha}|{slug}"] = v
            if not v.get("match") or v.get("confidence", 0) < args.min_vision_confidence:
                continue
            n_conf += 1
            grand["vision_confirmed"] += 1
            if args.dry_run:
                continue
            try:
                emb_proj = cand_mat[ci].tolist()   # original CLIP, not projected
                db.table("photo_embeddings").upsert({
                    "sha256": sha,
                    "hse_type_slug": slug,
                    "location_slug": "Common_working_area",
                    "label_source": LABEL_SOURCE,
                    "project_code": f"P_{rec.get('project_id') or 'AECIS'}",
                    "issue_id": str(rec.get("issue_id") or ""),
                    "source_path": f"aecis_curated_rare_nbr/{rec.get('filepath','')}",
                    "embedding": emb_proj,
                    "tenant_id": tenant_id,
                }, on_conflict="sha256").execute()
                n_emb += 1
                grand["embedded_final"] += 1
                existing_shas.add(sha)
            except Exception as e:  # noqa: BLE001
                print(f"    upsert err: {e}")

        print(f"  ▸ checked={n_check}  confirmed={n_conf}  embedded={n_emb}")

        # Cache flush after each class
        try:
            RARE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = RARE_CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(rare_cache, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(RARE_CACHE_PATH)
        except Exception:  # noqa: BLE001
            pass

    print(f"\n=== grand totals ===")
    for k, v in grand.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
