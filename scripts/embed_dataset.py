"""Embed all training photos and upload to Supabase pgvector.

Walks ~/Desktop/aecis-violations/*/*/metadata.json, computes a CLIP embedding
for each photo, maps it to the CONSOLIDATED taxonomy (using taxonomy_merges.json),
and upserts rows into `photo_embeddings` keyed by sha256.

Idempotent: photos whose sha256 is already in the DB are skipped.

Usage:
  python scripts/embed_dataset.py
  python scripts/embed_dataset.py --limit 100         # test run
  python scripts/embed_dataset.py --dataset <path>
  python scripts/embed_dataset.py --skip-existing false   # re-embed everything
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

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
log = logging.getLogger("embed")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_source_to_consolidated_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (hse_source -> hse_consolidated, loc_source -> loc_consolidated)."""
    merges = json.loads((REPO_ROOT / "taxonomy_merges.json").read_text(encoding="utf-8"))
    hse_map: dict[str, str] = {}
    for cluster in merges.get("hse_type_clusters", []):
        for src in cluster["absorbs"]:
            hse_map[src] = cluster["slug"]
    loc_map: dict[str, str] = {}
    for cluster in merges.get("location_clusters", []):
        for src in cluster["absorbs"]:
            loc_map[src] = cluster["slug"]
    return hse_map, loc_map


def _gather(root: Path, hse_map: dict[str, str], loc_map: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in root.glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        src_hse = meta_path.parent.parent.name
        src_loc = (meta.get("location_en") or "").replace(" ", "_")
        cons_hse = hse_map.get(src_hse, src_hse)
        cons_loc = loc_map.get(src_loc, src_loc) if src_loc else None
        for ph in (meta.get("photos") or []):
            fname = ph.get("file")
            if not fname:
                continue
            img = meta_path.parent / fname
            if not img.exists():
                continue
            records.append({
                "img": img,
                "sha256": ph.get("sha256"),
                "hse_type_slug": cons_hse,
                "location_slug": cons_loc,
                "label_source": meta.get("label_source", "dtag"),
                "project_code": meta.get("project_code"),
                "issue_id": meta.get("issue_id"),
                "source_path": str(img.relative_to(root)).replace("\\", "/"),
            })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", default="true",
                    help="Set to 'false' to re-embed photos already in the DB.")
    ap.add_argument("--batch-db", type=int, default=200, help="Upsert batch size")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve() if args.dataset else DEFAULT_ROOT
    skip_existing = args.skip_existing.lower() in ("true", "1", "yes")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        log.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return 2
    from supabase import create_client
    db = create_client(url, key)

    hse_map, loc_map = _build_source_to_consolidated_maps()
    log.info("Loaded merge maps: %d hse, %d loc", len(hse_map), len(loc_map))

    records = _gather(root, hse_map, loc_map)
    log.info("Gathered %d photo records", len(records))

    # Dedupe by sha256 in-memory: the same photo often appears in multiple issue
    # folders (inspectors re-attach images). We embed each unique photo once.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in records:
        sha = r.get("sha256")
        if not sha:
            # Compute on-the-fly for the rare case metadata lacks sha256
            sha = _sha256_file(r["img"])
            r["sha256"] = sha
        if sha in seen:
            continue
        seen.add(sha)
        deduped.append(r)
    log.info("Unique photos (by sha256): %d (skipped %d in-file duplicates)",
             len(deduped), len(records) - len(deduped))
    records = deduped

    if skip_existing:
        existing = set()
        # page through all existing sha256 (pgvector can hold thousands — list in chunks)
        offset = 0
        while True:
            batch = (
                db.table("photo_embeddings")
                  .select("sha256")
                  .range(offset, offset + 999)
                  .execute()
                  .data or []
            )
            if not batch:
                break
            existing.update(r["sha256"] for r in batch)
            offset += len(batch)
            if len(batch) < 1000:
                break
        log.info("Already embedded: %d", len(existing))
        before = len(records)
        records = [r for r in records if r["sha256"] and r["sha256"] not in existing]
        log.info("Remaining to embed: %d (skipped %d)", len(records), before - len(records))

    if args.limit:
        records = records[:args.limit]
    if not records:
        print("Nothing to embed.")
        return 0

    # Lazy-import embeddings (heavy) now that we know there's work to do
    from src.embeddings import embed_images

    started = time.time()
    batch_em = 32
    upserts: list[dict[str, Any]] = []
    placed = 0
    for i in range(0, len(records), batch_em):
        sub = records[i:i + batch_em]
        log.info("[%d/%d] embedding %d...", i + len(sub), len(records), len(sub))
        vecs = embed_images([r["img"] for r in sub])
        for r, v in zip(sub, vecs):
            sha = r["sha256"] or _sha256_file(r["img"])
            upserts.append({
                "sha256": sha,
                "hse_type_slug": r["hse_type_slug"],
                "location_slug": r["location_slug"],
                "label_source": r["label_source"],
                "project_code": r["project_code"],
                "issue_id": r["issue_id"],
                "source_path": r["source_path"],
                "embedding": v.tolist(),
            })
        # Flush db in chunks
        if len(upserts) >= args.batch_db:
            db.table("photo_embeddings").upsert(upserts, on_conflict="sha256").execute()
            placed += len(upserts)
            upserts = []

    if upserts:
        db.table("photo_embeddings").upsert(upserts, on_conflict="sha256").execute()
        placed += len(upserts)

    elapsed = time.time() - started
    log.info("Done. Uploaded %d embeddings in %.1fs (%.1f img/s)",
             placed, elapsed, placed / max(elapsed, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
