"""Backdoor seed path — writes CLIP embeddings directly to
photo_embeddings, bypassing the /api/upload + classify_jobs +
worker queue chain. Implements FUTURE_IMPLEMENTATIONS.txt §2.

Why this exists:
  The webapp's dropzone uploads photos serially via /api/upload,
  one POST per file, each waiting on R2 + Supabase + classify_job
  insert. Observed throughput in production: ~35 s per photo.
  For a 580-photo seed batch that's 5.6 hours of UI driving.

What this does instead:
  1. Read Issue_Gen/photos/manifest.validated.jsonl (the VLM-gated
     580 from scripts/seed_validate_aecis.py).
  2. For each photo, compute the 512-dim CLIP embedding locally
     (src.embeddings.embed_image) — no API call, ~50 ms/photo on
     CPU.
  3. Build the upsert payload: sha256, hse_type_slug (from the
     Haiku validation), location_slug, embedding, plus the
     provenance fields seed_label_source / project_code / issue_id.
  4. Batch upsert into photo_embeddings (Supabase accepts 1000+
     rows per HTTP request). label_source = 'aecis_seed_v1_batch_NNN'
     so each chunk is independently rollback-able with one DELETE.
  5. After each chunk of 50, re-run a k-NN-only eval against the
     100 holdouts (no Sonnet calls; just nearest-neighbour vote)
     and append a row to tmp/seed_accuracy_curve.csv.

What this does NOT do:
  - Upload bytes to R2 (k-NN retrieval doesn't need the bytes,
    only the embedding + labels).
  - Insert into photos / classifications / classify_jobs tables
    (those are for the inspector UI surface — seed photos don't
    need to be reviewable).
  - Trigger the worker queue.
  - Touch the production daily-quota / per-user counters.

Throughput target: 50 photos per chunk in < 30 s end-to-end
(CLIP embeddings dominate; ~50 ms × 50 = 2.5 s, then a single
upsert + a 100-row eval).

Rollback per batch:
  DELETE FROM photo_embeddings WHERE label_source = 'aecis_seed_v1_batch_NNN';

Usage:
  ./.venv-webapp/Scripts/python.exe scripts/seed_direct.py \\
      [--manifest Issue_Gen/photos/manifest.validated.jsonl] \\
      [--chunk-size 50] \\
      [--start-batch 1] \\
      [--limit-batches N] \\
      [--skip-eval] \\
      [--dry-run]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MANIFEST = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.validated.jsonl"
PHOTOS_ROOT = REPO_ROOT / "Issue_Gen" / "photos"
CURVE_CSV = REPO_ROOT / "tmp" / "seed_accuracy_curve.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed_direct")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path, chunk_size: int):
    """Read manifest.validated.jsonl, dedup by filepath, split into chunks.
    Returns list[list[dict]]."""
    if not manifest_path.exists():
        log.error("manifest missing at %s", manifest_path)
        sys.exit(2)
    seen, out = set(), []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = rec.get("filepath")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            # Match the downloader's local-path sanitizer (Windows-reserved
            # chars like ':' get rewritten to '_'). Original filepath stays
            # in the record for upstream signers / DB rows.
            safe = fp
            for c in ':*?"<>|':
                safe = safe.replace(c, "_")
            local = PHOTOS_ROOT / safe
            if not local.exists():
                continue
            rec["local_path"] = str(local)
            out.append(rec)
    if not out:
        log.error("no photos in manifest")
        sys.exit(2)
    return [out[i : i + chunk_size] for i in range(0, len(out), chunk_size)]


def fetch_holdouts(db) -> list[dict]:
    """Return the 100 holdout rows (sha256, embedding, hse_type_slug).
    Used for the k-NN eval at each checkpoint."""
    rows = (db.table("photo_embeddings")
              .select("sha256, embedding, hse_type_slug, location_slug")
              .eq("is_holdout", True)
              .execute().data or [])
    return rows


def eval_holdouts_via_knn(db, holdouts: list[dict]) -> dict:
    """For each holdout, run match_photo_embeddings (which excludes
    is_holdout=TRUE rows) and check whether the nearest neighbours
    carry the same hse_type_slug. Returns top-1 / top-3 hit rates.

    Pure retrieval-quality measurement — no LLM calls. The reasoning
    is that if adding seeds improves nearest-neighbour vote, the
    production classifier (which uses Sonnet over the same top-K
    candidates) will also improve."""
    top1 = top3 = total = 0
    distances: list[float] = []
    for h in holdouts:
        emb = h.get("embedding")
        gt = h.get("hse_type_slug")
        if not emb or not gt:
            continue
        # Some Supabase clients return embedding as a string like "[0.12,...]"
        # rather than a list. Normalize.
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue
        try:
            res = db.rpc("match_photo_embeddings", {
                "query_embedding": emb,
                "match_k": 5,
            }).execute().data or []
        except Exception as e:  # noqa: BLE001
            log.warning("knn rpc failed for %s: %s", h["sha256"][:10], e)
            continue
        total += 1
        if not res:
            continue
        first_slug = res[0].get("hse_type_slug")
        top3_slugs = [r.get("hse_type_slug") for r in res[:3]]
        if first_slug == gt:
            top1 += 1
        if gt in top3_slugs:
            top3 += 1
        if res[0].get("distance") is not None:
            distances.append(float(res[0]["distance"]))
    return {
        "n_evaluated": total,
        "top1": top1 / total if total else 0.0,
        "top3": top3 / total if total else 0.0,
        "mean_nn_distance": sum(distances) / len(distances) if distances else 0.0,
    }


def embed_one(rec: dict, embed_fn) -> dict | None:
    """Compute CLIP embedding for one photo. Returns the upsert
    payload, or None on error. Verifies sha256 matches the
    manifest's claim (catches partial downloads)."""
    p = Path(rec["local_path"])
    try:
        sha_disk = _sha256(p)
    except Exception as e:  # noqa: BLE001
        log.warning("sha read failed for %s: %s", p.name, e)
        return None
    if rec.get("sha256") and rec["sha256"] != sha_disk:
        log.warning("sha mismatch for %s (manifest=%s, disk=%s)",
                    p.name, rec["sha256"][:10], sha_disk[:10])
        # Use disk's sha; manifest might be stale.
    try:
        vec = embed_fn(p)
    except Exception as e:  # noqa: BLE001
        log.warning("embed failed for %s: %s", p.name, e)
        return None
    return {
        "sha": sha_disk,
        "hse": rec.get("vlm_hse_type_slug"),
        "loc": rec.get("vlm_location_slug"),
        "embedding": vec.tolist(),
        "issue_id": rec.get("issue_id", ""),
        "project_id": rec.get("project_id", ""),
        "filepath": rec.get("filepath", ""),
    }


def upsert_chunk(db, prepared: list[dict], label_source: str,
                 tenant_id: str | None) -> int:
    """Batch upsert prepared embedding payloads. Returns count
    successfully written. on_conflict='sha256' so a sha already in
    photo_embeddings gets its labels/embedding REPLACED — that's
    fine for our seed case because we trust the new label more.

    Dedupes by sha256 within the batch first: AECIS sometimes stores
    the same JPEG under multiple filepaths (different issues
    referencing the same underlying file). Postgres rejects an
    upsert that touches the same conflict key twice — keep the last
    occurrence and proceed."""
    if not prepared:
        return 0
    by_sha: dict[str, dict] = {}
    for p in prepared:
        sha = p.get("sha") or ""
        if not sha:
            continue   # skip rows with no sha — they'd all collide on ""
        by_sha[sha] = p   # last write wins per sha
    deduped = list(by_sha.values())
    if len(deduped) < len(prepared):
        log.info("  dedup: %d → %d unique sha256", len(prepared), len(deduped))

    def row_for(p: dict) -> dict:
        return {
            "sha256": p["sha"],
            "hse_type_slug": p["hse"],
            "location_slug": p["loc"],
            "label_source": label_source,
            "project_code": f"P_{p['project_id']}" if p["project_id"] else "AECIS",
            "issue_id": str(p["issue_id"]),
            "source_path": f"aecis_seed/{p['filepath']}",
            "embedding": p["embedding"],
            "tenant_id": tenant_id,
        }

    payload = [row_for(p) for p in deduped]
    try:
        db.table("photo_embeddings").upsert(
            payload, on_conflict="sha256"
        ).execute()
        return len(payload)
    except Exception as e:  # noqa: BLE001
        log.warning("batch upsert failed (%s) — falling back to per-row", e)
        n_ok = 0
        for row in payload:
            try:
                db.table("photo_embeddings").upsert(
                    [row], on_conflict="sha256"
                ).execute()
                n_ok += 1
            except Exception as ee:  # noqa: BLE001
                log.warning("  per-row failed sha=%s: %s",
                            (row["sha256"] or "")[:12], ee)
        return n_ok


def resolve_tenant(db) -> str | None:
    """Look up the demo tenant id so seed embeddings carry the
    same scope as the existing production rows."""
    try:
        rows = (db.table("tenants").select("id")
                  .eq("name", "Public Demo").limit(1).execute().data or [])
        if rows:
            return rows[0]["id"]
    except Exception as e:  # noqa: BLE001
        log.warning("tenant lookup failed: %s", e)
    return None


def append_curve(row: dict) -> None:
    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CURVE_CSV.exists()
    with CURVE_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts", "batch_n", "seed_label_source", "n_seeds_total",
            "n_evaluated", "top1", "top3", "mean_nn_distance",
            "chunk_wall_s", "eval_wall_s",
        ])
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--start-batch", type=int, default=1)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4,
                    help="threads for parallel CLIP embedding")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--label-source-prefix", default="aecis_seed_v1_batch_")
    args = ap.parse_args()

    chunks = load_manifest(Path(args.manifest), args.chunk_size)
    log.info("manifest: %d photos, %d chunks of up to %d",
             sum(len(c) for c in chunks), len(chunks), args.chunk_size)
    if args.dry_run:
        for i, c in enumerate(chunks, 1):
            log.info("  batch %3d: %d photos", i, len(c))
        return 0

    from webapp.app import get_db
    from src.embeddings import embed_image
    db = get_db()
    tenant_id = resolve_tenant(db)
    log.info("tenant: %s", tenant_id)

    holdouts = []
    if not args.skip_eval:
        holdouts = fetch_holdouts(db)
        log.info("holdouts: %d", len(holdouts))

        # Baseline (no new seeds) at batch_n=0.
        log.info("baseline eval (no new seeds yet)…")
        t0 = time.perf_counter()
        result = eval_holdouts_via_knn(db, holdouts)
        eval_s = time.perf_counter() - t0
        log.info("  baseline: top1=%.1f%% top3=%.1f%% n=%d (%.1fs)",
                 result["top1"] * 100, result["top3"] * 100,
                 result["n_evaluated"], eval_s)
        append_curve({
            "ts": time.strftime("%H:%M:%S"),
            "batch_n": 0,
            "seed_label_source": "(baseline, no seeds)",
            "n_seeds_total": 0,
            "n_evaluated": result["n_evaluated"],
            "top1": round(result["top1"], 4),
            "top3": round(result["top3"], 4),
            "mean_nn_distance": round(result["mean_nn_distance"], 4),
            "chunk_wall_s": 0,
            "eval_wall_s": round(eval_s, 1),
        })

    # Lazy-load CLIP (this triggers the ~600 MB download on first run).
    log.info("loading CLIP model…")
    _ = embed_image(Path(chunks[0][0]["local_path"]))   # warm
    log.info("CLIP ready")

    n_seeds_total = 0
    end = (args.start_batch - 1 + args.limit_batches) if args.limit_batches else len(chunks)
    for i in range(args.start_batch - 1, min(end, len(chunks))):
        chunk = chunks[i]
        batch_n = i + 1
        tag = f"{args.label_source_prefix}{batch_n:03d}"
        log.info("--- batch %d/%d [%s] · %d photos ---", batch_n, len(chunks), tag, len(chunk))

        # Stage 1: embed in parallel
        c0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            prepared = list(ex.map(lambda r: embed_one(r, embed_image), chunk))
        prepared = [p for p in prepared if p is not None]
        log.info("  embedded %d/%d photos in %.1fs",
                 len(prepared), len(chunk), time.perf_counter() - c0)

        # Stage 2: batch upsert
        n_written = upsert_chunk(db, prepared, tag, tenant_id)
        chunk_s = time.perf_counter() - c0
        log.info("  upserted %d rows (label_source=%s) in %.1fs total",
                 n_written, tag, chunk_s)
        n_seeds_total += n_written

        if args.skip_eval:
            continue

        # Stage 3: eval against holdouts
        e0 = time.perf_counter()
        result = eval_holdouts_via_knn(db, holdouts)
        eval_s = time.perf_counter() - e0
        log.info("  eval: top1=%.1f%% top3=%.1f%% n=%d (%.1fs)",
                 result["top1"] * 100, result["top3"] * 100,
                 result["n_evaluated"], eval_s)
        append_curve({
            "ts": time.strftime("%H:%M:%S"),
            "batch_n": batch_n,
            "seed_label_source": tag,
            "n_seeds_total": n_seeds_total,
            "n_evaluated": result["n_evaluated"],
            "top1": round(result["top1"], 4),
            "top3": round(result["top3"], 4),
            "mean_nn_distance": round(result["mean_nn_distance"], 4),
            "chunk_wall_s": round(chunk_s, 1),
            "eval_wall_s": round(eval_s, 1),
        })

    log.info("done — curve at %s", CURVE_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
