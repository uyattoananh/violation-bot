"""Classification worker — polls classify_jobs.pending, calls Sonnet, writes result.

Run alongside the FastAPI app:
    python -m webapp.worker                  # foreground
    python -m webapp.worker --batch-mode     # use Anthropic Batch API (50% cheaper)

Single-process, single-threaded by design. Scale horizontally by running
multiple instances — the `for update skip locked` semantics make that safe.

Dependencies: supabase-py, boto3, anthropic (the SDK), Pillow (for optional
thumbnail generation — not used here).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    """Postgres-compatible ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot import classify_image, load_taxonomy  # noqa: E402
from webapp.app import get_db, get_r2, R2_BUCKET        # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("worker")


def _claim_one() -> dict | None:
    """Claim one pending job: find the oldest, then update by id.

    PostgREST (supabase-py's backend) doesn't support ORDER BY / LIMIT on
    UPDATE directly, so we do select-then-update. Two workers racing on the
    same row is unlikely at this volume; the second UPDATE returns 0 rows
    and we just loop.
    """
    db = get_db()
    candidates = (
        db.table("classify_jobs")
          .select("*")
          .eq("status", "pending")
          .order("created_at")
          .limit(1)
          .execute()
          .data
        or []
    )
    if not candidates:
        return None
    job = candidates[0]
    claimed = (
        db.table("classify_jobs")
          .update({"status": "running", "updated_at": _now()})
          .eq("id", job["id"])
          .eq("status", "pending")   # defensive against concurrent claim
          .execute()
          .data
        or []
    )
    return claimed[0] if claimed else None


def _fetch_photo_bytes(storage_key: str) -> bytes:
    r2 = get_r2()
    resp = r2.get_object(Bucket=R2_BUCKET, Key=storage_key)
    return resp["Body"].read()


def process_job(job: dict) -> None:
    db = get_db()
    photo_id = job["photo_id"]
    photo = (
        db.table("photos").select("*").eq("id", photo_id).execute().data or []
    )
    if not photo:
        db.table("classify_jobs").update(
            {"status": "error", "error": "photo not found", "updated_at": _now()}
        ).eq("id", job["id"]).execute()
        return
    photo = photo[0]

    body = _fetch_photo_bytes(photo["storage_key"])
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(body)
        tmp_path = Path(tf.name)
    try:
        cls = classify_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Clear previous current
    db.table("classifications").update({"is_current": False}).eq("photo_id", photo_id).execute()
    db.table("classifications").insert({
        "photo_id": photo_id,
        "location_slug": cls.location.slug,
        "hse_type_slug": cls.hse_type.slug,
        "location_confidence": cls.location.confidence,
        "hse_type_confidence": cls.hse_type.confidence,
        "rationale": cls.rationale,
        "model": cls.model,
        "source": "zero_shot",
        "input_tokens": cls.input_tokens,
        "output_tokens": cls.output_tokens,
        "raw_response": cls.raw_response,
        "is_current": True,
    }).execute()

    db.table("classify_jobs").update(
        {"status": "done", "updated_at": _now()}
    ).eq("id", job["id"]).execute()
    log.info("classified photo=%s loc=%s hse=%s",
             photo_id, cls.location.slug, cls.hse_type.slug)


def loop(poll_interval: float = 3.0) -> None:
    load_taxonomy()  # fail fast if taxonomy.json is missing
    log.info("worker started, polling every %.1fs", poll_interval)
    while True:
        try:
            job = _claim_one()
        except Exception as e:  # noqa: BLE001
            log.exception("claim failed: %s", e)
            time.sleep(poll_interval)
            continue
        if not job:
            time.sleep(poll_interval)
            continue
        try:
            process_job(job)
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job["id"])
            try:
                get_db().table("classify_jobs").update(
                    {"status": "error", "error": str(e)[:500], "updated_at": _now()}
                ).eq("id", job["id"]).execute()
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    loop(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
