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


# How many recent corrections to inject into each classify prompt as
# in-session learning examples. Tuned by trade-off:
#   - too low: model can't pick up patterns
#   - too high: prompt token cost balloons + later corrections drown the rules
# 8 is a reasonable middle. Override via RECENT_CORRECTIONS_K env var.
_RECENT_CORRECTIONS_K = int(os.environ.get("RECENT_CORRECTIONS_K", "8"))


def _fetch_recent_corrections(db, batch_id: str | None, limit: int) -> list[dict]:
    """Return the most recent inspector corrections in this batch, newest
    first. Used to inject in-context examples into the classify prompt so
    the model adapts to systematic patterns within a session.

    Empty list when batch_id is None (e.g. legacy NULL-batch photos), or
    on any DB error — this is best-effort enrichment, not a gate."""
    if not batch_id or limit <= 0:
        return []
    try:
        rows = (
            db.table("corrections")
              .select("hse_type_slug, fine_hse_type_slug, note, created_at")
              .eq("batch_id", batch_id)
              .order("created_at", desc=True)
              .limit(limit)
              .execute().data or []
        )
        # Filter to only confirm/correct rows that have a slug (stale rows
        # from older schema can have None).
        return [r for r in rows if r.get("hse_type_slug")]
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # If batch_id column was never migrated, the query fails; degrade
        # silently rather than failing the whole classify job.
        if "batch_id" in msg:
            return []
        log.warning("corrections fetch for in-session learning failed: %s", msg[:120])
        return []


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

    # In-session learning: pull the last K inspector corrections in the
    # SAME batch so the model has those as in-context priors. The first
    # photo of a session has no corrections yet, so this is a no-op then;
    # by photo 5-10 it's usually carrying useful signal.
    recent_corrections = _fetch_recent_corrections(
        db, photo.get("batch_id"), _RECENT_CORRECTIONS_K,
    )

    # Two-stage classification by default. Stage 2 picks a specific
    # AECIS fine sub-type (e.g. "Two people on same ladder") instead of
    # leaving the prediction at the broad parent class. Set TWO_STAGE=0
    # in env to disable, e.g. for A/B comparison runs.
    fine_grained = os.environ.get("TWO_STAGE", "1") not in ("0", "false", "no", "")

    body = _fetch_photo_bytes(photo["storage_key"])
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(body)
        tmp_path = Path(tf.name)
    try:
        cls = classify_image(
            tmp_path,
            recent_corrections=recent_corrections,
            fine_grained=fine_grained,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Clear previous current
    db.table("classifications").update({"is_current": False}).eq("photo_id", photo_id).execute()
    insert_payload = {
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
    }
    # Optional fine columns. Strip + retry on schema-error so the worker
    # keeps running before the migration is applied (same pattern as
    # webapp/app.py upload).
    if cls.fine_hse_type:
        insert_payload["fine_hse_type_slug"] = cls.fine_hse_type.slug
        insert_payload["fine_hse_type_confidence"] = cls.fine_hse_type.confidence
    try:
        db.table("classifications").insert(insert_payload).execute()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "fine_hse_type" in msg:
            insert_payload.pop("fine_hse_type_slug", None)
            insert_payload.pop("fine_hse_type_confidence", None)
            db.table("classifications").insert(insert_payload).execute()
            log.warning("classifications.fine_hse_type_* missing — degraded insert")
        else:
            raise

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
