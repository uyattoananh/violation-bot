"""Nightly backup of the irreplaceable inspector-labor data to R2.

Backs up two tables that represent months of inspector work:
  - corrections     (the gold: every confirm/correct decision)
  - classifications (AI predictions, useful for re-training analysis)
  - photos          (metadata only — the actual image bytes are
                     already durable in R2 via storage_key)

Output format: gzipped newline-delimited JSON (one row per line). Cheap
to compress, easy to grep, easy to restore via supabase-py inserts.

R2 layout:
  s3://<R2_BUCKET>/backups/YYYY-MM-DD/{corrections,classifications,photos}.jsonl.gz

Why JSONL + gzip rather than a SQL dump:
  - No pg_dump / postgresql-client system dep
  - JSONL is forward-compatible across schema changes
  - gzip on construction-violation row counts is ~10x compression
  - Trivial to restore: pipe through `gunzip | jq`, then re-insert via
    a small Python script

Idempotent — uploads under a date-stamped key. Re-running the same day
overwrites that day's file.

Usage:
  python scripts/backup_to_r2.py
  python scripts/backup_to_r2.py --tables corrections,classifications

Cron suggestion (UTC midnight):
  0 0 * * * cd /root/violation-bot && .venv-webapp/bin/python scripts/backup_to_r2.py >> /var/log/violation-backup.log 2>&1
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger("backup")


DEFAULT_TABLES = ["corrections", "classifications", "photos"]
PAGE = 1000   # supabase-py default cap is also 1000


def _dump_table(db, table: str) -> bytes:
    """Page through every row in `table`, return as gzipped JSONL bytes."""
    out_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=out_buf, mode="wb", compresslevel=6) as gz:
        offset = 0
        n = 0
        while True:
            try:
                rows = (
                    db.table(table).select("*")
                      .order("created_at", desc=False)
                      .range(offset, offset + PAGE - 1)
                      .execute().data or []
                )
            except Exception as e:  # noqa: BLE001
                # Some tables don't have a created_at column — fall back
                # to id-based pagination.
                if "created_at" in str(e):
                    rows = (
                        db.table(table).select("*")
                          .order("id", desc=False)
                          .range(offset, offset + PAGE - 1)
                          .execute().data or []
                    )
                else:
                    raise
            if not rows:
                break
            for r in rows:
                gz.write((json.dumps(r, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
                n += 1
            if len(rows) < PAGE:
                break
            offset += PAGE
        log.info("%s: %d rows dumped", table, n)
    return out_buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES),
                    help="Comma-separated table names")
    ap.add_argument("--prefix", default="backups",
                    help="R2 key prefix")
    ap.add_argument("--dry-run", action="store_true",
                    help="Dump locally, skip R2 upload")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        log.error("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required")
        return 2

    bucket = os.environ.get("R2_BUCKET")
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_access = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not args.dry_run and not all([bucket, r2_account, r2_access, r2_secret]):
        log.error("R2_* env vars required for upload (or pass --dry-run)")
        return 2

    from supabase import create_client
    db = create_client(url, key)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]

    r2 = None
    if not args.dry_run:
        import boto3
        r2 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_access,
            aws_secret_access_key=r2_secret,
            region_name="auto",
        )

    total_bytes = 0
    for t in tables:
        log.info("dumping %s...", t)
        try:
            blob = _dump_table(db, t)
        except Exception as e:  # noqa: BLE001
            log.exception("dump %s failed: %s", t, e)
            continue
        total_bytes += len(blob)
        if args.dry_run:
            local = REPO_ROOT / f"_backup_{today}_{t}.jsonl.gz"
            local.write_bytes(blob)
            log.info("  dry-run: wrote %s (%d bytes)", local.name, len(blob))
            continue
        key_path = f"{args.prefix}/{today}/{t}.jsonl.gz"
        try:
            r2.put_object(
                Bucket=bucket, Key=key_path,
                Body=blob,
                ContentType="application/gzip",
                ContentEncoding="gzip",
            )
            log.info("  uploaded s3://%s/%s (%d bytes)", bucket, key_path, len(blob))
        except Exception as e:  # noqa: BLE001
            log.exception("upload %s failed: %s", key_path, e)

    log.info("done. total %.2f MB across %d tables", total_bytes / 1e6, len(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
