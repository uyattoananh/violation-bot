"""Wipe all uploaded photos + their classifications/corrections/jobs.

Keeps intact:
  - taxonomy (locations, hse_types)
  - tenants, projects, users
  - photo_embeddings rows from the scraped training set (label_source != 'manual')

Removes:
  - R2 objects under the tenant/project prefixes
  - photos, classifications, corrections, classify_jobs rows
  - photo_embeddings rows added via the inspector feedback loop
    (label_source = 'manual')

Use with care. This does NOT reset the taxonomy or re-seed anything.

Usage:
  python scripts/clear_site_photos.py
  python scripts/clear_site_photos.py --keep-r2    # wipe DB only, leave R2
  python scripts/clear_site_photos.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-r2", action="store_true",
                    help="Skip R2 deletions (DB only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without deleting.")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET", "violation-bot")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required.", file=sys.stderr)
        return 2

    from supabase import create_client
    db = create_client(url, key)

    # ---- 1. R2 delete ----
    if not args.keep_r2 and r2_account and r2_key and r2_secret:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_key, aws_secret_access_key=r2_secret,
            region_name="auto",
        )
        paginator = s3.get_paginator("list_objects_v2")
        to_delete: list[dict] = []
        for page in paginator.paginate(Bucket=r2_bucket):
            for obj in page.get("Contents") or []:
                to_delete.append({"Key": obj["Key"]})
        print(f"R2: found {len(to_delete)} objects in {r2_bucket}")
        if args.dry_run:
            print("  (dry-run) not deleting")
        elif to_delete:
            # S3 DeleteObjects max 1000 per call
            for i in range(0, len(to_delete), 1000):
                chunk = to_delete[i:i + 1000]
                s3.delete_objects(Bucket=r2_bucket, Delete={"Objects": chunk})
            print(f"R2: deleted {len(to_delete)} objects")

    # ---- 2. DB rows ----
    zero = "00000000-0000-0000-0000-000000000000"
    for tbl in ["corrections", "classifications", "classify_jobs"]:
        n = db.table(tbl).select("id", count="exact").limit(1).execute().count
        print(f"DB: {tbl} = {n} rows")
        if n and not args.dry_run:
            db.table(tbl).delete().neq("id", zero).execute()

    n_photos = db.table("photos").select("id", count="exact").limit(1).execute().count
    print(f"DB: photos = {n_photos} rows")
    if n_photos and not args.dry_run:
        db.table("photos").delete().neq("id", zero).execute()

    # ---- 3. pgvector feedback-loop rows only ----
    manual_rows = (
        db.table("photo_embeddings").select("sha256", count="exact")
          .eq("label_source", "manual").limit(1).execute()
    )
    n_manual = manual_rows.count or 0
    print(f"pgvector: manual (feedback-loop) embeddings = {n_manual}  [scraped rows kept]")
    if n_manual and not args.dry_run:
        db.table("photo_embeddings").delete().eq("label_source", "manual").execute()

    # ---- 4. Counts after ----
    if not args.dry_run:
        remaining_embeddings = (
            db.table("photo_embeddings").select("sha256", count="exact").limit(1).execute()
        ).count
        print()
        print(f"Cleanup complete.")
        print(f"  photo_embeddings kept: {remaining_embeddings}")
        print(f"  photos / classifications / corrections / classify_jobs: 0")
    else:
        print()
        print("Dry-run complete. Re-run without --dry-run to execute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
