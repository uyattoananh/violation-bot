"""Find R2 objects with no matching photos.storage_key and delete them.

Usage:
  python scripts/clean_orphan_r2.py              # dry-run — counts only
  python scripts/clean_orphan_r2.py --apply      # actually delete
  python scripts/clean_orphan_r2.py --apply --no-prompt  # skip confirm

What's orphaned: any R2 object whose key is NOT present in
photos.storage_key. After we hard-delete photos rows (e.g. the
QA wipe in scripts/wipe_test_stats.py, or the eval batch wipes
inline earlier), the R2 blobs stay because R2 has no FK
relationship with the DB. This reaper is the second half of
that cleanup.

Safety:
  - dry-run by default: lists what would be deleted, deletes nothing
  - prompts before deleting unless --no-prompt
  - batches deletes (R2 supports up to 1000 keys per call)
  - reports total bytes freed
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import boto3
from botocore.config import Config
from supabase import create_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is dry-run)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip confirmation prompt before deleting")
    ap.add_argument("--max-list", type=int, default=1_000_000,
                    help="cap on the R2 list pass (safety; default 1M)")
    args = ap.parse_args()

    bucket = os.environ["R2_BUCKET"]
    account = os.environ["R2_ACCOUNT_ID"]
    r2 = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(retries={"max_attempts": 3}),
    )
    sb = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # ---- 1. Build the set of LIVE storage_keys from the DB ----
    print("fetching live storage_keys from photos ...")
    live_keys: set[str] = set()
    page_size = 1000
    offset = 0
    while True:
        rows = (
            sb.table("photos").select("storage_key")
              .range(offset, offset + page_size - 1)
              .execute().data or []
        )
        if not rows:
            break
        live_keys.update(r["storage_key"] for r in rows if r.get("storage_key"))
        offset += page_size
        if len(rows) < page_size:
            break
    print(f"  live keys: {len(live_keys):,}")

    # ---- 2. Walk R2, collect orphans ----
    print(f"scanning bucket s3://{bucket} ...")
    orphans: list[tuple[str, int]] = []   # (key, bytes)
    total_seen = 0
    total_bytes = 0
    paginator = r2.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents") or []:
            total_seen += 1
            total_bytes += obj["Size"]
            if obj["Key"] not in live_keys:
                orphans.append((obj["Key"], obj["Size"]))
            if total_seen >= args.max_list:
                print(f"  hit --max-list cap ({args.max_list}); stopping scan")
                break
        if total_seen >= args.max_list:
            break

    orphan_bytes = sum(b for _, b in orphans)
    print(f"  R2 objects scanned: {total_seen:,} ({total_bytes/(1024**3):.2f} GB)")
    print(f"  orphans found:      {len(orphans):,} ({orphan_bytes/(1024**3):.2f} GB)")
    if not orphans:
        print("clean — no orphans to remove.")
        return 0

    # Sample of what would be deleted
    print()
    print(f"sample of {min(10, len(orphans))} orphan keys:")
    for k, b in orphans[:10]:
        print(f"  {b/1024:8.0f} KB  {k}")

    if not args.apply:
        print()
        print(f"DRY-RUN — re-run with --apply to delete {len(orphans):,} keys "
              f"({orphan_bytes/(1024**3):.2f} GB)")
        return 0

    if not args.no_prompt:
        print()
        ans = input(f"DELETE {len(orphans):,} keys ({orphan_bytes/(1024**3):.2f} GB) "
                    f"from s3://{bucket}? type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            print("cancelled")
            return 0

    # ---- 3. Delete in batches of 1000 ----
    print(f"\ndeleting ...")
    deleted = 0
    failed: list[str] = []
    for i in range(0, len(orphans), 1000):
        chunk = orphans[i:i+1000]
        resp = r2.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k, _ in chunk], "Quiet": True},
        )
        # delete_objects with Quiet=True only echoes errors
        errs = resp.get("Errors") or []
        deleted += len(chunk) - len(errs)
        for e in errs:
            failed.append(e.get("Key", "?"))
        print(f"  batch {i//1000 + 1}/{(len(orphans)+999)//1000}: "
              f"+{len(chunk) - len(errs)} deleted, {len(errs)} failed")
    print()
    print(f"done — {deleted:,} keys deleted, {len(failed)} failures")
    if failed:
        print("  first 5 failures:")
        for k in failed[:5]:
            print(f"    {k}")
    print(f"  freed: {orphan_bytes/(1024**3):.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
