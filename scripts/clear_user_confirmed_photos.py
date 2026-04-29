"""Clear user-uploaded photos (and their training-set embeddings) that were
confirmed through the webapp.

Scope (intentionally narrow):
  IN:  photos in tenants OTHER than 'Training Corpus' that have at least
       one corrections row. These are user-uploaded photos the inspector
       reviewed and either confirmed or corrected — both forms add the
       photo to the training set via photo_embeddings.
  OUT: photos in 'Training Corpus' tenant (auto-seeded data, ~2587 rows).
       These stay regardless of what the inspector did.

Cascade: same as /api/photos/{id} DELETE handler in webapp/app.py:
  classify_jobs -> corrections -> classifications -> photo_embeddings
  (only sha unique to deleted photos) -> photos -> R2 objects.

Idempotent. Re-running after a partial failure is safe — it only
touches rows that still match the criteria.

Usage:
  python scripts/clear_user_confirmed_photos.py            # dry-run
  python scripts/clear_user_confirmed_photos.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
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
                    datefmt="%H:%M:%S")
log = logging.getLogger("clear")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (default = dry-run).")
    ap.add_argument("--include-corrected", action="store_true",
                    help="Also clear photos with action='correct' in addition "
                         "to action='confirm'. Default: BOTH are cleared (any "
                         "correction row counts as 'reviewed' = added to "
                         "training).")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        return 2

    bucket = os.environ.get("R2_BUCKET")
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_access = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")

    from supabase import create_client
    db = create_client(url, key)

    # ---- 1. Identify the Training Corpus tenant (= EXCLUDED from clear) ----
    train = (
        db.table("tenants").select("id,name")
          .eq("name", "Training Corpus").limit(1).execute().data
    )
    train_id = train[0]["id"] if train else None
    if train_id:
        log.info("Training Corpus tenant: %s (will NOT be touched)", train_id)
    else:
        log.warning("No 'Training Corpus' tenant — proceeding without exclusion. "
                    "If auto-seeded data is in another tenant, ABORT and re-check.")

    # ---- 2. Find candidate photos: any photo NOT in Training Corpus that
    # has at least one corrections row. ----
    log.info("Scanning photos...")
    photos: list[dict] = []
    offset = 0
    while True:
        q = db.table("photos").select(
            "id,sha256,storage_key,tenant_id,original_filename,uploaded_at"
        ).range(offset, offset + 999)
        if train_id:
            q = q.neq("tenant_id", train_id)
        batch = q.execute().data or []
        if not batch:
            break
        photos.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    log.info("Total non-training photos: %d", len(photos))

    if not photos:
        print("Nothing to clear.")
        return 0

    photo_ids = [p["id"] for p in photos]
    # Find which have at least one correction row
    reviewed_ids: set[str] = set()
    for i in range(0, len(photo_ids), 200):
        chunk = photo_ids[i:i + 200]
        q = db.table("corrections").select("photo_id, action").in_("photo_id", chunk)
        if not args.include_corrected:
            # Default: both confirm AND correct count as "reviewed/added to
            # training" — passing --no flag is a misnomer here. We keep BOTH
            # because both add embeddings to the training set.
            pass
        rows = q.limit(10000).execute().data or []
        for r in rows:
            reviewed_ids.add(r["photo_id"])

    target_photos = [p for p in photos if p["id"] in reviewed_ids]
    log.info("Photos with at least one correction: %d", len(target_photos))

    # ---- 3. Show preview ----
    print()
    print("=" * 70)
    print("PHOTOS TO CLEAR (preview)")
    print("=" * 70)
    for p in target_photos:
        sha = (p.get("sha256") or "")[:10]
        fn = p.get("original_filename") or "(unnamed)"
        when = (p.get("uploaded_at") or "")[:10]
        print(f"  {sha}  {when}  {fn}")
    if not target_photos:
        print("  (none)")
        return 0

    print()
    print(f"Counts: {len(target_photos)} photos")
    print("Cascade per photo: classify_jobs + corrections + classifications "
          "+ photo_embeddings (sha-unique) + photos row + R2 object")

    if not args.apply:
        print()
        print("DRY-RUN. Pass --apply to actually delete.")
        return 0

    if not all([bucket, r2_account, r2_access, r2_secret]):
        log.warning("R2_* env vars missing — DB rows will be deleted but R2 "
                    "objects will be left as orphans.")
        r2 = None
    else:
        import boto3
        r2 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_access,
            aws_secret_access_key=r2_secret,
            region_name="auto",
        )

    target_ids = [p["id"] for p in target_photos]
    target_shas = list({p["sha256"] for p in target_photos if p.get("sha256")})

    # ---- 4. Cascade in batches ----
    log.info("Deleting child rows...")
    for tbl in ("classify_jobs", "corrections", "classifications"):
        for i in range(0, len(target_ids), 100):
            chunk = target_ids[i:i + 100]
            try:
                db.table(tbl).delete().in_("photo_id", chunk).execute()
            except Exception as e:  # noqa: BLE001
                log.warning("  %s delete failed (chunk %d): %s", tbl, i, e)

    # photo_embeddings: only drop a sha if NO other (surviving) photo
    # references it. Skip shas that overlap with photos we're not deleting
    # (e.g. a Training Corpus photo with the same sha — exact-bytes dup).
    log.info("Computing safe-to-drop shas in photo_embeddings...")
    safe_shas: list[str] = []
    for sha in target_shas:
        try:
            others = (
                db.table("photos").select("id").eq("sha256", sha)
                  .not_.in_("id", target_ids).limit(1).execute().data or []
            )
            if not others:
                safe_shas.append(sha)
        except Exception as e:  # noqa: BLE001
            log.warning("  survival-check %s: %s", sha[:10], e)
    log.info("Safe to drop %d/%d sha-rows in photo_embeddings", len(safe_shas), len(target_shas))
    for i in range(0, len(safe_shas), 100):
        chunk = safe_shas[i:i + 100]
        try:
            db.table("photo_embeddings").delete().in_("sha256", chunk).execute()
        except Exception as e:  # noqa: BLE001
            log.warning("  photo_embeddings delete failed: %s", e)

    # photos rows
    log.info("Deleting photo rows...")
    deleted = 0
    for i in range(0, len(target_ids), 100):
        chunk = target_ids[i:i + 100]
        try:
            db.table("photos").delete().in_("id", chunk).execute()
            deleted += len(chunk)
        except Exception as e:  # noqa: BLE001
            log.error("  photos delete failed (chunk %d): %s", i, e)

    # R2 objects (best-effort)
    if r2:
        log.info("Deleting R2 objects...")
        for p in target_photos:
            key_ = p.get("storage_key")
            if not key_:
                continue
            try:
                r2.delete_object(Bucket=bucket, Key=key_)
            except Exception as e:  # noqa: BLE001
                log.warning("  R2 delete failed for %s: %s", key_, e)

    print()
    print(f"Done. Deleted {deleted} photos + cascade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
