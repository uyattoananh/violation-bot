"""One-shot: move auto-seeded photos (those with batch_id=NULL) out of
the Public Demo tenant into a dedicated Training Corpus tenant.

Why: with the auto-seeded photos in the same tenant as user uploads, every
time a user uploads a photo whose sha256 matches a seeded one (which is
common when developers test with the source dataset), the dedup-attach
moves the existing photo into the new batch — yanking it from the
training pile in the process. Cleanest fix: put the auto-seed data in
its own tenant. RAG retrieval still works (photo_embeddings are queried
globally, no tenant filter), but uploads can no longer collide.

Effects:
  - photo_embeddings rows are NOT touched. The CLIP+RAG corpus stays whole.
  - photos rows with batch_id=NULL get tenant_id = <Training Corpus>
  - corrections + classifications + classify_jobs rows reference photos
    by id, so no FK update needed there.
  - User uploads to Public Demo continue to dedup ONLY against other
    user uploads in Public Demo — the auto-seed pool no longer collides.

Idempotent: skips photos that are already in the training tenant.
Dry-run by default; pass --apply to actually move.

Usage:
  python scripts/separate_training_corpus.py            # dry-run
  python scripts/separate_training_corpus.py --apply
"""
from __future__ import annotations

import argparse
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

TRAINING_TENANT_NAME = "Training Corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually run the UPDATEs (default = dry-run)")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        return 2

    from supabase import create_client
    db = create_client(url, key)

    # 1. Resolve / create the training tenant
    rows = db.table("tenants").select("id").eq("name", TRAINING_TENANT_NAME).execute().data or []
    if rows:
        training_tid = rows[0]["id"]
        print(f"Training tenant: {training_tid} (existing)")
    else:
        if not args.apply:
            print(f"DRY-RUN: would create tenant '{TRAINING_TENANT_NAME}'")
            training_tid = "<would-be-created>"
        else:
            training_tid = db.table("tenants").insert({"name": TRAINING_TENANT_NAME}).execute().data[0]["id"]
            print(f"Training tenant: {training_tid} (created)")

    # 2. Resolve the public tenant
    pub_rows = db.table("tenants").select("id").eq("name", "Public Demo").execute().data or []
    if not pub_rows:
        print("Public Demo tenant not found — nothing to migrate", file=sys.stderr)
        return 1
    public_tid = pub_rows[0]["id"]
    print(f"Public Demo tenant: {public_tid}")

    # 3. Find all photos with batch_id IS NULL in Public Demo
    photos: list[dict] = []
    offset = 0
    while True:
        b = (
            db.table("photos").select("id, sha256")
              .eq("tenant_id", public_tid)
              .is_("batch_id", "null")
              .range(offset, offset + 999).execute().data or []
        )
        if not b:
            break
        photos.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
    print(f"Photos with batch_id=NULL in Public Demo: {len(photos)}")

    if not photos:
        print("Nothing to do.")
        return 0

    if not args.apply:
        print()
        print(f"DRY-RUN: would set tenant_id = {training_tid} on {len(photos)} photos.")
        print("Re-run with --apply to actually move them.")
        return 0

    # 4. Update in chunks of 100
    moved = 0
    failed = 0
    for i in range(0, len(photos), 100):
        chunk_ids = [p["id"] for p in photos[i:i + 100]]
        try:
            db.table("photos").update({"tenant_id": training_tid}) \
              .in_("id", chunk_ids).execute()
            moved += len(chunk_ids)
        except Exception as e:  # noqa: BLE001
            print(f"chunk {i}: FAILED — {e}")
            failed += len(chunk_ids)
        if (i + 100) % 500 == 0:
            print(f"  moved {moved}/{len(photos)}...")

    # 5. Also update photo_embeddings.tenant_id for those photos
    # (so per-tenant RAG queries — should we ever add them — group correctly)
    print(f"  updating photo_embeddings.tenant_id for {len(photos)} sha256s...")
    sha_list = [p["sha256"] for p in photos if p.get("sha256")]
    for i in range(0, len(sha_list), 100):
        chunk = sha_list[i:i + 100]
        try:
            db.table("photo_embeddings").update({"tenant_id": training_tid}) \
              .in_("sha256", chunk).execute()
        except Exception as e:  # noqa: BLE001
            print(f"  embedding chunk {i}: FAILED — {e}")

    print(f"Done. Moved {moved}, failed {failed}, total {len(photos)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
