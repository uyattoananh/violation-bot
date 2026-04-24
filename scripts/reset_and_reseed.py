"""Apply a new taxonomy to Supabase safely.

Drops existing classifications/corrections/jobs (their slugs may no longer
exist in the consolidated taxonomy), wipes locations + hse_types, re-seeds
from taxonomy.json, then re-enqueues one classify_job per existing photo so
the worker re-classifies against the new vocabulary.

Idempotent — safe to run multiple times.
"""
from __future__ import annotations

import json
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
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.", file=sys.stderr)
        return 2

    from supabase import create_client
    db = create_client(url, key)

    tax = json.loads((REPO_ROOT / "taxonomy.json").read_text(encoding="utf-8"))

    # 1. Wipe dependent tables first (FKs prevent slug deletion otherwise)
    for tbl in ["corrections", "classifications", "classify_jobs"]:
        resp = db.table(tbl).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print(f"[1] cleared {tbl:<18} ({len(resp.data or [])} rows deleted)")

    # 2. Wipe taxonomy tables
    for tbl in ["locations", "hse_types"]:
        resp = db.table(tbl).delete().neq("slug", "__noop__").execute()
        print(f"[2] cleared {tbl:<18} ({len(resp.data or [])} rows deleted)")

    # 3. Insert the consolidated taxonomy
    loc_rows = [
        {"slug": l["slug"], "label_en": l["label_en"], "label_vn": l.get("label_vn") or None}
        for l in tax["locations"]
    ]
    hse_rows = [
        {"slug": h["slug"], "label_en": h["label_en"], "label_vn": h.get("label_vn") or None}
        for h in tax["hse_types"]
    ]
    db.table("locations").insert(loc_rows).execute()
    db.table("hse_types").insert(hse_rows).execute()
    print(f"[3] seeded {len(loc_rows)} locations, {len(hse_rows)} hse_types")

    # 4. Re-enqueue existing photos for classification
    photos = db.table("photos").select("id").execute().data or []
    if photos:
        jobs = [{"photo_id": p["id"]} for p in photos]
        db.table("classify_jobs").insert(jobs).execute()
        print(f"[4] re-enqueued {len(jobs)} photos for classification")
    else:
        print(f"[4] no existing photos to re-enqueue")

    print()
    print(f"Done. Worker will re-classify {len(photos)} photos against the new "
          f"{len(hse_rows)}-class taxonomy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
