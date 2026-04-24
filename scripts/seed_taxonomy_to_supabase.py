"""Seed the `locations` and `hse_types` tables in Supabase from taxonomy.json.

Idempotent: uses upsert on the `slug` primary key.

Usage:
  python scripts/seed_taxonomy_to_supabase.py
  python scripts/seed_taxonomy_to_supabase.py --taxonomy path/to/taxonomy.json

Prereq: .env with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY populated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAX = REPO_ROOT / "taxonomy.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=str, default=str(DEFAULT_TAX))
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.", file=sys.stderr)
        return 1

    from supabase import create_client
    db = create_client(url, key)

    tax = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))

    loc_rows = [
        {"slug": l["slug"], "label_en": l["label_en"], "label_vn": l.get("label_vn") or None}
        for l in tax["locations"]
    ]
    hse_rows = [
        {"slug": h["slug"], "label_en": h["label_en"], "label_vn": h.get("label_vn") or None}
        for h in tax["hse_types"]
    ]

    print(f"Upserting {len(loc_rows)} locations...")
    db.table("locations").upsert(loc_rows, on_conflict="slug").execute()
    print(f"Upserting {len(hse_rows)} hse_types...")
    db.table("hse_types").upsert(hse_rows, on_conflict="slug").execute()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
