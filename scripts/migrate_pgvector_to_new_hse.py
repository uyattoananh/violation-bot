"""Run on the VPS AFTER deploying the new (consolidated) taxonomy.

Updates every row in photo_embeddings, classifications, and corrections
that has a hse_type_slug matching one of the OLD 13 classes — rewriting
it to the NEW consolidated parent slug per data/old_to_new_hse_map.json.

Idempotent: running twice is safe (rows already on the new vocab won't
match the OLD slug filter).

Usage on the VPS:
  cd ~/violation-bot
  source .venv-webapp/bin/activate
  python scripts/migrate_pgvector_to_new_hse.py            # dry-run
  python scripts/migrate_pgvector_to_new_hse.py --apply    # actually update
"""
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually run the UPDATEs (default = dry-run)")
    args = ap.parse_args()

    map_path = REPO_ROOT / "data" / "old_to_new_hse_map.json"
    if not map_path.exists():
        print("missing data/old_to_new_hse_map.json — run apply_aecis_consolidation.py first",
              file=sys.stderr)
        return 1
    mp = json.loads(map_path.read_text(encoding="utf-8"))
    old_to_new = mp["old_to_new_hse_type_slug"]

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        return 2

    from supabase import create_client
    db = create_client(url, key)

    print(f"=== HSE migration — {'APPLY' if args.apply else 'DRY-RUN'} ===")
    print(f"  {len(old_to_new)} legacy slugs in the migration map")
    print()

    grand_total = 0
    for table in ("photo_embeddings", "classifications", "corrections"):
        print(f"--- table: {table} ---")
        per_slug_count = {}
        for old, new in old_to_new.items():
            try:
                res = (
                    db.table(table).select("*", count="exact")
                      .eq("hse_type_slug", old).limit(1).execute()
                )
                n = res.count or 0
            except Exception as e:  # noqa: BLE001
                print(f"  {old:<32} -> {new:<32}  COUNT FAILED: {e}")
                continue
            per_slug_count[old] = (n, new)
            if n:
                print(f"  {n:5d}  {old:<32} -> {new}")
        total = sum(c[0] for c in per_slug_count.values())
        print(f"  TOTAL rows to update: {total}")
        grand_total += total

        if args.apply and total:
            for old, (n, new) in per_slug_count.items():
                if n == 0 or old == new:
                    continue
                try:
                    db.table(table).update({"hse_type_slug": new}).eq("hse_type_slug", old).execute()
                    print(f"  applied: {old} -> {new}  ({n} rows)")
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED applying {old} -> {new}: {e}")
        print()

    print(f"Grand total rows that would be updated: {grand_total}")
    if not args.apply:
        print("Re-run with --apply to actually write the updates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
