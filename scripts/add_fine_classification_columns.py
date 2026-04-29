"""Print SQL for adding fine sub-type columns to the classifications table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why these columns: the two-stage classifier (see webapp/worker.py) now
picks both the broad parent class (hse_type_slug) AND a specific AECIS
fine sub-type (fine_hse_type_slug) per photo. The corrections table
already had fine_hse_type_slug — this migration adds it to
classifications too so the AI's fine pick lives next to its parent pick.

Until applied, the worker's insert silently strips the fine columns,
so it's safe to deploy code before SQL.

Usage:
  python scripts/add_fine_classification_columns.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- AI's specific (fine) sub-type pick from Stage 2 of the two-stage classifier.
-- May be NULL when Stage 2 returned low confidence (no specific item fit) or
-- when the parent has no fine sub-types defined.
ALTER TABLE classifications
  ADD COLUMN IF NOT EXISTS fine_hse_type_slug TEXT NULL;
ALTER TABLE classifications
  ADD COLUMN IF NOT EXISTS fine_hse_type_confidence DOUBLE PRECISION NULL;

-- Index speeds up exports / filters that group by fine type.
CREATE INDEX IF NOT EXISTS idx_classifications_fine_hse
  ON classifications (fine_hse_type_slug)
  WHERE fine_hse_type_slug IS NOT NULL;
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print("Existing classifications rows will have NULL fine_hse_type_slug —")
    print("only NEW classifications (after this migration + new worker code)")
    print("populate it. Backfill is optional and described below.")
    print()
    print("Optional one-shot backfill of existing classifications (the AI's")
    print("Stage 1 pick is preserved; Stage 2 is run anew on each photo):")
    print()
    print("  python scripts/backfill_fine_classifications.py --dry-run")
    print("  python scripts/backfill_fine_classifications.py --apply")
    print("  (script provided separately — run after deploying the new code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
