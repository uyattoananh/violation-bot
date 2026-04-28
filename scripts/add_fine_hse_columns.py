"""One-shot Supabase migration: add fine_hse_type_slug TEXT NULL column to
classifications, corrections, and photo_embeddings tables.

The fine column holds the inspector's chosen AECIS-canonical sub-type
(if any) within the predicted coarse parent. NULL means "no fine
refinement specified", which is the default behaviour after Confirm.

Idempotent: skips columns that already exist.

Usage:
  python scripts/add_fine_hse_columns.py
"""
from __future__ import annotations

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
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        return 2

    # Supabase doesn't expose ALTER TABLE via the REST API. Use the SQL
    # editor in the Supabase dashboard, or psql against the connection
    # string. This script just prints the SQL you need to run manually
    # so the migration is reproducible.
    sql = """\
-- Run in Supabase SQL editor (or via psql).
-- Idempotent: ALTER TABLE ... ADD COLUMN IF NOT EXISTS works on PG 9.6+.

ALTER TABLE classifications  ADD COLUMN IF NOT EXISTS fine_hse_type_slug TEXT NULL;
ALTER TABLE corrections      ADD COLUMN IF NOT EXISTS fine_hse_type_slug TEXT NULL;
ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS fine_hse_type_slug TEXT NULL;

-- Optional: index the corrections lookup if you'll filter by fine slug in the UI
CREATE INDEX IF NOT EXISTS idx_corrections_fine_hse ON corrections (fine_hse_type_slug);
CREATE INDEX IF NOT EXISTS idx_photo_emb_fine_hse  ON photo_embeddings (fine_hse_type_slug);
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print("After it succeeds, redeploy the webapp — the new code reads/writes")
    print("the fine_hse_type_slug column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
