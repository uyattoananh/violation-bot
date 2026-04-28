"""Print SQL for adding batch tracking to the photos table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Usage:
  python scripts/add_batch_columns.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- Group photos uploaded together as a single 'job'. The frontend mints
-- a UUID per browser session and sends it on every upload, so the user
-- only sees / downloads the photos in their current upload session.
ALTER TABLE photos ADD COLUMN IF NOT EXISTS batch_id    UUID NULL;
ALTER TABLE photos ADD COLUMN IF NOT EXISTS batch_label TEXT NULL;

-- Speed up the per-batch queries the webapp does on every poll
CREATE INDEX IF NOT EXISTS idx_photos_batch ON photos (tenant_id, batch_id);

-- Per-batch review state. Without this, a photo that was confirmed in a
-- previous batch (e.g. via auto_seed_from_disk) shows up as "already
-- confirmed" when re-uploaded into a new batch, skipping the user's
-- inspector review. With it, /api/pending only counts a photo as
-- reviewed when there's a correction row tagged with the CURRENT batch.
ALTER TABLE corrections ADD COLUMN IF NOT EXISTS batch_id UUID NULL;
CREATE INDEX IF NOT EXISTS idx_corrections_batch ON corrections (batch_id);
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print("After it succeeds, redeploy the webapp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
