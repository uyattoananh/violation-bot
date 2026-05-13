"""Print SQL for adding seed-provenance columns to the photos table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Usage:
  python scripts/add_is_seed_column.py

================================================================
PURPOSE

Bulk-seeded photos (e.g. ~2,444 AECIS HSE inspection photos pulled
via scripts/seed_download_aecis_photos.py and assigned via the
Playwright assigner) need to feed pgvector retrieval WITHOUT
inflating the production-facing counters:

  - /api/usage/today           (per-user quota meter)
  - /api/batches per-row stats (photo_count / reviewed_count)
  - scripts/ai_audit_*.py      (held-out eval accuracy figures)
  - picker "training set size" hint
  - /admin/stats               (admin dashboard)

This migration adds three columns to flag seed rows + an index that
makes the "WHERE is_seed = false" filter cheap.

================================================================
AUDIT CHECKLIST — every reader that aggregates / counts must filter
is_seed = false. Tick each one off after the migration lands:

  [ ] webapp/app.py  /api/usage/today
  [ ] webapp/app.py  /api/batches            (photo_count, reviewed_count)
  [ ] webapp/app.py  /api/pending            (training_set_size)
  [ ] webapp/app.py  /admin/stats            (per-user uploads/confirms)
  [ ] webapp/app.py  /admin/proposals        (if it counts photos)
  [ ] webapp/app.py  _cleanup_expired_*      (don't auto-delete seed!)
  [ ] webapp/app.py  refreshFineSelect       (picker "training set size")
  [ ] webapp/worker.py  classify loop        (skip seed rows in worker)
  [ ] scripts/ai_audit_*.py                  (held-out eval scripts)
  [ ] scripts/evaluate_rag.py                (any aggregate that touches
                                              photos table)

Seed rows MUST still appear in:
  - photo_embeddings (pgvector k-NN retrieval — that's the whole point)
  - corrections (when an inspector reviews a seeded photo, the
    correction is real training signal regardless of is_seed)

================================================================
ROLLBACK

If a bad seed import contaminated retrieval and needs reversal,
delete by seed_label_source rather than dropping the column:

  DELETE FROM photos WHERE is_seed = true
    AND seed_label_source = 'aecis_issue_v1';

The corresponding photo_embeddings rows cascade automatically via
the foreign key.
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- ----------------------------------------------------------------
-- Seed-provenance columns on photos
-- ----------------------------------------------------------------
-- is_seed:           true for programmatically-imported photos
--                    that must NOT show up in user-facing stats.
-- seed_label_source: e.g. 'aecis_issue_v1' so we can reverse one
--                    bad ingest without touching unrelated seeds.
-- seed_imported_at:  audit timestamp; lets us roll back by date.
ALTER TABLE photos
  ADD COLUMN IF NOT EXISTS is_seed            BOOLEAN     NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS seed_label_source  TEXT        NULL,
  ADD COLUMN IF NOT EXISTS seed_imported_at   TIMESTAMPTZ NULL;

-- Partial index makes the "WHERE is_seed = FALSE" filter on every
-- production aggregate query a btree-scan cost, not a seq-scan
-- penalty. Most queries we care about hit this branch — the
-- ~1,100 hand-corrected production photos vs the ~2,400 seed rows
-- means is_seed=false is the hot path.
CREATE INDEX IF NOT EXISTS idx_photos_not_seed
    ON photos (tenant_id, batch_id)
    WHERE is_seed = FALSE;

-- Mirror index for the rarer "show me the seeds" admin views.
CREATE INDEX IF NOT EXISTS idx_photos_is_seed
    ON photos (seed_label_source, seed_imported_at DESC)
    WHERE is_seed = TRUE;

-- ----------------------------------------------------------------
-- Sanity check after migration — should report 0 seeds initially
-- ----------------------------------------------------------------
-- SELECT count(*) FILTER (WHERE is_seed)            AS seeds,
--        count(*) FILTER (WHERE NOT is_seed)        AS production,
--        count(DISTINCT seed_label_source)          AS seed_sources
-- FROM photos;
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print("After it succeeds, work through the AUDIT CHECKLIST in this")
    print("file's header — every photo-aggregate query needs WHERE is_seed = FALSE")
    print("BEFORE the first seed row is inserted via the assigner pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
