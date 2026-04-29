"""Print SQL to wipe deployment + training + test data from the
stats and source tables.

Run the printed SQL in Supabase SQL editor AFTER reviewing the
WHERE clauses carefully. NOT idempotent in the destructive sense —
deleted rows are gone for good.

What this targets:
  - rows with user_id IS NULL (anonymous, pre-auth, dev/test)
  - rows whose user_key matches the QA-probe prefix `qa_test_*`
  - rows from before the production go-live date (configurable)

Why:
  Stats currently include any photo/correction/classification that
  ever hit the DB, including:
    - my deployment smoke tests
    - the QA probe's automated submissions
    - early model-training uploads from the dev team
    - random visitors who hit / before AUTH_REQUIRED was on
  None of those should count toward "real inspector activity".

Usage:
  python scripts/wipe_test_stats.py
  # review printed SQL, run in Supabase SQL editor
"""
from __future__ import annotations
import sys


# Configurable — bump this when you want to also discard real
# inspector activity from before a fresh start. Leave blank to
# only filter on user_id IS NULL + qa_test_* prefix.
GO_LIVE_DATE = ""  # e.g. "2026-04-29"


def main() -> int:
    cutoff_clause = f"OR uploaded_at < '{GO_LIVE_DATE}'" if GO_LIVE_DATE else ""
    cutoff_corrs = f"OR created_at < '{GO_LIVE_DATE}'" if GO_LIVE_DATE else ""

    sql = f"""\
-- Run in Supabase SQL editor. DESTRUCTIVE — review WHERE clauses.

-- =========================================================
-- STEP 1: AUDIT — count what would be deleted before doing it.
-- =========================================================
-- Run this first. If counts look right, proceed to STEP 2.

SELECT 'photos with NULL user_id'         AS what,
       COUNT(*) AS rows
  FROM photos
 WHERE user_id IS NULL
UNION ALL
SELECT 'photos with qa_test_ user_key',
       COUNT(*)
  FROM photos
 WHERE user_key LIKE 'qa_test_%'
{f"UNION ALL SELECT 'photos before go-live ({GO_LIVE_DATE})', COUNT(*) FROM photos WHERE uploaded_at < '{GO_LIVE_DATE}'" if GO_LIVE_DATE else ""}
UNION ALL
SELECT 'corrections with NULL user_id',
       COUNT(*)
  FROM corrections
 WHERE user_id IS NULL
UNION ALL
SELECT 'corrections with qa_test_ user_key',
       COUNT(*)
  FROM corrections
 WHERE user_key LIKE 'qa_test_%'
UNION ALL
SELECT 'classify_jobs orphaned (photo missing)',
       COUNT(*)
  FROM classify_jobs cj
 WHERE NOT EXISTS (SELECT 1 FROM photos p WHERE p.id = cj.photo_id)
UNION ALL
SELECT 'hse_class_proposals with NULL user_id',
       COUNT(*)
  FROM hse_class_proposals
 WHERE user_id IS NULL
UNION ALL
SELECT 'daily_quota_usage qa_test_ rows',
       COUNT(*)
  FROM daily_quota_usage
 WHERE user_key LIKE 'qa_test_%';


-- =========================================================
-- STEP 2: DELETE — uncomment + run AFTER reviewing STEP 1 counts.
-- =========================================================
-- Cascade order: child rows first (corrections, classifications,
-- classify_jobs), then photos. Embeddings keyed on sha256 stay
-- (other photos may share the sha and find them useful).

-- BEGIN;
--
-- -- Wipe corrections that came from anonymous/test traffic.
-- DELETE FROM corrections
--  WHERE user_id IS NULL
--     OR user_key LIKE 'qa_test_%'
--     {cutoff_corrs};
--
-- -- Wipe proposals from anonymous/test traffic.
-- DELETE FROM hse_class_proposals
--  WHERE user_id IS NULL
--     OR proposed_by_user_key LIKE 'qa_test_%'
--     {cutoff_corrs};
--
-- -- Wipe quota usage rows (recomputed naturally on next upload).
-- DELETE FROM daily_quota_usage
--  WHERE user_id IS NULL
--     OR user_key LIKE 'qa_test_%'
--     {cutoff_corrs.replace('created_at', 'day')};
--
-- -- Wipe email_jobs from anonymous/test traffic.
-- DELETE FROM email_jobs
--  WHERE user_id IS NULL
--     OR user_key LIKE 'qa_test_%'
--     {cutoff_corrs};
--
-- -- Wipe classifications + classify_jobs whose photo we're about to drop.
-- WITH doomed_photos AS (
--   SELECT id FROM photos
--    WHERE user_id IS NULL
--       OR user_key LIKE 'qa_test_%'
--       {cutoff_clause}
-- )
-- DELETE FROM classifications WHERE photo_id IN (SELECT id FROM doomed_photos);
-- WITH doomed_photos AS (
--   SELECT id FROM photos
--    WHERE user_id IS NULL
--       OR user_key LIKE 'qa_test_%'
--       {cutoff_clause}
-- )
-- DELETE FROM classify_jobs WHERE photo_id IN (SELECT id FROM doomed_photos);
--
-- -- Finally drop the photos themselves.
-- DELETE FROM photos
--  WHERE user_id IS NULL
--     OR user_key LIKE 'qa_test_%'
--     {cutoff_clause};
--
-- -- Rebuild the materialised stats from scratch.
-- TRUNCATE daily_user_stats;
-- SELECT public.refresh_all_user_stats_v1();
--
-- COMMIT;


-- =========================================================
-- STEP 3: VERIFY — confirm wiped, stats refreshed.
-- =========================================================

-- SELECT COUNT(*) FROM daily_user_stats;
-- SELECT COUNT(*) FROM photos WHERE user_id IS NULL;
-- SELECT user_id, day, photos_uploaded, classifications, estimated_cost_usd
--   FROM daily_user_stats ORDER BY day DESC LIMIT 20;
"""
    print(sql)
    print()
    print("Workflow:")
    print("  1. Run STEP 1 to see counts. Verify nothing real is in there.")
    print("  2. If counts look right, uncomment STEP 2 and run.")
    print("  3. Run STEP 3 to verify the wipe + stats rebuild.")
    print()
    print("R2 storage:")
    print("  Photos in R2 corresponding to wiped DB rows are NOT auto-deleted.")
    print("  Run scripts/clean_orphan_r2.py separately if you also want to free")
    print("  the R2 bucket. Cheap to leave — R2 is cents per TB-month.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
