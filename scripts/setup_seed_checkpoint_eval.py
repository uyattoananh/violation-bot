"""Print SQL to wire up the seed-with-checkpoint evaluation harness.

Goal:
  Seed photos in batches of 200 -- after each batch, evaluate the
  classifier against a FIXED set of 100 production photos that
  NEVER enter k-NN retrieval. This produces a learning curve:
  accuracy as a function of seed count. If the seeded photos help,
  accuracy goes up; if their labels are noisy, accuracy goes down.
  The 100 holdouts stay constant across iterations so the test is
  comparable.

Run the printed SQL ONCE in the Supabase SQL editor before the
first seed batch. Idempotent.

Usage:
  python scripts/setup_seed_checkpoint_eval.py

================================================================
WHAT THE SQL DOES

1. Adds `is_holdout BOOLEAN` to photo_embeddings.
2. Updates match_photo_embeddings() so holdouts are excluded from
   k-NN retrieval. Without this, the model would "find itself" in
   the reference index and every eval prediction would be a
   trivially perfect distance=0 match.
3. Picks 100 stratified-by-hse_type production photos and tags
   them as holdouts. Stratification keeps low-frequency classes
   from being dropped entirely.
4. Creates a seed_progress view for at-a-glance status: how many
   embeddings per label_source, how many are holdouts.

The seed_label_source convention this expects:
  manual                       -- production inspector corrections
  aecis_seed_v1_batch_001      -- first 200 photos from the assigner
  aecis_seed_v1_batch_002      -- next 200
  ...
  aecis_seed_v1_batch_012      -- final partial batch (~157 photos)

================================================================
WORKFLOW

  1. Paste the printed SQL in Supabase SQL editor. Run it.
  2. Verify holdout count: should be ~100, spread across hse_types.
     Re-run section (5) of the SQL to inspect.
  3. Apply scripts/add_is_seed_column.py too if you haven't yet.
  4. Run scripts/seed_with_checkpoints.py to drive the loop.
     That script:
        - chunks Issue_Gen/photos/manifest.jsonl into 200-photo batches
        - for each batch, uploads via the local UI (Playwright) with
          seed_label_source = 'aecis_seed_v1_batch_NNN'
        - waits for the worker to embed them
        - runs the held-out 100 against the now-larger k-NN index
        - appends a row to tmp/seed_accuracy_curve.csv:
             batch_n, n_seeds_total, top1_acc, top3_acc, mean_conf
  5. Watch the CSV. If top1_acc trends up, the seeds are helping.
     If it trends down or stays flat with high variance, the
     seed labels are probably noisy and should be filtered tighter.

================================================================
ROLLBACK

If a particular seed batch tanks the metric, remove just that batch:

    DELETE FROM photos          WHERE seed_label_source = 'aecis_seed_v1_batch_007';
    DELETE FROM photo_embeddings WHERE label_source     = 'aecis_seed_v1_batch_007';

The held-out 100 stay reserved across re-runs because `is_holdout`
is sticky -- only the embeddings of seed batches get rolled back.
"""
from __future__ import annotations
import sys


SQL = """\
-- Run in Supabase SQL editor. Idempotent.
-- One-time setup for the seed-with-checkpoint evaluation harness.

-- ----------------------------------------------------------------
-- 1. Holdout flag on photo_embeddings
-- ----------------------------------------------------------------
-- TRUE = this embedding is reserved for evaluation only; it must
-- NOT appear in k-NN retrieval. We tag at the embedding level
-- (not the photos table) because match_photo_embeddings is what
-- the classifier actually walks.
ALTER TABLE photo_embeddings
  ADD COLUMN IF NOT EXISTS is_holdout BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS photo_embeddings_holdout_idx
  ON photo_embeddings(is_holdout) WHERE is_holdout = TRUE;

-- ----------------------------------------------------------------
-- 2. Patch the k-NN RPC to skip holdouts
-- ----------------------------------------------------------------
-- Same shape as supabase/migrations/01_photo_rag.sql defined
-- originally; the only change is the extra AND NOT is_holdout
-- clause. Without this clause the eval would find each holdout
-- as its own perfect distance=0 neighbour.
CREATE OR REPLACE FUNCTION match_photo_embeddings(
    query_embedding vector(512),
    match_k int default 5
)
RETURNS TABLE (
    sha256 text,
    hse_type_slug text,
    location_slug text,
    project_code text,
    issue_id text,
    distance float
) LANGUAGE sql STABLE AS $$
    SELECT
        sha256, hse_type_slug, location_slug, project_code, issue_id,
        (embedding <=> query_embedding) AS distance
    FROM photo_embeddings
    WHERE embedding IS NOT NULL
      AND NOT is_holdout
    ORDER BY embedding <=> query_embedding
    LIMIT match_k;
$$;

-- ----------------------------------------------------------------
-- 3. Sample 100 stratified holdouts from the production set
-- ----------------------------------------------------------------
-- We pull from label_source = 'manual' so the holdouts are
-- inspector-validated labels (the gold standard) rather than any
-- previously-seeded data. Stratifying by hse_type_slug means rare
-- classes get represented even if they have <10 photos total.
--
-- IDEMPOTENT: re-running this won't add new holdouts (we filter on
-- NOT is_holdout). To reshuffle the holdout set, clear it first:
--     UPDATE photo_embeddings SET is_holdout = FALSE;
-- Then re-run.
WITH eligible AS (
    SELECT sha256, hse_type_slug
    FROM photo_embeddings
    WHERE label_source = 'manual'
      AND hse_type_slug IS NOT NULL
      AND NOT is_holdout
),
n_classes AS (
    SELECT COUNT(DISTINCT hse_type_slug) AS n FROM eligible
),
ranked AS (
    SELECT
        sha256,
        hse_type_slug,
        ROW_NUMBER() OVER (
            PARTITION BY hse_type_slug
            ORDER BY random()
        ) AS rn
    FROM eligible
),
per_class AS (
    SELECT sha256
    FROM ranked, n_classes
    WHERE rn <= GREATEST(1, CEIL(100.0 / NULLIF(n_classes.n, 0)))
    LIMIT 100
)
UPDATE photo_embeddings
SET is_holdout = TRUE
WHERE sha256 IN (SELECT sha256 FROM per_class);

-- ----------------------------------------------------------------
-- 4. Verify holdout count + distribution
-- ----------------------------------------------------------------
-- Expect ~100 total, spread across the hse_type vocabulary.
SELECT
    hse_type_slug,
    COUNT(*) AS n_holdout
FROM photo_embeddings
WHERE is_holdout = TRUE
GROUP BY 1
ORDER BY n_holdout DESC, hse_type_slug;

-- ----------------------------------------------------------------
-- 5. Progress view: at-a-glance per-source counts
-- ----------------------------------------------------------------
-- Re-query this view after each seed batch to confirm the new
-- rows are landing and the holdout set hasn't drifted.
--   SELECT * FROM seed_progress;
CREATE OR REPLACE VIEW seed_progress AS
SELECT
    COALESCE(label_source, '(null)')     AS source,
    COUNT(*)                              AS n_embeddings,
    COUNT(*) FILTER (WHERE is_holdout)    AS n_holdout,
    MAX(created_at)                       AS last_inserted
FROM photo_embeddings
GROUP BY 1
ORDER BY 1;
"""


def main() -> int:
    print(SQL)
    print()
    print("After running the SQL above:")
    print("  - confirm section 4 reports ~100 holdouts across multiple hse_types")
    print("  - confirm `SELECT * FROM seed_progress;` shows 'manual' (production")
    print("    embeddings) + 1 holdout row")
    print("  - then run scripts/seed_with_checkpoints.py to start the seed loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
