"""Print SQL for the per-user-stats redesign.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  The original daily_user_stats table was provisioned in Phase 2
  but never populated. It was keyed by `user_key` (cookie value),
  which made it impossible to distinguish:
    - real authenticated inspectors
    - pre-auth visitors with cookies
    - QA probe test data (qa_test_* prefix)
    - rows ticked during deployment / model training

  The redesign:
    - keys on user_id (only authenticated users land here)
    - rows for anonymous / pre-auth / test traffic NEVER get
      created, so deployment + training noise is excluded by
      construction
    - adds classifications + estimated_cost_usd so admin can see
      per-user spend without re-aggregating raw classifications

  Aggregation lives in webapp/app.py:_refresh_user_stats(),
  called lazily by /admin (cheap; touches one (user_id, day) row
  per active user per request).

Usage:
  python scripts/add_per_user_stats.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- Drop the legacy placeholder table — it was provisioned in Phase 2
-- but never populated, so nothing depends on its data.
DROP TABLE IF EXISTS public.daily_user_stats CASCADE;

CREATE TABLE public.daily_user_stats (
    user_id              UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day                  DATE NOT NULL,
    -- Primary counters; one tick per matching raw event.
    photos_uploaded      INT  NOT NULL DEFAULT 0,
    classifications      INT  NOT NULL DEFAULT 0,
    confirms             INT  NOT NULL DEFAULT 0,
    corrections_count    INT  NOT NULL DEFAULT 0,
    proposals_submitted  INT  NOT NULL DEFAULT 0,
    -- Spend on this user's classifications, summed from
    -- classifications.input_tokens + output_tokens times the public
    -- list rate per model. Approximate to ~1% of OpenRouter's
    -- billing dashboard.
    estimated_cost_usd   DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Refresh tracking — admin panel re-aggregates if last_refreshed
    -- is older than the day's end (i.e. data could still be moving).
    last_refreshed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, day)
);

CREATE INDEX IF NOT EXISTS idx_daily_user_stats_day
    ON public.daily_user_stats (day DESC);

-- Backend-only access. Webapp uses service-role; RLS blocks anon.
ALTER TABLE public.daily_user_stats ENABLE ROW LEVEL SECURITY;


-- Aggregation function — refreshes the row for one (user_id, day).
-- Does an INSERT ... ON CONFLICT DO UPDATE so it's safe to call
-- repeatedly; idempotent even under concurrent calls (each call
-- recomputes from the source-of-truth raw tables).
CREATE OR REPLACE FUNCTION public.refresh_user_stats_v1(
    p_user_id UUID,
    p_day     DATE
) RETURNS public.daily_user_stats AS $$
DECLARE
    v_uploaded     INT;
    v_classified   INT;
    v_confirms     INT;
    v_corrects     INT;
    v_proposals    INT;
    v_cost         DOUBLE PRECISION;
    v_row          public.daily_user_stats;
BEGIN
    -- Photos uploaded by this user on this day.
    SELECT COUNT(*) INTO v_uploaded
      FROM photos
     WHERE user_id = p_user_id
       AND uploaded_at >= p_day::timestamp
       AND uploaded_at <  p_day::timestamp + INTERVAL '1 day';

    -- Classifications + cost: classifications has photo_id (FK to
    -- photos). Join to scope by user. Cost uses a rough flat rate
    -- of $0.0008 per 1k input tokens + $0.004 per 1k output tokens
    -- (gemini-2.5-flash list price; refined below if model tagged).
    SELECT
        COUNT(*),
        COALESCE(SUM(
            (c.input_tokens  / 1e6) * CASE
                WHEN c.model LIKE '%opus%'  THEN 15.0
                WHEN c.model LIKE '%sonnet%' THEN  3.0
                WHEN c.model LIKE '%haiku%'  THEN  0.8
                WHEN c.model LIKE '%pro%'    THEN  1.25
                ELSE 0.075   -- gemini flash & similar default
            END
          + (c.output_tokens / 1e6) * CASE
                WHEN c.model LIKE '%opus%'  THEN 75.0
                WHEN c.model LIKE '%sonnet%' THEN 15.0
                WHEN c.model LIKE '%haiku%'  THEN  4.0
                WHEN c.model LIKE '%pro%'    THEN 10.0
                ELSE 0.30
            END
        ), 0)
      INTO v_classified, v_cost
      FROM classifications c
      JOIN photos p ON p.id = c.photo_id
     WHERE p.user_id = p_user_id
       AND c.created_at >= p_day::timestamp
       AND c.created_at <  p_day::timestamp + INTERVAL '1 day';

    -- Confirms / corrects this user MADE on this day (the audit
    -- trail records the actor on each correction row, which may
    -- differ from the photo's uploader).
    SELECT
        COUNT(*) FILTER (WHERE action = 'confirm'),
        COUNT(*) FILTER (WHERE action = 'correct')
      INTO v_confirms, v_corrects
      FROM corrections
     WHERE user_id = p_user_id
       AND created_at >= p_day::timestamp
       AND created_at <  p_day::timestamp + INTERVAL '1 day';

    -- Proposals (HSE class) submitted by this user on this day.
    SELECT COUNT(*) INTO v_proposals
      FROM hse_class_proposals
     WHERE user_id = p_user_id
       AND created_at >= p_day::timestamp
       AND created_at <  p_day::timestamp + INTERVAL '1 day';

    INSERT INTO daily_user_stats (
        user_id, day, photos_uploaded, classifications, confirms,
        corrections_count, proposals_submitted, estimated_cost_usd,
        last_refreshed_at
    ) VALUES (
        p_user_id, p_day, v_uploaded, v_classified, v_confirms,
        v_corrects, v_proposals, v_cost, NOW()
    )
    ON CONFLICT (user_id, day) DO UPDATE SET
        photos_uploaded     = EXCLUDED.photos_uploaded,
        classifications     = EXCLUDED.classifications,
        confirms            = EXCLUDED.confirms,
        corrections_count   = EXCLUDED.corrections_count,
        proposals_submitted = EXCLUDED.proposals_submitted,
        estimated_cost_usd  = EXCLUDED.estimated_cost_usd,
        last_refreshed_at   = NOW()
    RETURNING * INTO v_row;

    RETURN v_row;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE ALL ON FUNCTION public.refresh_user_stats_v1(UUID, DATE)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.refresh_user_stats_v1(UUID, DATE)
    TO service_role;


-- Bulk backfill: populates daily_user_stats for every (user_id, day)
-- pair that has any activity. Run ONCE after applying the migration.
-- Idempotent — re-running just re-aggregates the same data.
CREATE OR REPLACE FUNCTION public.refresh_all_user_stats_v1()
RETURNS INT AS $$
DECLARE
    r RECORD;
    v_count INT := 0;
BEGIN
    FOR r IN
        SELECT DISTINCT user_id, uploaded_at::date AS day
          FROM photos
         WHERE user_id IS NOT NULL
        UNION
        SELECT DISTINCT user_id, created_at::date AS day
          FROM corrections
         WHERE user_id IS NOT NULL
        UNION
        SELECT DISTINCT user_id, created_at::date AS day
          FROM hse_class_proposals
         WHERE user_id IS NOT NULL
    LOOP
        PERFORM public.refresh_user_stats_v1(r.user_id, r.day);
        v_count := v_count + 1;
    END LOOP;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

REVOKE ALL ON FUNCTION public.refresh_all_user_stats_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.refresh_all_user_stats_v1()
    TO service_role;
"""
    print(sql)
    print()
    print("After applying:")
    print("  - Run ONCE to backfill: SELECT public.refresh_all_user_stats_v1();")
    print("  - The webapp /admin endpoint will lazy-refresh today's stats")
    print("    on each visit and read from daily_user_stats for the rest.")
    print("  - To wipe legacy/test data, see scripts/wipe_test_stats.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
