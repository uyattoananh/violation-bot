"""Print SQL for the daily-quota + per-user-stats tables.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  - Phase 2 of FUTURE_FEATURES.txt: cap free-tier users at 30 photos
    per UTC day, and roll up per-user usage stats for an admin
    dashboard. Both keyed by `user_key` — a stable identifier set as
    a cookie until real auth (Azure AD) ships.

When auth lands later, a one-shot migration aliases the cookie-keys
to real user.id values and the same tables keep working.

Usage:
  python scripts/add_quota_and_stats_tables.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- Per-user, per-day classification quota (30 photos/day on free tier).
-- Incremented on every successful classify_jobs insert (= one row per
-- photo through /api/upload). Reset implicit at UTC midnight.
CREATE TABLE IF NOT EXISTS daily_quota_usage (
    user_key             TEXT NOT NULL,
    day                  DATE NOT NULL,
    photos_classified    INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (user_key, day)
);
-- Used to look up "did this user hit their cap today" on every upload.
CREATE INDEX IF NOT EXISTS idx_daily_quota_user
    ON daily_quota_usage (user_key, day DESC);

-- Per-user, per-day rolled-up usage stats. Populated either lazily
-- (on /admin/stats fetch) or via a nightly cron job. Reads from the
-- existing photos + corrections + classifications tables, no
-- duplicated source-of-truth.
CREATE TABLE IF NOT EXISTS daily_user_stats (
    day                  DATE NOT NULL,
    user_key             TEXT NOT NULL,
    photos_uploaded      INT  NOT NULL DEFAULT 0,
    photos_reviewed      INT  NOT NULL DEFAULT 0,
    confirms             INT  NOT NULL DEFAULT 0,
    corrections_count    INT  NOT NULL DEFAULT 0,
    estimated_cost_usd   DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (day, user_key)
);
CREATE INDEX IF NOT EXISTS idx_daily_user_stats_day
    ON daily_user_stats (day DESC);

-- Augment photos + corrections with the cookie-key. Lets us attribute
-- existing rows to the new key namespace AND lets the rollup query
-- group correctly. Until backfill, NULL means "pre-Phase-2 row".
ALTER TABLE photos      ADD COLUMN IF NOT EXISTS user_key TEXT NULL;
ALTER TABLE corrections ADD COLUMN IF NOT EXISTS user_key TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_photos_user_key
    ON photos (user_key, uploaded_at DESC) WHERE user_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_corrections_user_key
    ON corrections (user_key, created_at DESC) WHERE user_key IS NOT NULL;
"""
    print(sql)
    print()
    print("Copy the SQL above into the Supabase SQL editor and run it.")
    print()
    print("After the migration:")
    print("  - Quota check on /api/upload becomes active (defaults: 30/day)")
    print("  - Footer ticker shows N/30 photos used today")
    print("  - /admin/stats route reads from daily_user_stats (lazy-rolled-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
