"""Postgres function for atomic quota increment.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  _quota_increment in webapp.app does select-then-upsert. Two
  parallel uploads from the same user_key + day can race:
    T0  thread A reads photos_classified=10
    T1  thread B reads photos_classified=10
    T2  thread A upserts 11
    T3  thread B upserts 11    <- B's increment was lost
  Net effect: +1 instead of +2, and a determined user with N
  parallel uploads can effectively bypass the daily cap.

  This RPC does the increment in ONE atomic statement using
  Postgres' INSERT ... ON CONFLICT DO UPDATE with a RETURNING
  clause, eliminating the race. The webapp calls it via
  db.rpc('quota_increment_v1', {...}).execute().

Usage:
  python scripts/add_quota_increment_rpc.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent (CREATE OR REPLACE).

CREATE OR REPLACE FUNCTION public.quota_increment_v1(
    p_user_key TEXT,
    p_day      DATE,
    p_n        INT
)
RETURNS INT AS $$
DECLARE
    v_total INT;
BEGIN
    -- Single atomic statement. The ON CONFLICT clause increments the
    -- existing row's photos_classified by p_n in one round-trip,
    -- holding the row lock just for the UPDATE. RETURNING surfaces
    -- the post-increment value so the caller can react (e.g. honour
    -- the quota cap on the SAME request that ticked it over).
    INSERT INTO daily_quota_usage (user_key, day, photos_classified)
    VALUES (p_user_key, p_day, p_n)
    ON CONFLICT (user_key, day) DO UPDATE
        SET photos_classified =
            daily_quota_usage.photos_classified + EXCLUDED.photos_classified
    RETURNING photos_classified INTO v_total;
    RETURN v_total;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Lock down execution to the service role; this function bypasses
-- RLS via SECURITY DEFINER so we don't want anon/authenticated to
-- be able to call it directly. The webapp's service-role key has
-- usage rights by default.
REVOKE ALL ON FUNCTION public.quota_increment_v1(TEXT, DATE, INT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.quota_increment_v1(TEXT, DATE, INT)
    TO service_role;

COMMENT ON FUNCTION public.quota_increment_v1 IS
    'Atomic quota increment. Replaces the select-then-upsert pattern '
    'that had a race window allowing parallel uploads to bypass the '
    'daily cap. Called from webapp _quota_increment().';
"""
    print(sql)
    print()
    print("After applying:")
    print("  - webapp/app.py _quota_increment switches to db.rpc(...)")
    print("  - the race window in select-then-upsert is gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
