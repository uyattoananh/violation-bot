"""Print SQL for the users + sessions tables and the user_id columns
on existing per-user tables.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  Phase 5 of FUTURE_FEATURES.txt: Google OAuth login. Before today
  every per-user row was keyed by the cookie value (vai_uid).
  After today an authenticated user_id replaces the cookie key as
  the canonical identity, but the cookie key stays as the
  pre-auth fallback so:

    - logged-out visitors keep working exactly as before
    - admin stats stay continuous across the auth boundary
    - retro-attribution links a user's pre-auth rows to user_id on
      their first sign-in (matching photos.user_key etc. to the
      session's vai_uid value)

  Phase 7 adds a second provider (Azure Entra ID) — the schema is
  already provider-scoped so that's a config change, not a migration.

Usage:
  python scripts/add_users_and_sessions_tables.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

-- ---------- users ----------
-- One row per authenticated identity. Provider-scoped uniqueness on
-- (provider, provider_user_id) so the same email at Google vs Azure
-- creates two distinct user rows (Phase 7 ergonomic — a sign-in
-- button per provider, not a "merge accounts" UX).
CREATE TABLE IF NOT EXISTS users (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 'google' | 'azure' | 'email' (future). Populated on sign-up.
    provider           TEXT NOT NULL,
    -- Stable identifier from the provider. For Google: the 'sub' claim.
    -- For Azure: the 'oid' claim. Never the email — emails change.
    provider_user_id   TEXT NOT NULL,
    email              TEXT,
    name               TEXT,
    picture_url        TEXT,
    -- Set by SQL on the bootstrap admin user, or by another admin
    -- via a future /admin/users/{id}/promote endpoint. Used to
    -- replace the env-var ADMIN_PASSWORD gate at /admin/* eventually.
    is_admin           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at       TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_users_provider_subject
    ON users (provider, provider_user_id);

CREATE INDEX IF NOT EXISTS idx_users_email
    ON users (email);


-- ---------- sessions ----------
-- One row per active browser session. Cookie value is the `token`
-- column (random opaque string, never the user_id). Lookups are by
-- token; expired rows are not auto-deleted, just rejected at lookup.
-- Periodic cleanup is a future cron, not a v1 concern.
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 32-byte hex string set at sign-in time. The cookie value sent
    -- to the browser is exactly this; never expose user_id to the
    -- client side.
    token           TEXT NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Default 30 days from issuance — long enough for a typical
    -- inspector to keep returning, short enough to limit the blast
    -- radius if a session token leaks.
    expires_at      TIMESTAMPTZ NOT NULL,
    last_used_at    TIMESTAMPTZ,
    -- Best-effort attribution; use only as a debugging hint, not
    -- for any security decision.
    user_agent      TEXT,
    ip              TEXT
);

-- Look up by token on every authenticated request. Token has a unique
-- index; this composite covers the freshness check inline.
CREATE INDEX IF NOT EXISTS idx_sessions_token_expires
    ON sessions (token, expires_at);

-- Background cleanup query: delete sessions where expires_at < now.
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
    ON sessions (expires_at);


-- ---------- user_id columns on per-user tables ----------
-- These columns are populated on first sign-in (retro-attribute) and
-- on every subsequent insert. user_key stays as the pre-auth fallback
-- and is never cleared, so admin queries can JOIN on either column.

ALTER TABLE photos
    ADD COLUMN IF NOT EXISTS user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE corrections
    ADD COLUMN IF NOT EXISTS user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE hse_class_proposals
    ADD COLUMN IF NOT EXISTS user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE daily_quota_usage
    ADD COLUMN IF NOT EXISTS user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE email_jobs
    ADD COLUMN IF NOT EXISTS user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL;

-- Indexes for "give me everything user X did" admin queries.
CREATE INDEX IF NOT EXISTS idx_photos_user_id
    ON photos (user_id) WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_corrections_user_id
    ON corrections (user_id) WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_proposals_user_id
    ON hse_class_proposals (user_id) WHERE user_id IS NOT NULL;


-- ---------- RLS ----------
-- All tables are backend-only — webapp uses the service-role key
-- which bypasses RLS. ENABLE without policies = block anon /
-- authenticated entirely (defence in depth in case the service-role
-- key ever leaks somewhere unexpected).
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
"""
    print(sql)
    print()
    print("After applying the SQL:")
    print("  - Configure GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET on VPS .env")
    print("  - Phase 5b webapp code wires up /auth/login/google + /auth/callback/google")
    print("  - First sign-in retro-attributes the user's cookie-keyed rows")
    print("  - Promote your own user row to admin once you've signed in:")
    print("      UPDATE users SET is_admin = TRUE WHERE email = 'lang@aecis.com';")
    return 0


if __name__ == "__main__":
    sys.exit(main())
