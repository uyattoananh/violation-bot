"""Print SQL for the hse_class_proposals table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  Phase 4 of FUTURE_FEATURES.txt: inspectors can propose a new fine
  HSE sub-type that's not in the AECIS canonical taxonomy when the
  correct-region modal doesn't show a fitting option. Admin reviews
  pending proposals at /admin/proposals; on approval the slug joins
  data/fine_hse_types_by_parent.json (hot-reloaded into app.state)
  and any corrections that referenced the proposal's pending slug
  get retro-patched to the approved slug.

  Identity is cookie-keyed (vai_uid) until Phase 5 ships Google OAuth.
  At that point the migration script in Phase 5 retro-attributes
  proposed_by_user_key to the matching users.id row.

Usage:
  python scripts/add_hse_class_proposals_table.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

CREATE TABLE IF NOT EXISTS hse_class_proposals (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Cookie-keyed identity until Phase 5 (OAuth) ships.
    proposed_by_user_key     TEXT NOT NULL,
    -- Parent the new sub-type slots under, e.g. 'Mass_piling_unsafe'.
    -- Must be a known parent slug at submit time; server validates.
    parent_slug              TEXT NOT NULL,
    -- Auto-generated from label_en: lowercased, non-alnum -> '_',
    -- truncated to 80 chars, suffixed with -2/-3/... on collision.
    -- Stored as 'pending:<short>' until the proposal is approved,
    -- at which point it becomes the final slug and replaces the
    -- placeholder on any corrections that referenced it.
    proposed_slug            TEXT NOT NULL,
    -- Both EN + VN required (the inspector might be using either
    -- locale, and the prompt builder needs both for the model).
    label_en                 TEXT NOT NULL,
    label_vn                 TEXT NOT NULL,
    -- Inspector's free-text rationale ("why is this class needed").
    -- 280-char soft limit enforced server-side, hard cap 1000 here.
    description              TEXT,
    -- Optional pointer at the photo the inspector was looking at
    -- when they hit "propose new". Single photo for v1; later we
    -- can let admins add more example_photo_ids[] during triage.
    example_photo_id         UUID NULL REFERENCES photos(id) ON DELETE SET NULL,
    -- 'pending' | 'approved' | 'rejected' | 'duplicate'
    -- 'duplicate' is operationally a rejection but admins want to
    -- distinguish "this is the same as existing slug X" from
    -- "this isn't a real HSE concept" for triage stats.
    status                   TEXT NOT NULL DEFAULT 'pending',
    -- Admin's note shown to the user when they next sign in.
    -- For 'duplicate' the admin sets this to the matching slug.
    reviewer_note            TEXT,
    -- Final slug after approval (populated on approve only). Lets
    -- us keep the original proposed_slug for audit + show the
    -- inspector the slug name they ended up with.
    approved_slug            TEXT,
    -- Whether the user has seen the decision toast on their next
    -- visit. Toggled to true on first GET /api/proposals/decisions.
    notified_user            BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by              TEXT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at              TIMESTAMPTZ NULL
);

-- Per-user daily throttle: count today's submissions by user_key.
CREATE INDEX IF NOT EXISTS idx_proposals_user_day
    ON hse_class_proposals (proposed_by_user_key, created_at DESC);

-- Admin pending queue: small partial index, kept warm.
CREATE INDEX IF NOT EXISTS idx_proposals_pending
    ON hse_class_proposals (created_at)
    WHERE status = 'pending';

-- One-time decision toast lookup: find decided proposals by user
-- that haven't been shown yet. Tiny set in steady state.
CREATE INDEX IF NOT EXISTS idx_proposals_unnotified
    ON hse_class_proposals (proposed_by_user_key)
    WHERE status <> 'pending' AND notified_user = FALSE;

-- Slug uniqueness across the proposed-slug space. Approved slugs
-- get checked against the live taxonomy at approve-time, not here.
CREATE UNIQUE INDEX IF NOT EXISTS uq_proposals_proposed_slug
    ON hse_class_proposals (proposed_slug);

-- Backend-only access — webapp uses service-role key which bypasses
-- RLS. ENABLE without policies = block anon/authenticated entirely.
ALTER TABLE hse_class_proposals ENABLE ROW LEVEL SECURITY;
"""
    print(sql)
    print()
    print("After applying the SQL:")
    print("  - Phase 4b/c webapp code reads/writes this table")
    print("  - 'Propose new sub-type' button appears in correct-region modal")
    print("  - /admin/proposals page surfaces pending review queue")
    print("  - Approval appends to data/fine_hse_types_by_parent.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
