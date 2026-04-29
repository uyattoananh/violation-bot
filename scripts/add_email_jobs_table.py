"""Print SQL for the email_jobs table.

Run the printed SQL in the Supabase SQL editor. Idempotent.

Why:
  Phase 3 of FUTURE_FEATURES.txt: inspector clicks "Email this batch"
  in the export menu, gets the export delivered to their email
  (or a recipient they specify) instead of downloading. Each request
  becomes a row in email_jobs for audit + rate-limiting purposes.

  The actual email send is INLINE in the request handler (synchronous)
  for v1 — fits an inspector's "click and wait a few seconds" mental
  model. If batches grow huge later, refactor to a background worker.

Usage:
  python scripts/add_email_jobs_table.py
"""
from __future__ import annotations
import sys


def main() -> int:
    sql = """\
-- Run in Supabase SQL editor. Idempotent.

CREATE TABLE IF NOT EXISTS email_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Cookie-keyed identity until real auth (Phase 5) ships.
    user_key        TEXT NOT NULL,
    -- Batch being exported. NULL means "all photos in tenant" — not
    -- supported yet, reserved for future "everything I have" exports.
    batch_id        UUID NULL,
    -- 'pdf' | 'zip' | 'csv' | 'json' | 'html'  (matches the existing
    -- /api/export/* download formats). Server validates against the
    -- known list before generating.
    format          TEXT NOT NULL,
    to_email        TEXT NOT NULL,
    -- Optional inspector-supplied subject + body. Defaults applied
    -- server-side when blank.
    subject         TEXT,
    body            TEXT,
    -- 'pending' | 'sending' | 'sent' | 'failed' | 'too_large_emailed_link'
    -- The 'too_large_emailed_link' status is for batches whose generated
    -- file exceeded the inline-attachment limit (~20MB) and were
    -- delivered as a presigned R2 URL instead.
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    attempts        INT  NOT NULL DEFAULT 0,
    -- Size of the generated export blob in bytes — useful for capacity
    -- planning + diagnosing whether large batches consistently fail.
    blob_bytes      INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at         TIMESTAMPTZ NULL
);

-- Index for the per-user daily rate-limit lookup (count today's
-- emails by user_key). Most queries hit a small recent slice so a
-- DESC sort on created_at is the natural order.
CREATE INDEX IF NOT EXISTS idx_email_jobs_user_day
    ON email_jobs (user_key, created_at DESC);

-- Index for any future background-worker that polls pending rows.
CREATE INDEX IF NOT EXISTS idx_email_jobs_pending
    ON email_jobs (status, created_at)
    WHERE status = 'pending';

-- Backend-only access — webapp uses service-role key which bypasses
-- RLS. ENABLE without policies = block anon/authenticated entirely.
ALTER TABLE email_jobs ENABLE ROW LEVEL SECURITY;
"""
    print(sql)
    print()
    print("After applying the SQL:")
    print("  - Configure SMTP env vars on the VPS .env (see .env.example)")
    print("  - Restart violation-webapp")
    print("  - Test by clicking the new 'Email it' button in the export menu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
