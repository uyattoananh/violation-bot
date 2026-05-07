# HSE Detector — project brief for Claude

Two-axis classifier for AECIS construction-site safety photos. Inspector
uploads a photo, the model returns a `(location, hse_type)` pair plus a
top-3 alternative list. Per-photo cost ~$0.014; held-out 100-photo eval:
65.7% top-1 / 84.8% top-3.

## Layout

- `webapp/app.py` — FastAPI server. ~6,200 lines; one file by design,
  not a microservice. Owns auth, batches, photos, R2 presigning,
  cleanup, /api/* surface.
- `webapp/worker.py` — background worker that drains `classify_jobs`
  via Supabase polling. Calls `src/zero_shot.py`.
- `webapp/templates/index.html` — the entire SPA in one Jinja
  template. Whole script body wrapped in `DOMContentLoaded`, so
  top-level functions are NOT on `window` (matters when probing
  via Playwright — drive the user paths instead of calling JS directly).
- `webapp/templates/_base.html` — shell, header, modals, toast.
- `webapp/static/app.css` — design tokens + custom CSS. Tailwind
  loads from CDN as a complement, not the source of truth.
- `webapp/static/service-worker.js` — bump `CACHE_VERSION` on every
  release that changes rendered HTML/JS, otherwise returning users
  read the old shell from cache.
- `src/zero_shot.py` — classification core: CLIP embed → pgvector
  k-NN → Sonnet 4.5 prompt with neighbour hints → parse JSON top-3.
- `scripts/` — flat folder of one-off DB migrations, seeders,
  audits. Each script names what it does.

Storage: Supabase Postgres + pgvector for embeddings + corrections;
Cloudflare R2 for photo blobs (presigned URLs).
Provider: OpenRouter → `anthropic/claude-sonnet-4.5` with prompt caching.

## Local dev

```
./.venv-webapp/Scripts/python.exe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8765
```

Set `AUTH_REQUIRED=0` to skip Google/Microsoft sign-in and land
directly in the app shell — needed for Playwright verifiers.

Sanity check after template/handler edits:
`./.venv-webapp/Scripts/python.exe scripts/test_html_export.py`

## Deploy

Service runs on VPS at `/root/violation-bot` under
`violation-webapp.service` (uvicorn) and `violation-worker.service`.
Note: there's a stale `/var/www/violation-bot` checkout — don't
deploy there, the running service ignores it.

```
ssh vps 'cd /root/violation-bot && sudo git pull origin <branch> && sudo systemctl restart violation-webapp.service'
```

Confirm the deploy with:
`curl -s "https://hse.aecis.ca/static/service-worker.js?cb=$(date +%s)" | grep CACHE_VERSION`

## Conventions worth keeping

- Bump `CACHE_VERSION` in service-worker.js on every shipped commit.
- Photos and batches expire on `_PHOTO_EXPIRY_DAYS` (currently 2).
  `_cleanup_expired_batches()` wipes whole batches once any photo
  hits cutoff; `_cleanup_expired_photos()` mops up unbatched
  stragglers. Both run hourly via `_maybe_cleanup()` from
  `/api/batches`, plus once on startup.
- Toast-undo for destructive single-item actions (Gmail/iOS Mail
  pattern, 5 s). Bulk destructive actions get a real modal — see
  `#bulk-delete-modal`.
- Atkinson Hyperlegible loaded from Google Fonts; the font ships
  weights 400 + 700 only, so font-medium/semibold are remapped to
  700 in app.css. Don't introduce a third weight.
- Construction palette: slate-50 / slate-900 / safety-orange #F97316
  + emerald-700 / amber-700 / rose-700. Off-scale Tailwind shades
  are aliased back to the canonical trio in app.css.
- Active selection state = emerald everywhere (toggle button, ring,
  ✓, thumb wash). Orange is reserved for primary CTAs.

## Roadmap — next focus

### 1. MS SQL ingestion (read path)

Pull lookups from AECIS's MS SQL Server — the inverse of the
already-drafted MS SQL export pipeline (see
`MS_SQL_EXPORT_QUESTIONS.txt` for the open decision tree). Goal: use
the AECIS-side authoritative data (projects, sites, users, possibly
historical inspections) as a richer source than the curated
`data/fine_hse_types_by_parent.json`.

Open before code:
- Hosting (on-prem vs Azure SQL) — drives the connector + auth.
- Cadence — most likely nightly batch into Supabase, but could be
  on-demand for picker lookups.
- Schema mapping — AECIS columns to our Supabase tables. Don't
  blindly mirror; we keep the slugs we've trained on.

When implementing, default to a worker-style script in `scripts/`
(`ingest_mssql_<table>.py`) that writes to Supabase. Use
`pyodbc` or `pymssql`; document the ODBC driver required in the
script header.

### 2. Backdoor seed path that doesn't pollute stats

The accuracy figures (65.7% top-1 etc.) and the per-user quota /
training-set counters all aggregate over the production `photos`
table. Ingesting bulk MS SQL data through the same path would
double-count seed data into:

- the eval split (taints held-out evaluation)
- the per-user stats (inspectors see inflated "training set size")
- per-batch reviewed/total counts in the inspector UI

Build a sidechannel insert path that bypasses these:

- A `photos.source` enum (or boolean `is_seed`) marking
  programmatically-seeded rows.
- An `_admin_authed` POST endpoint or CLI script that takes
  pre-classified rows from MS SQL, inserts with `is_seed = true`,
  and writes the embedding straight into pgvector — without going
  through the worker queue, so we don't touch quotas.
- Every read site that currently reports stats — `/api/usage/today`,
  `/api/batches`, the eval scripts in `scripts/ai_audit_*.py`,
  the picker's "training set size" hint — needs to filter
  `is_seed = false` to keep production numbers honest. The seed rows
  CAN still appear in the embedding k-NN (that's the point — they
  improve retrieval) but they shouldn't appear in stats or in any
  inspector-facing list.

Treat this as a schema migration first
(`scripts/add_is_seed_column.py`), then a bulk-insert tool, then a
follow-up audit pass to add the `is_seed = false` filter to every
counter. The audit is the part that's easy to forget — keep a
checklist of read sites in the migration script header so it's
clear which queries still need the filter applied.
