# Construction Violation AI

Two-axis violation classifier for Vietnamese construction sites. Given a site
photograph it returns:

- **location** — from a prepared vocabulary (e.g. `Working at height`, `Scaffolding and Platform`)
- **hse_type** — from a prepared vocabulary (e.g. `No_access_unsafe_access`, `Lack_of_guarding_on_moving_parts`)

The two vocabularies are extracted directly from scraped AECIS DTag metadata —
the same labels inspectors already pick from in the field.

---

## Two phases, one codebase

| | Phase 1 (zero-shot) | Phase 2 (fine-tuned) |
|---|---|---|
| **When** | Day 1 — no training data needed | After ~2.5k corrected labels |
| **Inference** | Anthropic Claude Sonnet 4.5 (vision) | HuggingFace AutoTrain classifiers (ViT) + Sonnet fallback |
| **Cost / image** | ~$0.005 (batch API) / $0.010 (live) | ~$0.001 with fallback |
| **Latency** | 2-5s | 200ms + occasional 2-5s fallback |
| **Code** | `src/zero_shot.py`, `webapp/` | `src/train.py`, `src/predict.py`, `src/server.py` |

You ship Phase 1 now. Inspector corrections flow into a training table.
When you have enough labels, trigger Phase 2 — no architectural change.

## Repository layout

```
.
├── taxonomy.json               # canonical prepared vocabularies (locations + hse_types)
├── taxonomy.md                 # human-readable summary
├── prompts/classifier.md       # reference prompt + design notes
├── src/
│   ├── zero_shot.py            # Phase-1 Sonnet classifier
│   ├── taxonomy.py             # Phase-2 hardcoded taxonomy (legacy; use taxonomy.json instead)
│   ├── train.py evaluate.py    # Phase-2 PyTorch training
│   ├── predict.py server.py    # Phase-2 FastAPI inference
│   └── dataset.py model.py
├── webapp/
│   ├── app.py                  # FastAPI + Jinja UI
│   ├── worker.py               # queue consumer for classify_jobs
│   ├── templates/              # Tailwind + HTMX HTML
│   └── static/
├── supabase/
│   └── schema.sql              # Postgres schema (tenants, photos, classifications, corrections)
├── scripts/
│   ├── extract_taxonomy.py     # build taxonomy.json from scraped AECIS data
│   ├── seed_taxonomy_to_supabase.py
│   └── prepare_data.py         # Phase-2 dataset prep
├── configs/train_config.yaml   # Phase-2 ConvNeXt config
├── .env.example
├── Dockerfile                  # Phase-2 (CUDA + PyTorch)
├── Dockerfile.webapp           # Phase-1 (CPU, Sonnet)
├── fly.toml                    # Phase-1 deploy config
└── requirements*.txt
```

---

## Setup — Phase 1 webapp

### 1. Dependencies

```bash
python -m venv .venv-webapp
.venv-webapp/Scripts/pip install -r requirements-webapp.txt    # Windows
# or: .venv-webapp/bin/pip install -r requirements-webapp.txt  # Unix
```

### 2. External accounts (one-time, manual)

| Service | Why | Where |
|---|---|---|
| Anthropic | Sonnet inference | https://console.anthropic.com |
| Supabase | Postgres + auth | https://app.supabase.com |
| Cloudflare R2 | Photo storage | https://dash.cloudflare.com → R2 |
| Fly.io (or Railway) | App hosting | https://fly.io |

Create each, copy credentials into `.env`:

```bash
cp .env.example .env
# edit .env — fill ANTHROPIC_API_KEY, SUPABASE_*, R2_*, APP_ADMIN_PASSWORD
```

### 3. Prepare the taxonomy

If you have scraped data at `~/Desktop/aecis-violations/`, regenerate the
taxonomy from it (keeps slugs in sync with the actual data):

```bash
.venv-webapp/Scripts/python scripts/extract_taxonomy.py --min-count 3
```

### 4. Initialize Supabase

In the Supabase SQL editor paste + run `supabase/schema.sql`. Then seed the
prepared vocabularies:

```bash
.venv-webapp/Scripts/python scripts/seed_taxonomy_to_supabase.py
```

Create an R2 bucket named `aecis-violations` (or whatever you set in `.env`).

### 5. Run locally

```bash
# Terminal 1 — web UI
.venv-webapp/Scripts/uvicorn webapp.app:app --reload --port 8000

# Terminal 2 — classification worker
.venv-webapp/Scripts/python -m webapp.worker
```

Open http://localhost:8000 . Log in with username `admin` + your
`APP_ADMIN_PASSWORD`.

### 6. Deploy to Fly.io

```bash
fly launch --no-deploy --name violation-ai-webapp --dockerfile Dockerfile.webapp
fly secrets set ANTHROPIC_API_KEY=... SUPABASE_URL=... ...  # all keys from .env
fly deploy
```

The `fly.toml` defines two processes (`app` and `worker`); they share one
container image but run different commands.

---

## Phase 1 classifier — CLI

Sanity-check the classifier against a single photo (bypasses DB):

```bash
.venv-webapp/Scripts/python -m src.zero_shot --image path/to/photo.jpg
```

Output is JSON: `{location, hse_type, rationale, model, input_tokens, output_tokens}`.

---

## Phase 2 (later — when you have ~2.5k corrected labels)

Path is already scaffolded in this repo (the pre-existing PyTorch code):

```bash
pip install -r requirements.txt                # heavy: torch, torchvision, etc.
python scripts/prepare_data.py                 # builds train/val/test splits
python -m src.train --config configs/train_config.yaml
python -m src.server --checkpoint checkpoints/best_model.pt
```

Or, simpler: upload the dataset in `~/Desktop/aecis-violations-training/by_hse_type/`
to **HuggingFace AutoTrain** (two separate runs: hse_type + location) and
swap the inference hot-path in `webapp/worker.py` from `zero_shot.classify_image`
to a HuggingFace Inference API call.

---

## Dataset pipeline (upstream)

Not in this repo — lives at `../construct bot/`. Produces the
`~/Desktop/aecis-violations/` tree that feeds:

- `scripts/extract_taxonomy.py` → `taxonomy.json`
- Phase-2 training dataset

Key scripts in the other repo:

- `scripts/run_issues_scrape.py --project SVN` — scrape AECIS issues
- `scripts/audit_violations_dataset.py` — class balance + dedup report
- `scripts/reorg_for_training.py --link` — flat ImageFolder trees
- `scripts/generate_review_html.py` — browsable QA page

---

## Commercial economics

At 86 photos / project / month (one weekly site inspection batch):

- Phase 1 per-project COGS: **~$0.50/month** (Sonnet batch + storage)
- Phase 1 per-project revenue: **$200/month**
- Gross margin: **~97%** at 20+ projects

Scaling constraint is customer support + sales, not compute. See the
architecture design doc (separate deliverable) for full cost tables.
