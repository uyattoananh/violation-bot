# violation-bot

Two-axis AI classifier for Vietnamese construction-site safety violations.
Upload a site photograph, get a **location** + **HSE violation type** from
prepared vocabularies, with a top-3 alternatives list for one-click inspector
correction. Every correction trains the retrieval index.

**Current state** (shipped MVP):
- Running: FastAPI webapp + classification worker at http://localhost:8000
- Provider: OpenRouter → `anthropic/claude-sonnet-4.5` + prompt caching
- RAG knowledge base: **1,108 labeled photos** in Supabase pgvector (SVN + MJNT)
- Taxonomy: **13 HSE types / 9 locations** consolidated from 71 raw AECIS DTags
- Accuracy on a 100-photo held-out eval (seed=42): **65.7% top-1 · 84.8% top-3**
- Per-photo cost: **~$0.014** with cache warm

---

## End-to-end flow

```
┌──────────────┐  drag-drop   ┌────────────┐
│  Inspector   │ ───photos──▶ │  FastAPI   │
│   browser    │              │  webapp/   │
└──────┬───────┘              │  app.py    │
       │                      └─────┬──────┘
       │ poll /api/pending          │ insert
       │                            ▼
       │                      ┌──────────────┐
       │                      │  Supabase    │
       │                      │  photos +    │
       │                      │  classify_   │
       │                      │  jobs queue  │
       │                      └─────┬────────┘
       │                            │ pick up
       │                            ▼
       │                      ┌──────────────┐
       │                      │ webapp/      │
       │                      │ worker.py    │
       │                      └─────┬────────┘
       │                            │
       │                            ▼
       │              ┌───────────────────────┐
       │              │ src/zero_shot.py      │
       │              │  1. CLIP-embed query  │
       │              │  2. pgvector k-NN     │
       │              │     for 5 neighbours  │
       │              │  3. Sonnet 4.5 prompt │
       │              │     with taxonomy +   │
       │              │     neighbour hints   │
       │              │  4. parse JSON → top-3│
       │              └───────────┬───────────┘
       │                          │
       │ ◀── confirm/correct ─────┘ (primary + 2 alts)
       │
       ▼
┌────────────────────┐
│ Feedback loop:     │  every click → CLIP embed
│ corrections row +  │  + upsert to photo_embeddings
│ pgvector upsert    │  with label_source='manual'
└────────────────────┘
```

## Repo layout

```
.
├── webapp/                 # Phase-1 FastAPI + Jinja2 webapp
│   ├── app.py              # HTTP endpoints, default tenant/project
│   ├── worker.py           # polls classify_jobs, runs Sonnet, writes results
│   └── templates/          # drag-drop index.html, _base.html
├── src/
│   ├── zero_shot.py        # Sonnet classifier, RAG retrieval, top-3 parsing
│   ├── embeddings.py       # CLIP ViT-B/32 via sentence-transformers
│   └── (train.py etc.)     # Phase-2 PyTorch scaffold — unused until fine-tune
├── supabase/
│   ├── schema.sql          # tenants / projects / photos / classifications /
│   │                       # corrections / classify_jobs / taxonomy tables + RLS
│   └── migrations/
│       ├── 01_photo_rag.sql       # enable pgvector, photo_embeddings table,
│       │                           # match_photo_embeddings RPC
│       └── 02_diverse_retrieval.sql  # replace RPC with 3-nearest + 2-diverse
├── scripts/
│   ├── extract_taxonomy.py         # build taxonomy_source.json from scraped data
│   ├── consolidate_taxonomy.py     # apply taxonomy_merges.json → taxonomy.json
│   ├── seed_taxonomy_to_supabase.py
│   ├── embed_dataset.py            # CLIP-embed scraped photos into pgvector (bulk init)
│   ├── auto_seed_from_disk.py      # bulk insert photos + labels from disk
│   │                               # using DTag ground truth + LLM fallback
│   ├── visual_seed_from_disk.py    # stricter variant: VLM looks at each photo,
│   │                               # skips admin / paperwork photos
│   ├── evaluate_rag.py             # 100-photo held-out accuracy measurement
│   ├── reset_and_reseed.py         # wipe user tables + re-seed taxonomy
│   ├── clear_site_photos.py        # wipe R2 + photos table (keep pgvector)
│   ├── test_r2.py                  # credential smoke-test
│   └── test_supabase.py            # schema + seed smoke-test
├── taxonomy.json           # 13 HSE types + 9 locations (generated)
├── taxonomy_merges.json    # consolidation rules from raw AECIS DTag slugs
├── taxonomy.md             # human-readable summary
├── prompts/classifier.md   # reference prompt + design notes
├── Dockerfile.webapp       # CPU-only, torch + sentence-transformers
├── Dockerfile              # Phase-2 CUDA image, unused for MVP
├── fly.toml                # Fly.io deploy config (app + worker processes)
├── .env.example            # credential template
└── requirements-webapp.txt # webapp deps
```

---

## Setup — first time

### 1. Dependencies

```bash
python -m venv .venv-webapp
.venv-webapp/Scripts/pip install -r requirements-webapp.txt    # Windows
```

On first classification call, sentence-transformers downloads `clip-ViT-B/32`
(~600 MB one-time). It's cached under `%USERPROFILE%\.cache\huggingface`.

### 2. External accounts

| Service | What it's for | Console |
|---|---|---|
| OpenRouter | Sonnet 4.5 inference | https://openrouter.ai/keys |
| Supabase | Postgres + pgvector + auth | https://app.supabase.com |
| Cloudflare R2 | Photo storage (S3-compat) | https://dash.cloudflare.com → R2 |
| Fly.io *(optional)* | Hosting | https://fly.io |

Create each, then:

```bash
cp .env.example .env
# fill OPENROUTER_API_KEY, SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY,
#      R2_ACCOUNT_ID + R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY + R2_BUCKET
```

### 3. Initialize Supabase

In the Supabase SQL editor, run in order:

1. `supabase/schema.sql` — core tables + RLS policies
2. `supabase/migrations/01_photo_rag.sql` — pgvector + embeddings table + RPC
3. `supabase/migrations/02_diverse_retrieval.sql` — replace RPC with class-diverse variant

Verify the connection:

```bash
.venv-webapp/Scripts/python scripts/test_supabase.py --bootstrap
.venv-webapp/Scripts/python scripts/test_r2.py
```

### 4. Seed the taxonomy

Populates `locations` + `hse_types` tables from `taxonomy.json`:

```bash
.venv-webapp/Scripts/python scripts/seed_taxonomy_to_supabase.py
```

### 5. (Optional) Bulk-load training data

If you have scraped photo archive at `~/Desktop/aecis-violations/`:

```bash
# CLIP-embed every photo into pgvector with its source DTag labels (one-time):
.venv-webapp/Scripts/python scripts/embed_dataset.py

# Or auto-seed from scrape (per project), with LLM mapping for unknown slugs:
.venv-webapp/Scripts/python scripts/auto_seed_from_disk.py --project SVN
.venv-webapp/Scripts/python scripts/auto_seed_from_disk.py --project MJNT --no-titles
```

### 6. Run locally

```bash
# Terminal 1 — web UI
.venv-webapp/Scripts/uvicorn webapp.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — classification worker
.venv-webapp/Scripts/python -m webapp.worker
```

Open http://localhost:8000 — no login prompt, just drop photos.

### 7. Deploy to Fly.io

```bash
fly launch --no-deploy --name violation-bot --dockerfile Dockerfile.webapp
fly secrets set OPENROUTER_API_KEY=... SUPABASE_URL=... [all keys from .env]
fly scale memory 1024   # torch needs >512 MB
fly deploy
```

`fly.toml` defines `app` (uvicorn) and `worker` (python -m webapp.worker).

---

## Measuring accuracy

```bash
.venv-webapp/Scripts/python scripts/evaluate_rag.py --n 100 --seed 42
```

Samples 100 DTag-labeled photos from the scraped archive, runs them through
the live classifier (Sonnet + RAG), and reports:

- HSE top-1 + top-3 accuracy
- Location top-1 + top-3 accuracy
- Per-class precision / recall / F1
- Top confusions (ground-truth → prediction)
- Estimated OpenRouter cost

Leak-guard: the query photo's own pgvector embedding is temporarily nulled
during retrieval so it can't be its own neighbour.

---

## Phase 2 (fine-tune, future)

Scaffolded under `src/` + `configs/` + `Dockerfile` (CUDA). Trigger when:

- ≥ 2,500 corrected labels accumulated in `corrections` table
- OR you want provider independence from OpenRouter

Flow:
1. `python scripts/prepare_data.py` — build train/val/test splits from pgvector
2. `python -m src.train --config configs/train_config.yaml` — ConvNeXt training
3. `python -m src.server --checkpoint checkpoints/best_model.pt` — serve trained model

Or simpler: upload the flat training tree to HuggingFace AutoTrain and swap
the hot-path in `webapp/worker.py` from `zero_shot.classify_image` to an HF
Inference API call.

---

## What we learned (distilled during data curation)

### Taxonomy consolidation (71 → 13 classes) worked
Raw AECIS DTags have enormous overlap (Garbage / 5S / Cleaning / Materials
not arranged neatly were all visually indistinct "messy site" photos).
Merging into 13 well-defined clusters lifted top-1 accuracy from 20% → 58%
before we did anything else. See `taxonomy_merges.json`.

### RAG via CLIP k-NN added +8pp
After seeding 684 SVN DTag-labeled photos with their labels into pgvector,
neighbour hints in the prompt pushed accuracy from 58% → 66%.

### Top-3 output is the shipping metric, not top-1
Inspector UI shows primary + 2 alternatives. Top-3 "correct answer is in
the options" = 85% vs. top-1 = 66%. At UX level, inspectors need one click
to correct, not a dropdown.

### Cross-domain scraping HURTS accuracy
Adding photos from a domain different from the query distribution
actively regressed the eval. Tested with three projects:

| Project | Type | Outcome |
|---|---|---|
| MJNT (Logicross Nam Thuan) | industrial logistics | **+8pp** |
| AR (Aqua City Resort) | hotel/resort | -11pp |
| RUVN (RMIT University) | university | -10pp |
| H9 (Primary School) | school | -14pp |

Reason: CLIP ViT-B/32's embedding space maps different project types into
different regions. When cross-domain photos surface as nearest neighbours,
they mislead. Takeaway: **same-domain data only** until a custom CLIP
fine-tune bridges the embedding space.

### Inspector corrections are the sustainable growth path
Real uploads are same-domain by definition — an inspector on project X
uploads photos of project X. Every Confirm/Correct click feeds pgvector
via the feedback loop. Organic growth beats synthetic data augmentation.

---

## Cost model (for commercial context)

Assumes 86 photos / project / month (one weekly inspection batch):

| Scale | Projects | AI / mo | Infra / mo | COGS / mo | Gross margin |
|---|---|---|---|---|---|
| Pilot | 1 | $1.20 | $95 | $96 | vs $200 → 52% |
| Small | 10 | $12 | $95 | $107 | vs $2,000 → 95% |
| Scaled | 100 | $120 | $200 | $320 | vs $20,000 → 98% |

See `prompts/classifier.md` for prompt design notes.
