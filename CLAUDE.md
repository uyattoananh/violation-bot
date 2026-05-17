# HSE Detector — project brief for Claude

Two-axis classifier for AECIS construction-site safety photos. Inspector
uploads a photo, the model returns a `(location, hse_type)` pair plus a
top-3 alternative list.

Honest accuracy (multi-seed N=50 mean, inspector-validated `manual`
ground truth, as of 2026-05-16 — full DB + SupCon + rare-mine):

  Axis        Top-1   Top-3   Target          Status
  HSE type    63.0%   80.5%   50% / 80%       ALL PASS
  Location    63.7%   94.7%   50% / 80%       ALL PASS

**All four targets pass at the multi-seed mean.** Per-photo cost
~$0.0015 (Gemini Flash 2.5 via OpenRouter with prompt caching).

The final +6.7pp top-3 HSE jump came from `mine_rare_class_seeds.py`
adding 128 LLM-curated rare-class seeds (Pressure_equipment 17,
Site_lighting 15, Formwork 15, Garbage_waste 12, Chemicals_hazmat
27, etc.) — only ~3% of the v2 corpus but targeted at the exact
classes the multi-class classifier rarely picks. See "Targeted
rare-class mining" below.

SupCon projection layer (env SUPCON_RAG=1) adds +10pp top-1 HSE,
+7pp top-1 LOC, +2.7pp top-3 HSE, +4pp top-3 LOC vs raw CLIP. See
"SupCon contrastive projection head" section below.

### Targeted rare-class mining (`scripts/mine_rare_class_seeds.py`)

For each of 15 sparse classes (Welding, Truck_vehicle, Mass_piling,
Smoking, Concrete, Site_lighting, Garbage_waste, Pressure_equipment,
Formwork, Common_area, First_aid_kit, Parking, Confined_space,
Chemicals_hazmat, Ladder) — scan manifest by keyword, then send a
TARGETED vision-verify ("does this photo depict THIS specific
class?") to Gemini Flash 2.5. Confirms become embeds with
label_source='aecis_curated_rare_v1'. No human judgement needed —
LLM does the visual gating.

Different from `/admin/seed/aecis-label-ingest` which classifies
across all 29 classes; this hunts a single class hypothesis at a
time, much higher recall for rare classes.

Cache lives at `scripts/.rare_mine_cache.json` (sha+slug keyed).
Cost: ~$2-3 for a full sweep at min_vision_confidence=0.55,
max-checks=100 per class.

## Accuracy targets (north star)

We optimise for **top-1 ≥ 50%** and **top-3 ≥ 80%** on inspector-validated
ground truth. Top-1 is what the inspector sees as the AI's primary
suggestion; top-3 is the runner-up list they can one-click into when the
primary is wrong. Holding 80% top-3 means a single tap fixes 4 out of 5
mistakes — the UX is fast even when the model isn't quite right.

When measuring, always include both numbers. **Don't accept a Haiku
result that beats Sonnet** without first confirming the ground truth
isn't itself LLM-generated (see "Eval" below — comparing two LLMs against
a third LLM's labels measures agreement, not accuracy).

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
  k-NN → LLM prompt with neighbour hints → parse JSON top-3. The
  default model is `google/gemini-2.5-flash` (see
  `DEFAULT_OPENROUTER_MODEL` at line 72). Production VPS env confirms
  this — the `classifications.model` column shows
  `openrouter:google/gemini-2.5-flash+stage2` on all recent rows. The
  repo `.env` may set `OPENROUTER_MODEL=anthropic/claude-sonnet-4.5`,
  but that's a stale dev override; treat Gemini Flash 2.5 as the
  source of truth. Always eval with the same model production runs.
- `scripts/` — flat folder of one-off DB migrations, seeders,
  audits. Each script names what it does. Key seed pipeline pieces:
  - `seed_csv_from_sql.py` — rebuild full-corpus CSV from MSSQL dumps
  - `seed_download_with_iam.py` — wrapper that loads CSV credentials
  - `seed_download_aecis_photos.py` — fetch photos from AECIS S3
  - `seed_walk_manual_corpus.py` — sha256 map for `manual` eval files
- Admin endpoints (admin-only via `_admin_authed`):
  - `POST /admin/seed/aecis-label-ingest` — text→HSE classification
  - `POST /admin/eval/random-labelled` — end-to-end pipeline eval

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

## AECIS seed pipeline (text-labelled)

End-to-end flow that pulls AECIS photos, classifies their text labels
into our HSE taxonomy via OpenRouter, and writes the embeddings into
`photo_embeddings`. Cheap (~$0.0008/row text-only vs ~$0.014/row vision)
and the labels come straight from AECIS's own DB, so no human review
loop is needed before retrieval improvement shows up.

### The 4-step flow (each step is resumable)

```
1. scripts/seed_csv_from_sql.py
   Parses PM.Issue.sql + PM.IssueActivity.sql + PM.IssuePhoto.sql
   (UTF-16-LE Microsoft .sql dumps) and rebuilds the join CSV WITHOUT
   the `IssueActionID = 1` filter. The DB manager's hand-off CSV only
   contained ~39k of the 284,975 IssuePhoto rows — the action filter
   dropped photos attached to update / resolve / close events.
   Output: Issue_Gen/Issue_Gen/result_after_query.full.csv (275k rows)
   Also normalises Windows backslashes in FilePath → forward slashes
   (method.txt §3.1 — AECIS internal code does the same).

2. scripts/seed_download_with_iam.py --csv <full_csv> --disciplines all
   Wrapper that loads the IAM credentials from
   Issue_Gen/rnd-user_accessKeys.csv (gitignored) into env and execs
   seed_download_aecis_photos.py. NEVER echo the CSV contents.
   Output: Issue_Gen/photos/<sanitized_filepath> + manifest.jsonl
   The downloader's `_safe_local_path` rewrites Windows-reserved chars
   (`:*?"<>|`) to `_` because AECIS occasionally stores Apple HEIC
   asset URLs like `@asset_url:UUID/L0/...`.

3. POST /admin/seed/aecis-label-ingest?limit=200&model=...&min_confidence=0.6
   Reads Issue_Gen/photos/manifest.jsonl, dedups by sha256 (AECIS
   stores the same JPEG attached to ~12 issues on average), text-
   classifies each unique label pair (issue_name + description) via
   OpenRouter into the 33-parent taxonomy, CLIP-embeds the photo, and
   upserts to photo_embeddings with label_source='aecis_labelled_v1'.
   Cache lives at scripts/.aecis_label_cache.json (sha-text-keyed) so
   re-runs cost nothing for already-judged labels. The `processed`
   counter only increments on FRESH OpenRouter calls — cache-hit
   rejections don't burn the per-call budget.

4. POST /admin/eval/random-labelled?sample_size=100&label_source=manual
   See "Eval" below.
```

Drive step 3 in a loop until `"remaining": 0` — see `tmp/loop_ingest.sh`
for the pattern (curl `-m 1800` and `LIMIT=100` is a sane combo; larger
LIMIT increases per-request wall time enough to bump into HTTP timeouts).

Only run from a workstation. The downloader is local-only — the
`Issue_Gen/` bundle isn't deployed to the VPS, and pulling 275k photos
through the live worker's classify queue would dominate its OpenRouter
spend for the day.

## Eval (end-to-end pipeline accuracy)

`POST /admin/eval/random-labelled` samples N already-labelled rows from
`photo_embeddings`, marks each `is_holdout = TRUE` for the duration of
the call (so the kNN RPC excludes them from their own candidates —
leak guard), runs each image through `src.zero_shot.classify_image()`,
and compares predicted vs stored labels. The is_holdout flips are
wrapped in try/finally so an eval crash always restores the flag.

Two `label_source` filters are useful:

- `aecis_labelled_v1` — the photos we just text-classified. Useful as
  a fast self-test, but predicted-vs-stored is LLM-vs-LLM agreement,
  not real accuracy.
- `manual` — the 3,015 inspector-validated rows. THE honest benchmark.
  Their source files were originally in `~/Desktop/aecis-violations/`
  under folder names that have since changed; `scripts/seed_walk_manual_corpus.py`
  rebuilds a `{sha256: relative_path}` map at `tmp/manual_corpus_sha_map.json`
  so the eval can find the local JPEG by sha256 (path lookup no longer
  works). 3,004 of 3,015 rows are recoverable this way (99.6%).

Honest baseline on `manual` ground truth (sample=100, seed=42):

  Eval at 2026-05-14, label_source='manual' (inspector-validated):
    Haiku 4.5:        top-1 HSE=42.9%  top-1 LOC=41.8%  top-3 HSE=61.2%  top-3 LOC=81.6%
    Sonnet 4.5:       top-1 HSE=44.9%  top-1 LOC=49.0%  (top-3 not measured)
    Gemini Flash 2.5: top-1 HSE=46.0%  top-1 LOC=53.0%  top-3 HSE=68.0%  top-3 LOC=84.0%

  Gemini Flash 2.5 beats both Anthropic models on every metric AND
  is what production already serves. Use it for everything: classify,
  eval, and the seed pipeline's vision-verify step. Do NOT compare
  one model's predictions against another's labels (we did this with
  Haiku-predictions-vs-Haiku-labels and the numbers were
  meaningless — measure agreement, not accuracy).

  Two of four targets PASS on this eval: top-1 LOC ✅ (53% ≥ 50%) and
  top-3 LOC ✅ (84% ≥ 80%). HSE axis still needs work: top-1 HSE -4pp,
  top-3 HSE -12pp. The HSE axis has 29 classes vs LOC's 9, so it's
  intrinsically harder.

### Wins that stuck (2026-05-14)

Three stacking changes in `src/zero_shot.py`, validated at N=100 against
the manual ground truth (with deterministic sample order in
`webapp/app.py:7687` and `temperature=0.0` on the single-sample path):

  1. `RAG_NEIGHBOURS_DEFAULT = 5 → 15`         (line 81)
  2. Single-sample classify pass `temperature=0.0`  (line ~870)
  3. `_format_rag_block` adds per-class vote+mean-distance summary
     with weighting guidance to the model.

Cumulative result at N=100:

  Baseline (k=5, no vote summary):    top1_HSE=46.0% top1_LOC=53.0%
                                       top3_HSE=68.0% top3_LOC=84.0%
  Final (k=15 + vote summary + HK suppression):
                                       top1_HSE=46.5% top1_LOC=52.5%
                                       top3_HSE=75.8% top3_LOC=83.8%

Top-3 HSE moved 68.0% → 75.8% (+7.8pp, 60% of the gap to 80% closed).
Other metrics held flat.

### Housekeeping_general k-NN suppression — REVERTED 2026-05-16

We tested suppressing 698 of the 780 manual Housekeeping_general
rows via `is_holdout=TRUE` to combat the gravity-well failure mode
(9 of 11 top-3 misses were "<true class> → predicted Housekeeping").
Reduced active count to 80, parity with other major classes.

**Verdict: marginally helpful but inside N=50 noise band; reverted
when restoring the full pool gave at-least-as-good results.**

Multi-seed (3-seed) comparison:

  With HK suppressed (80 active):  top1_HSE=52.1% top1_LOC=57.7%
                                    top3_HSE=70.2% top3_LOC=85.9%
  After deflag (777 active):        top1_HSE=52.4% top1_LOC=57.1%
                                    top3_HSE=71.8% top3_LOC=88.0%

Deflag won on 3 of 4 metrics. The single-seed=25 reading of "top-3
HSE 81.6%" that motivated the suppression was sampling luck — the
multi-seed mean was ~70%, not ~80%.

Backup at `tmp/housekeeping_reduction.json` if you want to re-test.
The full manual pool is currently active.

### v2 rare-class additions caused a transient safeguard trip

On 2026-05-15 the loop's safeguard tripped at v2=894 (top-3 HSE
65.3%, -8.2pp from baseline). The trip was REAL on that N=49
sample but turned out to be N=50 sampling noise + interaction with
the active Housekeeping suppression. Investigation:

  - Identified the 252 most recent v2 rows (642 → 894). All went
    into RARE classes (Scaffolding 41, Lifting 28, Equipment 24,
    PPE 23, Hot_work 23, Ladder 17, etc.) — exactly what the
    max_per_class quota was designed to encourage.
  - Suspected these were polluting retrieval. Tested removing them
    via is_holdout=TRUE.
  - **Removing them made things WORSE**: top-1 HSE -5.0pp, top-3
    LOC -3.3pp, top-3 HSE -1.8pp. They were actively helping.
  - Restored them. Current state at v2=894 with full Housekeeping
    pool active is the local optimum.

Lesson: don't trust a single-seed N=50 trip to indicate a real
problem. Always re-eval the same configuration on 3+ seeds before
concluding the corpus is polluted. Backup at
`tmp/v2_problem_rows.json` (the 252 sha256s) preserved for future
investigation.

Important caveat about is_holdout: the flag is overloaded across at
least three uses (eval leak guard, manual pool-reduction experiment,
v2 problem-row test). Use a dedicated `pool_excluded` column if
any pool-shaping mechanism becomes permanent.

### Eval reproducibility — REQUIRED to A/B reliably

Two unrelated changes had to land before any of the per-1pp tuning
above was measurable:

  1. **Sort candidates by sha256 in the eval endpoint** before
     `random.sample`. Supabase doesn't guarantee row order, so without
     this two consecutive evals of the same configuration drift 5-10pp
     on every metric — far larger than any real intervention we're
     trying to measure. The fix is in
     `webapp/app.py:admin_eval_random_labelled`.

  2. **`temperature=0.0` on the single-sample classify path** in
     `src/zero_shot.py`. The default temperature was the provider's
     default (typically 1.0), so identical inputs produced
     non-identical outputs across runs.

After both: back-to-back evals of identical configuration give
identical per-photo predictions and metrics. 0/N disagreement.

### N=50 sample variance is huge — use multi-seed

The fixed seed=42 sample at N=50 was misleading us. Running the same
model + config under 5 different seeds (1, 7, 13, 19, 25), each N=50,
yielded these means and standard deviations:

  top-1 HSE  mean=52.1%  stdev=8.9pp  range=[44.0%, 65.3%]   target 50% — PASS at mean
  top-1 LOC  mean=57.7%  stdev=8.4pp  range=[48.0%, 67.3%]   target 50% — PASS at mean
  top-3 HSE  mean=70.2%  stdev=8.0pp  range=[64.0%, 81.6%]   target 80%
  top-3 LOC  mean=85.9%  stdev=6.2pp  range=[78.0%, 93.9%]   target 80% — PASS at mean

**3 of 4 targets PASS at the multi-seed mean.** Top-3 HSE one sample
(seed=25) crossed 80% on its own; the mean sits at 70.2% with stdev 8.0pp.

Practical implication: a single N=50 eval is essentially a coin-flip
of ±5-10pp. Any single-seed A/B measuring <8pp delta is statistically
indistinguishable from noise. The "+18pp top-3 HSE" excitement on the
Housekeeping reduction was inside this noise band — at multi-seed
mean the actual gain is ~+2-4pp, not +18pp.

When measuring future changes: run ≥3 seeds, average them. Or move
to N=200+ for a single seed (proportionally cheaper than 3× N=50 in
wall-clock thanks to OpenRouter parallel cost, ~4× the dollars but
one set of numbers).

### Top-3 HSE wall — what we did and didn't do

### Top-3 HSE ~68% wall

Per-class breakdown at N=100 shows top-3 HSE concentrates losses on
specific failure-mode classes (Site_access_unsafe, Truck_vehicle_unsafe,
Mass_piling_unsafe, Hot_work_hazard, etc.) — many of which have
plenty of `manual` seeds. The real failure mode is VISUAL AMBIGUITY
vs Housekeeping_general (780 of 3,015 manual rows = 26%, the
gravity well). Confusion pairs from one eval:

  - Site_access_unsafe → predicted Housekeeping_general: 2/3 misses
  - Fall_protection_personal → Housekeeping_general: 2/5 misses
  - Edge_protection_missing → Housekeeping_general: 2/4 misses
  - Hot_work_hazard → Housekeeping_general: 2/4 misses
  - Lifting_unsafe → Housekeeping_general: 2/4 misses

Many real photos look like "general site mess" and the model defaults
to Housekeeping_general when the specific hazard isn't unambiguous.

### Things tried that did NOT help (don't re-try without new evidence)

  - **k-NN class-diversity filter** (cap N-per-hse_slug in
    `_retrieve_similar_labels`): substituting closer Housekeeping
    neighbours for further-distance diverse ones cost more signal than
    it bought. -6pp top-1 HSE / -2pp top-3 HSE at N=50.
  - **Runner-up diversity rule in SYSTEM_PROMPT** ("if primary is
    catch-all, runner-ups MUST be specific-hazard classes"): -7.7pp
    top-1 HSE / -5.2pp top-3 HSE at N=100. The model second-guesses
    its top-1 and the top-3 ranking gets worse.
  - **Self-consistency `samples=3`**: ±2pp on different metrics,
    sometimes hurts top-3 LOC by 16pp. 3× cost for no consistent gain.
  - **Sonnet 4.5 swap**: top-1 HSE 51.9% (vs Gemini 46.0%) on identical
    sample — but came with 20 errors vs 0. Marginal, not worth 14× cost.
  - **Full-corpus class rebalance** (cap ALL classes >100 manual rows
    at 80): -6pp top-3 HSE. The other classes were HELPING; only
    Housekeeping was the gravity well. Reducing them deprived the
    model of valid signal.
  - **Unify seed taxonomy with production taxonomy** (rewrite
    `_aecis_label_taxonomy()` to use `src.zero_shot.load_taxonomy()`,
    map legacy AECIS-vocab location slugs to production-vocab via
    Excavation_or_pit → Excavation, Height_work → Working_at_height,
    etc.): -14pp top-1 HSE / -20pp top-1 LOC. The two location
    vocabularies are EACH internally consistent — collapsing them
    breaks k-NN match semantics. location_slug is a backend k-NN
    hint, not a user-facing dropdown value — don't try to "fix" it.
  - **k=20, k=25 RAG_NEIGHBOURS**: over-fetched, diluted signal.
    k=15 remains the sweet spot.
  - **`samples=3` with vote-summary RAG**: 3× cost, marginal or
    negative across metrics.

### Things still untried (worth investigating)

  - Hand-curate ~50 rare-class photos for the 7-8 zero-accuracy classes
    and seed them directly. Direct attack on the rare-class data
    sparsity (some classes have 0 v2_visionchecked seeds AND ≤27
    manual seeds — model has nothing to retrieve).
  - Hierarchical classification: broad-category-first, then refine.
  - **Distance-weighted kNN reranking via RPC** (rather than the
    `is_holdout` overload for HK suppression): modify
    `match_photo_embeddings` to multiply distance by a class-frequency
    factor so popular classes need to be visually MORE similar to win.
    More principled than the current exclusion hack.

## SupCon contrastive projection head (2026-05-16)

Wired in `src/zero_shot.py:_retrieve_similar_labels_supcon`. Active
when env `SUPCON_RAG=1`. Otherwise falls through to the pgvector RPC.

### What it does

A small projection head (`Linear(512, 256) → ReLU → Linear(256, 512)`
with residual + L2 normalize) is trained with Supervised Contrastive
loss on the 2,958 active manual photo embeddings, class-balanced
sampler, 50 epochs, ~3s on GPU. At inference time:

  1. CLIP-embed the query photo (as today).
  2. Project the query embedding through the head.
  3. Numpy-side cosine-kNN against a precomputed projected matrix of
     all photo_embeddings (saved at `tmp/clip_supcon_embeddings.npz`,
     ~9 MB, ~4,500 rows).
  4. Return the top-k as if from the pgvector RPC.

No DB schema change. No new RPC. Reversible by unsetting
`SUPCON_RAG`. Per-call overhead: a 256x512 + 512x256 matmul on the
query plus one (N×512) @ (512,) dot product — sub-millisecond.

### Measured impact (3-seed N=50 mean, manual ground truth)

  Metric          Original CLIP    SupCon         Δ
  top-1 HSE       52.4%            62.4%          +10.0pp
  top-1 LOC       57.1%            64.4%          + 7.3pp
  top-3 HSE       71.8%            74.5%          + 2.7pp
  top-3 LOC       88.0%            92.0%          + 4.0pp

Pure-kNN (Diagnostic 3, no LLM) showed +11.7pp top-3 HSE in
projected space; the LLM pipeline absorbed about ¼ of that into
the top-3 axis (most of the gain landed in top-1, which the kNN
alone wasn't measuring well).

### Per-class kNN-only gains (Diagnostic 3, 80/20 split)

Rare classes win big; common classes take small hits:

  Chemicals_hazmat_unsafe      n=13  orig=15%  → SupCon=92%  +77pp
  Site_general_unsafe          n=17  orig=29%  → SupCon=71%  +41pp
  Equipment_machinery_unsafe   n=15  orig=33%  → SupCon=73%  +40pp
  Warning_signs_missing        n=24  orig=71%  → SupCon=92%  +21pp
  Lifting_unsafe               n=22  orig=64%  → SupCon=82%  +18pp
  Materials_storage_unsafe     n=24  orig=71%  → SupCon=88%  +17pp
  Housekeeping_general         n=156 orig=99%  → SupCon=96%  - 3pp
  Electrical_unsafe            n= 74 orig=92%  → SupCon=88%  - 4pp
  Edge_protection_missing      n= 62 orig=84%  → SupCon=73%  -11pp

Trade-off is real — SupCon helps the rare classes that were the
top-3 HSE bottleneck, at the cost of a small accuracy hit on the
3 most populous classes. Net is positive in the multi-seed eval.

### Files involved

  - `scripts/train_supcon_head.py` — train the head, save .pt
  - `scripts/project_supcon_embeddings.py` — backfill projected
    embeddings for every photo_embeddings row, save .npz
  - `tmp/clip_supcon_head.pt` — trained weights (~700 KB)
  - `tmp/clip_supcon_embeddings.npz` — projected matrix (~9 MB)
  - `src/zero_shot.py:_retrieve_similar_labels_supcon` — inference path

### Retraining

Re-run when the manual corpus changes meaningfully (>10% new rows
in any class):

  ./.venv-webapp/Scripts/python.exe scripts/train_supcon_head.py
  ./.venv-webapp/Scripts/python.exe scripts/project_supcon_embeddings.py

### v2 + v3 head variants tried, did NOT win (2026-05-16)

We tried two follow-up training scripts; both regressed or stayed
flat across the 4-metric multi-seed mean, so v1 stays in production.
The .pt files (`src/clip_supcon_head_v2.pt`, `src/clip_supcon_head_v3.pt`)
remain in the tree as reproducibility artifacts. Toggle via env
`SUPCON_HEAD_VERSION=2` or `=3` to A/B locally.

**v2 — multi-axis loss + v2_visionchecked corpus + temperature anneal**
(`scripts/train_supcon_head_v2.py`):

  Metric       v1     v2     delta
  top-1 HSE   62.4%  65.1%   +2.7pp
  top-1 LOC   64.4%  64.4%   +0.0pp
  top-3 HSE   74.5%  74.5%   +0.0pp
  top-3 LOC   92.0%  89.9%   -2.1pp

The 0.3× location SupCon term pulled embedding capacity away from
hse_type separation. The v2_visionchecked rows (LLM-labelled, not
human-validated) seem to disagree with the manual ground truth on
the boundaries we measure — they dilute training signal at the
manual-eval target.

**v3 — confusion-pair hard-negative mining on manual-only corpus**
(`scripts/train_supcon_head_v3.py`):

  Metric       v1     v3     delta
  top-1 HSE   62.4%  61.8%   -0.6pp
  top-1 LOC   64.4%  65.1%   +0.7pp
  top-3 HSE   74.5%  75.2%   +0.7pp
  top-3 LOC   92.0%  89.3%   -2.7pp

Sample-weight boost based on per-class confusion rate, computed
from past eval per_photo files. The boost (1 + 2× rate) gave some
classes 20× sampling weight because confusion-rate-as-defined
overcounts (a class wrongly predicted AS X bumps X's miss count).
Net effect on the multi-seed eval was flat. The honest read: SupCon
already pulls same-class together implicitly; explicit hard-negative
sampling didn't compound. Save the lever for a richer confusion
dataset.

**What to try next** if pushing top-3 HSE further:

  1. Hand-curate ~50 rare-class photos and train v4 on a
     refresh of manual + curated_rare.
  2. LoRA fine-tune CLIP's last 1-2 attention layers — bigger
     capacity, real catastrophic-forgetting risk.
  3. Wait for more `aecis_labelled_v2_visionchecked` rows from the
     ongoing AECIS ingest, retrain v1 (manual-only) after that
     corpus has visibly improved per-class diversity.

## Seed-pipeline failure mode: text-only force-fit

Documented 2026-05-14 after the safeguard tripped at 1917→2140 embeds.

`/admin/seed/aecis-label-ingest` classifies AECIS issue TEXT via Haiku
into one of our 33 HSE slugs at conf≥0.6. The trap: when the text
describes a violation TYPE not in our taxonomy (insufficient lighting,
drainage, demolition cleanup, etc.), the model still picks the closest
slug at high confidence — producing a confidently-wrong label.

Investigation: pulled 556 of these embeds back out of the DB, ran each
photo through Gemini Flash 2.5 vision in free-form mode. Result:

  - 33% (182/556) of photos visually matched their Haiku-assigned slug
  - 65% (361/556) MISMATCHED — photo showed something else entirely
  -  2% (10/556)  were not violations at all
  -  1% (3/556)   were ambiguous

The 65% pollution explains the accuracy regression at checkpoint 4
(top-3 HSE 0.640 → 0.490 over 223 embeds). Per-class mismatch
hotspots: Fire_prevention_unsafe 81%, Materials_storage_unsafe 77%,
Excavation_unsafe 73%, Electrical_unsafe 71%, Edge_protection_missing 65%.

Fixes deployed in `webapp/app.py`:

  - `_AECIS_LABEL_SYSTEM_PROMPT` rewritten to enumerate 5 reject
    categories (administrative, workmanship, cleanup-AFTER, edge-cases-
    outside-taxonomy, vague placeholder). Tells Haiku to return null
    when no class in our list matches — don't force-fit.
  - Default `min_confidence` raised 0.6 → 0.7 to filter low-confidence
    forced fits.

Still required for the next full ingest:
  - Vision-verify step BEFORE embedding: after text-classifier proposes
    a slug, send the actual photo + proposed slug to Gemini Flash 2.5
    and only embed when the model confirms the photo matches the slug.
  - Cost ~$5 extra on top of text classify. See README of
    `tmp/investigate_v2.py` for the verified mismatch pattern.

## Live debugging path

When eval numbers shift unexpectedly:

  - Check `classifications.model` for the last 10-30 rows to confirm
    production is still on Gemini Flash 2.5.
  - Run `/admin/eval/random-labelled?label_source=manual&seed=42` to
    re-baseline. The seed=42 sample is stable across runs.
  - Use the `tmp/manual_corpus_sha_map.json` (built by
    `scripts/seed_walk_manual_corpus.py`) to resolve `manual`
    source_paths to local files — 99.6% of the 3,015 `manual` rows
    are recoverable this way after the folder rename.

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
