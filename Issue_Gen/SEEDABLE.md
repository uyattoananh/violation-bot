# Issue_Gen — seedability assessment

Audit of the MS SQL dump in this folder for whether the data can feed
the HSE Detector seed-backdoor path described in `FUTURE_IMPLEMENTATIONS.txt`
and `CLAUDE.md`. Drafted 2026-05-10.

## What's in this folder

| File | Size | What it is |
|---|---|---|
| `PM.Issue.sql` | 603 MB | T-SQL dump of `[AECIS_Projectmanagement].[PM].[Issue]` |
| `PM.IssueActivity.sql` | 457 MB | T-SQL dump of `[PM].[IssueActivity]` |
| `PM.IssuePhoto.sql` | 290 MB | T-SQL dump of `[PM].[IssuePhoto]` |
| `query HSE, issueaction.txt` | 4 KB | The SQL that produced the CSV below |
| `result_after_query.csv` | 76 MB | Joined Issue + Activity + Photo, filtered to `IssueActionID = 1 AND IsDeleted = 0` |

The CSV is the only file actually useful for seeding — the raw `.sql`
files are unfiltered table dumps. The CSV holds the **photo-creation
events** with their associated issue metadata.

## CSV totals

- **39,137** photo rows (one per IssuePhoto record)
- **18,479** unique Issues
- **2.1** photos per issue, average

## Discipline split

| DisciplineID | Rows | Description |
|---|---|---|
| `0`  | 22,320 | Default / unset. General construction finishing issues. |
| `10` | **2,444** | **HSE.** Filter to this for the seed pipeline. |
| `74` | 2,408 | Inspection / handover |
| `9`  | 2,166 | Construction finishing |
| `93` | 1,720 | Painting trade |
| 197, 8, 11, 231, 1, … | 8,000+ | Various other trades |

Of the 2,444 HSE-disciplined rows:
- **479 unique HSE issues** (5.1 photos per HSE issue)
- **95 % from one project** (P_210) — this is essentially one AECIS site's HSE inspection history.
- Extensions: 2,428 `.jpg`, 11 `.jpeg`, 5 `.png` — all standard formats.

## Label quality is mixed

Sampling DisciplineID=10 rows uncovered TWO populations:

### 1. Structured rows (only 10 of 2,444)

A handful of rows follow a clean bilingual taxonomy:

```
<EN parent>/ <VN parent> | <EN finding>/ <VN finding>
```

Examples:
- `Digging/Deep hole/ Đào đất/ Hố sâu | No way up/down the pit with handrails on both sides/...`
- `Working at height/ Làm việc trên cao | Materials are risk of falling from height/...`
- `Trucks/ Xe tải các loại | The reversing vehicle does not have a signal/...`

These parse trivially into `(hse_type, finding)` pairs and would
auto-map to our 13/29-class taxonomy via a one-time lookup table.

### 2. Free-text rows (the other 2,434)

The bulk of HSE-disciplined rows look like:

```
name: "LL5 Toilet"
desc: "Poor workmanship"
```

These are workmanship issues mis-filed into the HSE discipline.
They are NOT useful for seeding a safety classifier — labelling
them would be more harmful than helpful (the inverse of "garbage
in, garbage out" — actively wrong labels would corrupt retrieval).

Worse, **DisciplineID = 10 is not a reliable HSE filter**. Many
non-safety issues carry it. So even after disciplining, we need a
content filter.

## Photo binaries live on an AECIS S3 bucket

Every row has a `FilePath` column like:

```
P_2374/Issue/U_12896/12_05_2026/1ea25976b0724935b6b93f257c3f0dff.jpeg
```

This is a **bucket-relative key** — the JPEG/PNG bytes live on
AECIS's S3 bucket. Combined with the bucket's HTTPS base URL it
forms a direct download link, e.g.:

```
https://aecis-issues.s3.ap-southeast-1.amazonaws.com/P_2374/.../1ea25976...jpeg
```

The bucket base is wired via the `AECIS_PHOTO_S3_BASE` env var
in `webapp/app.py`. Once that's set, the admin endpoint

```
GET /admin/seed/aecis-urls?fmt=text   # one URL per line
GET /admin/seed/aecis-urls?fmt=json   # full record per row (default)
GET /admin/seed/aecis-urls?limit=100  # cap the first N for sample testing
```

streams one HTTPS URL per HSE-disciplined photo row. The `text`
format is straight `xargs curl -O` material for bulk download;
`json` carries `issue_id`, `project_id`, `issue_name`,
`description`, and `s3_url` so the seed ingest can stamp the
right metadata when persisting each photo.

The endpoint is gated by `_admin_authed` — it's not part of the
inspector-facing surface, only the seed-pipeline tooling.

### What's needed before the endpoint works:

1. Get the real S3 bucket URL from AECIS (region + bucket name) —
   the example above is illustrative.
2. Set `AECIS_PHOTO_S3_BASE` in the VPS service environment, OR
   run the seed pipeline locally with the var set.
3. Confirm the bucket allows anonymous/IAM read for our caller.
   If it requires signed URLs, the endpoint needs a small
   extension to sign each URL via `boto3.generate_presigned_url`.
4. The Issue_Gen/ folder needs to be present on the host running
   the endpoint. It's bundled in this repo but isn't auto-deployed
   to the VPS by default — either deploy it explicitly or run the
   endpoint locally.

## What's seedable today (with binaries)

If photo binaries are made available:

1. **Tier 1: structured rows.** The ~10 structured rows are
   immediately ingestable — auto-mapped to our hse_type slugs via
   a small curation table. Trivial volume but high quality. Useful
   for sanity-testing the seed pipeline end-to-end.

2. **Tier 2: structured rows from FUTURE exports.** The same
   `<EN>/<VN> | <finding>` template appears to come from one
   recent inspector workflow on P_2804. If AECIS standardises on
   that workflow, future exports will give us thousands of clean
   rows. Worth lobbying for.

3. **Tier 3: free-text rows via LLM classification.** The remaining
   ~2,400 rows can be sent through a Sonnet classification pass
   (`<issue_name + description>` → one of our 13 hse_type slugs OR
   "reject as non-HSE"). At ~$0.001 per row that's ~$2.50 for the
   whole set. Confidence-gated — only accept rows whose
   classification score is above some threshold (say 0.7).

4. **Tier 4: full 39 K rows.** Skip. Most are non-HSE workmanship
   issues that would pollute retrieval. The discipline filter is
   already eliminating these.

## Mapping table (Tier 1)

Once the structured pattern parser is built, the mapping is small:

| AECIS EN parent | Our hse_type slug |
|---|---|
| `Digging` / `Deep hole` | `excavation_hazard` |
| `Working at height` | `fall_hazard` (or `working_at_height` if added) |
| `Trucks` | `vehicle_hazard` / `traffic_management` |
| `Workshop area` | `housekeeping` (5S finding) |
| `Formwork` | `formwork` (we don't have this slug yet — check) |

Slugs to confirm against `data/fine_hse_types_by_parent.json`.

## Recommended next step

Before any code:

1. **Ask AECIS** if we can get the photo binaries for at least
   DisciplineID=10 rows. Even a sample of 100 photos lets us
   validate the seed pipeline end-to-end before requesting the
   full 2,444.
2. **Confirm** P_210 is a representative project (it dominates the
   HSE rows). If the inspectors on P_210 had different labelling
   conventions than other projects, the seed would over-fit to
   their domain.
3. **Decide** on a quality bar for Tier 3 LLM classification
   (confidence threshold, reject rate).

Once the S3 base is configured and the bucket grants read access,
follow the schema-migration-first order from
`FUTURE_IMPLEMENTATIONS.txt`:

```
scripts/add_is_seed_column.py    # is_seed + seed_label_source columns
scripts/ingest_aecis_hse.py      # walks /admin/seed/aecis-urls,
                                  # downloads each S3 URL, re-uploads
                                  # to our R2, inserts photos row with
                                  # is_seed=true, seed_label_source=
                                  # "aecis_issue_v1", CLIP-embeds, and
                                  # upserts to pgvector
# audit pass: filter is_seed=false in every stats reader
# (/api/usage/today, /api/batches, picker hint, ai_audit_*.py)
```

## Open questions

- **Is the `Size` column in KB or bytes?** Values like `Size = 18`
  suggest KB (an 18-byte JPEG isn't real). Confirm before
  attempting any size-based filtering.
- **Will the same `compressed_` prefix on `UserTitle` always mean
  the binary is a downscaled version?** If so we may want to ask
  AECIS for original-resolution photos before embedding.
- **Are issue Descriptions ever pure HTML (we saw `<br>`, `<p>`)?**
  Yes — Tier 3 prep needs HTML stripping before passing to the LLM.
- **Multi-finding issues** — when one Description carries two
  `<EN>/<VN> | finding<br><EN>/<VN> | finding` entries, do we want
  the seed pipeline to produce one row (with the first label) or
  duplicate the photo row with each label? Probably duplicate, but
  the answer affects de-dup logic downstream.
