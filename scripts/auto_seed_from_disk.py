"""Simulate inspector confirmations for scraped on-disk data.

For each on-disk photo with DTag-derived ground truth, create the full
audit trail in Supabase + R2 + pgvector as if an inspector had reviewed it:

  1. Upload the photo bytes to R2 (idempotent: skip if sha256 already there)
  2. Insert photos row (backdated to the scrape time)
  3. Insert classifications row with ground-truth labels as the "AI prediction"
     (model = 'synthetic_dtag_ground_truth', confidence = 1.0)
  4. Insert corrections row with action='confirm', labels = ground truth
  5. Upsert photo_embeddings with label_source='manual' + ground-truth labels
     (only if not already present as 'dtag' — we don't downgrade authority)

Net effect: the Review UI shows these as historical confirmed entries, and
the pgvector index gains inspector-verified manual labels to complement the
existing dtag-labeled scraped index.

Cost: zero Sonnet calls. CLIP embedding runs locally on CPU.

Usage:
  python scripts/auto_seed_from_disk.py --limit 50         # test batch
  python scripts/auto_seed_from_disk.py                    # all DTag-labeled
  python scripts/auto_seed_from_disk.py --project SVN      # only one project
  python scripts/auto_seed_from_disk.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("autoseed")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"
DEFAULT_TENANT_NAME = "Public Demo"
DEFAULT_PROJECT_CODE = "PUBLIC"


def _source_maps() -> tuple[dict[str, str], dict[str, str]]:
    merges = json.loads((REPO_ROOT / "taxonomy_merges.json").read_text(encoding="utf-8"))
    hse_map: dict[str, str] = {}
    for c in merges.get("hse_type_clusters", []):
        for src in c["absorbs"]:
            hse_map[src] = c["slug"]
    loc_map: dict[str, str] = {}
    for c in merges.get("location_clusters", []):
        for src in c["absorbs"]:
            loc_map[src] = c["slug"]
    return hse_map, loc_map


_TITLE_CACHE_PATH = REPO_ROOT / "scripts" / ".title_mapping_cache.json"


def _load_title_cache() -> dict[str, dict[str, str]]:
    if _TITLE_CACHE_PATH.exists():
        try:
            return json.loads(_TITLE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_title_cache(c: dict[str, dict[str, str]]) -> None:
    _TITLE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TITLE_CACHE_PATH.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")


def _map_title_via_llm(title_en: str, title_vn: str, tax: dict) -> tuple[str | None, str | None]:
    """Ask the LLM to assign a 12-class hse_type + 9-class location to a free
    text title. Returns (hse_slug, loc_slug) or (None, None) on failure."""
    import os
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    hse_list = "\n".join(f"  - {h['slug']}: {h['label_en']}" for h in tax["hse_types"])
    loc_list = "\n".join(f"  - {l['slug']}: {l['label_en']}" for l in tax["locations"])
    prompt = (
        "Map the following AECIS issue title to the closest HSE-type slug AND "
        "location slug from these vocabularies. Return ONE JSON object, no prose.\n\n"
        "IMPORTANT: some titles are administrative reminders, paperwork notes, "
        "insurance-expiration alerts, or document submission nudges — NOT visual "
        "site-safety violations. If the title clearly describes an administrative "
        "or paperwork matter (e.g., 'submit documents', 'renew insurance', 'update "
        "contractor records', 'training schedule', 'meeting note'), return "
        '`{"hse_type_slug": null, "location_slug": null}` so the photo is skipped.\n\n'
        "Only map titles that describe a visible physical safety violation "
        "(fall hazard, missing PPE, scaffolding issue, electrical, fire, spill, "
        "housekeeping, lifting, etc).\n\n"
        f"HSE_TYPES:\n{hse_list}\n\n"
        f"LOCATIONS:\n{loc_list}\n\n"
        f'Issue title (EN): "{title_en}"\n'
        f'Issue title (VN): "{title_vn}"\n\n'
        'Return: {"hse_type_slug": "...", "location_slug": "..."}  '
        '(or both null if not a visual violation)'
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"X-Title": os.environ.get("OPENROUTER_TITLE", "violation-bot")},
    )
    raw = resp.choices[0].message.content or ""
    # Extract JSON
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None, None
    try:
        j = json.loads(raw[s : e + 1])
        hse = j.get("hse_type_slug")
        loc = j.get("location_slug")
        valid_hse = {h["slug"] for h in tax["hse_types"]}
        valid_loc = {l["slug"] for l in tax["locations"]}
        if hse not in valid_hse:
            hse = None
        if loc not in valid_loc:
            loc = None
        return hse, loc
    except Exception:  # noqa: BLE001
        return None, None


def _gather(root: Path, project_filter: str | None,
            hse_map: dict[str, str], loc_map: dict[str, str],
            include_titles: bool, tax: dict | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    title_cache = _load_title_cache() if include_titles else {}
    cache_hits = 0
    llm_calls = 0

    for meta_path in root.glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        src = meta.get("label_source") or "dtag"
        if src != "dtag" and not include_titles:
            continue
        if src not in ("dtag", "title", "title_vn"):
            continue
        if project_filter and meta.get("project_code") != project_filter:
            continue

        if src == "dtag":
            src_hse = meta_path.parent.parent.name
            src_loc = (meta.get("location_en") or "").replace(" ", "_")
            cons_hse = hse_map.get(src_hse, src_hse)
            cons_loc = loc_map.get(src_loc, src_loc) if src_loc else None

            # If the DTag source slug isn't in our 12-class consolidation
            # (i.e. a project uses a new variant we haven't seen), LLM-map
            # it the same way as titles — treating the slug text as a
            # violation description.
            valid_hse = {h["slug"] for h in (tax["hse_types"] if tax else [])}
            valid_loc = {l["slug"] for l in (tax["locations"] if tax else [])}
            if tax and cons_hse not in valid_hse:
                pseudo_title = src_hse.replace("_", " ")
                cache_key = f"slug::{pseudo_title}"
                if cache_key in title_cache:
                    cache_hits += 1
                    cached = title_cache[cache_key]
                    cons_hse = cached.get("hse") or cons_hse
                    if cached.get("loc") and cons_loc not in valid_loc:
                        cons_loc = cached["loc"]
                else:
                    hse2, loc2 = _map_title_via_llm(pseudo_title, "", tax)
                    llm_calls += 1
                    title_cache[cache_key] = {"hse": hse2, "loc": loc2}
                    if hse2:
                        cons_hse = hse2
                    if loc2 and cons_loc not in valid_loc:
                        cons_loc = loc2
            # Also map unknown locations
            if tax and cons_loc and cons_loc not in valid_loc:
                cache_key = f"loc::{cons_loc}"
                if cache_key in title_cache:
                    cache_hits += 1
                    cons_loc = title_cache[cache_key].get("loc") or None
                else:
                    _, loc3 = _map_title_via_llm(cons_loc.replace("_", " "), "", tax)
                    llm_calls += 1
                    title_cache[cache_key] = {"hse": None, "loc": loc3}
                    cons_loc = loc3
        else:
            # LLM-map the title into the 12-class taxonomy (cached by title)
            title_en = (meta.get("issue_title_en") or "").strip()
            title_vn = (meta.get("issue_title_vn") or "").strip()
            cache_key = f"{title_en}||{title_vn}"
            if cache_key in title_cache:
                cache_hits += 1
                cached = title_cache[cache_key]
                cons_hse = cached.get("hse")
                cons_loc = cached.get("loc")
            else:
                if tax is None:
                    continue
                hse, loc = _map_title_via_llm(title_en, title_vn, tax)
                llm_calls += 1
                title_cache[cache_key] = {"hse": hse, "loc": loc}
                cons_hse = hse
                cons_loc = loc
            if not cons_hse:
                log.warning("Could not map title %r — skipping", title_en[:60])
                continue

        for ph in (meta.get("photos") or []):
            fname = ph.get("file")
            if not fname:
                continue
            img = meta_path.parent / fname
            if not img.exists():
                continue
            out.append({
                "img": img,
                "sha256": ph.get("sha256"),
                "bytes": ph.get("bytes") or 0,
                "hse_slug": cons_hse,
                "loc_slug": cons_loc,
                "project_code": meta.get("project_code") or "SVN",
                "issue_id": meta.get("issue_id"),
                "original_filename": fname,
                "label_source_original": src,
            })

    # Always save cache if anything new was added (slug-LLM calls happen even
    # when --no-titles is active, so we want to persist them).
    if cache_hits or llm_calls:
        _save_title_cache(title_cache)
        log.info("Title/slug mapping: %d cache hits, %d LLM calls (cache saved to %s)",
                 cache_hits, llm_calls, _TITLE_CACHE_PATH)

    # Final FK-safety filter: drop records whose cons_hse/cons_loc aren't in
    # the current taxonomy (LLM may have returned null for admin-style slugs
    # even in the dtag path).
    if tax:
        valid_hse_final = {h["slug"] for h in tax["hse_types"]}
        valid_loc_final = {l["slug"] for l in tax["locations"]}
        before_filter = len(out)
        out = [r for r in out if r["hse_slug"] in valid_hse_final]
        # location can be None (allowed); if non-null it must be valid
        for r in out:
            if r["loc_slug"] and r["loc_slug"] not in valid_loc_final:
                r["loc_slug"] = None
        dropped = before_filter - len(out)
        if dropped:
            log.info("Dropped %d records with unmappable hse slug (FK safety)", dropped)
    # Dedupe by sha256 — same photo attached to multiple issues
    seen, dedup = set(), []
    for r in out:
        if r["sha256"] and r["sha256"] in seen:
            continue
        if r["sha256"]:
            seen.add(r["sha256"])
        dedup.append(r)
    return dedup


def _resolve_tenant_project(db) -> tuple[str, str]:
    tenants = db.table("tenants").select("id, name").eq("name", DEFAULT_TENANT_NAME).execute().data or []
    if not tenants:
        raise RuntimeError(f"Default tenant {DEFAULT_TENANT_NAME!r} not found — start the webapp once so it's created.")
    tid = tenants[0]["id"]
    projects = (
        db.table("projects").select("id, code").eq("tenant_id", tid)
          .eq("code", DEFAULT_PROJECT_CODE).execute().data or []
    )
    if not projects:
        raise RuntimeError(f"Default project {DEFAULT_PROJECT_CODE!r} not found.")
    return tid, projects[0]["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--project", type=str, default=None,
                    help="Only process this AECIS project_code (e.g. SVN).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max photos to process this run.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-titles", action="store_true", default=True,
                    help="Include title-labeled photos by LLM-mapping to the taxonomy. "
                         "Default on. Use --no-titles to skip them.")
    ap.add_argument("--no-titles", action="store_true",
                    help="Skip title-labeled photos (DTag-only seed).")
    args = ap.parse_args()
    include_titles = args.include_titles and not args.no_titles

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    if not root.exists():
        log.error("Dataset root not found: %s", root)
        return 1

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    r2_account = os.environ.get("R2_ACCOUNT_ID")
    r2_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.environ.get("R2_BUCKET", "violation-bot")
    if not all([url, key, r2_account, r2_key, r2_secret]):
        log.error("Set SUPABASE_* + R2_* in .env")
        return 2

    from supabase import create_client
    import boto3
    db = create_client(url, key)
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
        aws_access_key_id=r2_key, aws_secret_access_key=r2_secret,
        region_name="auto",
    )

    tenant_id, project_id = _resolve_tenant_project(db)
    log.info("Using tenant=%s project=%s", tenant_id, project_id)

    hse_map, loc_map = _source_maps()
    tax = json.loads((REPO_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    records = _gather(root, args.project, hse_map, loc_map,
                      include_titles=include_titles, tax=tax)
    log.info("Gathered %d unique DTag-labeled photos", len(records))
    if args.limit:
        records = records[: args.limit]
        log.info("Limited to %d", len(records))

    # Check which sha256s are already in pgvector with any authoritative label
    existing_shas: set[str] = set()
    off = 0
    while True:
        b = (
            db.table("photo_embeddings").select("sha256, label_source")
              .range(off, off + 999).execute().data or []
        )
        if not b:
            break
        for r in b:
            if r.get("label_source") in ("dtag", "manual"):
                existing_shas.add(r["sha256"])
        if len(b) < 1000:
            break
        off += 1000
    log.info("Existing authoritative embeddings in pgvector: %d", len(existing_shas))

    # Check which photos are already in photos table (by tenant + sha)
    existing_photo_shas: set[str] = set()
    off = 0
    while True:
        b = (
            db.table("photos").select("sha256").eq("tenant_id", tenant_id)
              .range(off, off + 999).execute().data or []
        )
        if not b:
            break
        existing_photo_shas.update(r["sha256"] for r in b if r.get("sha256"))
        if len(b) < 1000:
            break
        off += 1000
    log.info("Existing photos row count for this tenant: %d", len(existing_photo_shas))

    # Lazy-import CLIP (heavy)
    from src.embeddings import embed_image

    created, skipped_photo, embedded, skipped_embed = 0, 0, 0, 0

    import hashlib
    from datetime import datetime, timezone

    for i, r in enumerate(records, 1):
        sha = r["sha256"]
        if not sha:
            try:
                with r["img"].open("rb") as f:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                sha = h.hexdigest()
            except Exception as e:  # noqa: BLE001
                log.warning("hash failed for %s: %s", r["img"], e)
                continue

        if i % 25 == 0 or i == len(records):
            log.info("[%d/%d] sha=%s  hse=%s  loc=%s  proj=%s",
                     i, len(records), sha[:10], r["hse_slug"], r["loc_slug"], r["project_code"])

        if args.dry_run:
            continue

        # -------- 1. R2 upload (skip if photos row already exists) --------
        ext = r["img"].suffix.lower() or ".jpg"
        storage_key = f"{tenant_id}/{project_id}/{sha[:2]}/{sha}{ext}"
        if sha in existing_photo_shas:
            skipped_photo += 1
        else:
            try:
                s3.put_object(
                    Bucket=r2_bucket, Key=storage_key,
                    Body=r["img"].read_bytes(),
                    ContentType="image/jpeg" if ext in (".jpg", ".jpeg") else "image/png",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("R2 put failed for %s: %s", sha[:10], e)
                continue

            # -------- 2. photos row --------
            photo = db.table("photos").insert({
                "tenant_id": tenant_id,
                "project_id": project_id,
                "storage_key": storage_key,
                "storage_bucket": r2_bucket,
                "sha256": sha,
                "original_filename": r["original_filename"],
                "bytes": r["img"].stat().st_size,
            }).execute().data[0]
            photo_id = photo["id"]
            existing_photo_shas.add(sha)

            # -------- 3. classifications row (ground truth masked as prediction) --------
            cls = db.table("classifications").insert({
                "photo_id": photo_id,
                "location_slug": r["loc_slug"],
                "hse_type_slug": r["hse_slug"],
                "location_confidence": 1.0,
                "hse_type_confidence": 1.0,
                "rationale": "synthetic: seeded from on-disk AECIS DTag ground truth",
                "model": "synthetic_dtag_ground_truth",
                "source": "manual",
                "is_current": True,
            }).execute().data[0]

            # -------- 4. corrections row (action=confirm) --------
            db.table("corrections").insert({
                "photo_id": photo_id,
                "classification_id": cls["id"],
                "action": "confirm",
                "location_slug": r["loc_slug"],
                "hse_type_slug": r["hse_slug"],
                "note": "auto-seeded from DTag ground truth",
            }).execute()
            created += 1

        # -------- 5. pgvector upsert (only if not already authoritative) --------
        if sha in existing_shas:
            skipped_embed += 1
            continue
        try:
            vec = embed_image(r["img"])
        except Exception as e:  # noqa: BLE001
            log.warning("embed failed for %s: %s", sha[:10], e)
            continue
        try:
            db.table("photo_embeddings").upsert({
                "sha256": sha,
                "hse_type_slug": r["hse_slug"],
                "location_slug": r["loc_slug"],
                "label_source": "manual",
                "project_code": r["project_code"],
                "issue_id": r["issue_id"],
                "source_path": f"autoseed/{r['project_code']}/{r['issue_id']}/{r['original_filename']}",
                "embedding": vec.tolist(),
            }, on_conflict="sha256").execute()
            existing_shas.add(sha)
            embedded += 1
        except Exception as e:  # noqa: BLE001
            log.warning("pgvector upsert failed for %s: %s", sha[:10], e)

    log.info("Done.")
    log.info("  created photo+classification+correction rows: %d", created)
    log.info("  skipped (photo already existed):               %d", skipped_photo)
    log.info("  added pgvector embeddings:                     %d", embedded)
    log.info("  skipped pgvector (already authoritative):      %d", skipped_embed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
