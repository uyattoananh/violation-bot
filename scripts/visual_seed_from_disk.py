"""Visual-review seed: go through each scraped photo, use VLM to read context
clues from the image itself (and optionally the title), and assign a label
ONLY if the photo depicts a real visual construction violation.

This is the smart version of `auto_seed_from_disk.py` — instead of trusting
the title text blindly, we show each photo to the VLM and let it decide:
  - Is this a construction violation?
  - If yes: which of the 13 classes?
  - If no (paperwork, portrait, office, administrative photo): skip

Only high-confidence classifications (≥ threshold) get added to pgvector
and the photos/classifications/corrections tables.

Idempotent: results cached by sha256 in .visual_seed_cache.json so re-runs
skip already-classified photos.

Usage:
  python scripts/visual_seed_from_disk.py --project AR
  python scripts/visual_seed_from_disk.py --project AR --limit 20       # test batch
  python scripts/visual_seed_from_disk.py --project AR --threshold 0.7  # stricter
  python scripts/visual_seed_from_disk.py --project AR --dry-run        # classify + cache, no DB writes
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("visualseed")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"
DEFAULT_TENANT_NAME = "Public Demo"
DEFAULT_PROJECT_CODE = "PUBLIC"
CACHE_PATH = REPO_ROOT / "scripts" / ".visual_seed_cache.json"


VISUAL_REVIEW_SYSTEM_PROMPT = """You are reviewing a photograph from a \
Vietnamese construction site to decide whether it should be added to a
safety-violation training dataset.

Your task:
  1. Look at the photo.
  2. Decide if it depicts a VISUAL safety violation on a construction site.
  3. If YES: classify it into one of the provided HSE-type and location slugs.
  4. If NO (paperwork photos, portraits, office scenes, document scans,
     meeting notes, equipment certificates, insurance forms, contractor
     registration photos, signed documents, blank/unreadable images): return
     null for both slugs. These photos must be excluded from training.

Use the issue title as AUXILIARY context — if it clearly indicates "remind
contractor to submit documents / update insurance / deliver papers" and the
photo shows a document or office scene, that's a skip. If the title is
administrative but the photo clearly shows a real hazard, classify the photo.

Output ONE JSON object, no prose, no markdown fences:

{
  "is_violation": true | false,
  "location": {"slug": "<one slug or null>", "confidence": <0..1>},
  "hse_type": {"slug": "<one slug or null>", "confidence": <0..1>},
  "reasoning": "<10-40 words>"
}

If is_violation is false, set both slugs to null and confidences to 0.
Use conservative confidence — < 0.6 means low certainty and the photo
will be skipped from training.
"""


def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")


def _encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    media_type = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp", "gif": "image/gif",
    }.get(suffix, "image/jpeg")
    data = path.read_bytes()
    return base64.standard_b64encode(data).decode("ascii"), media_type


def _call_vlm(image_path: Path, title_en: str, title_vn: str, tax: dict) -> dict:
    """Send photo + title context to the VLM. Return parsed JSON."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": os.environ.get("OPENROUTER_TITLE", "violation-bot"),
        },
    )
    hse_list = "\n".join(f"  - {h['slug']}: {h['label_en']}" for h in tax["hse_types"])
    loc_list = "\n".join(f"  - {l['slug']}: {l['label_en']}" for l in tax["locations"])
    b64, media_type = _encode_image(image_path)
    user_content = [
        {"type": "text",
         "text": f"HSE_TYPES:\n{hse_list}\n\nLOCATIONS:\n{loc_list}"},
        {"type": "image_url",
         "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        {"type": "text",
         "text": f'Issue title (EN): "{title_en}"\nIssue title (VN): "{title_vn}"\n\n'
                 'Classify the photo above. Return JSON only.'},
    ]
    resp = client.chat.completions.create(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        max_tokens=300,
        messages=[
            {"role": "system", "content": VISUAL_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    text = resp.choices[0].message.content or ""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(text[s : e + 1])


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


def _gather_photos(root: Path, project_filter: str | None) -> list[dict[str, Any]]:
    """Walk on-disk metadata; emit every photo record with enough context
    to classify + insert."""
    out: list[dict[str, Any]] = []
    for meta_path in root.glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if project_filter and meta.get("project_code") != project_filter:
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
                "title_en": meta.get("issue_title_en", "") or "",
                "title_vn": meta.get("issue_title_vn", "") or "",
                "primary_dtag_raw": meta.get("primary_dtag_raw", "") or "",
                "project_code": meta.get("project_code") or "",
                "issue_id": meta.get("issue_id", ""),
                "original_filename": fname,
            })
    # Dedupe by sha256
    seen, dedup = set(), []
    for r in out:
        if r["sha256"] and r["sha256"] in seen:
            continue
        if r["sha256"]:
            seen.add(r["sha256"])
        dedup.append(r)
    return dedup


def _resolve_tenant_project(db) -> tuple[str, str]:
    tenants = (
        db.table("tenants").select("id, name")
          .eq("name", DEFAULT_TENANT_NAME).execute().data or []
    )
    if not tenants:
        raise RuntimeError(f"Default tenant {DEFAULT_TENANT_NAME!r} not found")
    tid = tenants[0]["id"]
    projects = (
        db.table("projects").select("id, code").eq("tenant_id", tid)
          .eq("code", DEFAULT_PROJECT_CODE).execute().data or []
    )
    if not projects:
        raise RuntimeError(f"Default project {DEFAULT_PROJECT_CODE!r} not found")
    return tid, projects[0]["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--project", type=str, required=True,
                    help="AECIS project_code to process (e.g. AR)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max photos to process")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Min confidence on both axes to keep. Default 0.6.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Classify + cache, do not write to DB / R2 / pgvector")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        log.error("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required")
        return 2

    from supabase import create_client
    db = create_client(url, key)

    tax = json.loads((REPO_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    valid_hse = {h["slug"] for h in tax["hse_types"]}
    valid_loc = {l["slug"] for l in tax["locations"]}

    cache = _load_cache()
    records = _gather_photos(root, args.project)
    log.info("Gathered %d unique photos for project %s", len(records), args.project)
    if args.limit:
        records = records[: args.limit]
    log.info("Processing %d (threshold=%.2f)", len(records), args.threshold)

    # VLM pass — classify each photo (cached by sha256)
    tenant_id, project_id = (None, None)
    if not args.dry_run:
        tenant_id, project_id = _resolve_tenant_project(db)
        log.info("DB target: tenant=%s project=%s", tenant_id, project_id)
        # Existing sha256s to dedupe
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
        existing_embeddings: set[str] = set()
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
                    existing_embeddings.add(r["sha256"])
            if len(b) < 1000:
                break
            off += 1000

    # Lazy-import embed
    from src.embeddings import embed_image
    import boto3
    r2 = None
    if not args.dry_run:
        r2 = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    n_total = len(records)
    n_classified, n_kept, n_skipped_nonviolation, n_skipped_lowconf, n_errors = 0, 0, 0, 0, 0
    n_photos_created, n_embeds_upserted = 0, 0

    for i, r in enumerate(records, 1):
        sha = r["sha256"]
        if not sha:
            n_errors += 1
            continue

        # Cached classification?
        if sha in cache:
            decision = cache[sha]
        else:
            try:
                decision = _call_vlm(r["img"], r["title_en"], r["title_vn"], tax)
                cache[sha] = decision
                n_classified += 1
                # Periodic cache flush
                if n_classified % 20 == 0:
                    _save_cache(cache)
            except Exception as e:  # noqa: BLE001
                log.warning("VLM call failed for %s: %s", sha[:10], e)
                n_errors += 1
                continue

        is_violation = bool(decision.get("is_violation"))
        hse_slug = (decision.get("hse_type") or {}).get("slug")
        loc_slug = (decision.get("location") or {}).get("slug")
        hse_conf = float((decision.get("hse_type") or {}).get("confidence") or 0)
        loc_conf = float((decision.get("location") or {}).get("confidence") or 0)

        log_prefix = f"[{i}/{n_total}] {sha[:8]}"

        if not is_violation or not hse_slug or hse_slug not in valid_hse:
            n_skipped_nonviolation += 1
            if i % 25 == 0 or i == n_total:
                log.info("%s skip (non-violation)", log_prefix)
            continue
        if hse_conf < args.threshold or loc_conf < args.threshold:
            n_skipped_lowconf += 1
            if i % 25 == 0 or i == n_total:
                log.info("%s skip (low conf: hse=%.2f loc=%.2f)",
                         log_prefix, hse_conf, loc_conf)
            continue
        if loc_slug and loc_slug not in valid_loc:
            loc_slug = None

        n_kept += 1
        if i % 10 == 0 or i == n_total:
            log.info("%s keep hse=%s loc=%s conf=%.2f/%.2f",
                     log_prefix, hse_slug, loc_slug, hse_conf, loc_conf)

        if args.dry_run:
            continue

        # --- Write to DB + R2 + pgvector ---
        try:
            ext = r["img"].suffix.lower() or ".jpg"
            storage_key = f"{tenant_id}/{project_id}/{sha[:2]}/{sha}{ext}"
            if sha not in existing_photo_shas:
                r2.put_object(
                    Bucket=os.environ["R2_BUCKET"], Key=storage_key,
                    Body=r["img"].read_bytes(),
                    ContentType="image/jpeg" if ext in (".jpg", ".jpeg") else "image/png",
                )
                photo = db.table("photos").insert({
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "storage_key": storage_key,
                    "storage_bucket": os.environ["R2_BUCKET"],
                    "sha256": sha,
                    "original_filename": r["original_filename"],
                    "bytes": r["img"].stat().st_size,
                }).execute().data[0]
                photo_id = photo["id"]
                existing_photo_shas.add(sha)
                cls = db.table("classifications").insert({
                    "photo_id": photo_id,
                    "location_slug": loc_slug,
                    "hse_type_slug": hse_slug,
                    "location_confidence": loc_conf,
                    "hse_type_confidence": hse_conf,
                    "rationale": "visual-review: " + (decision.get("reasoning") or "")[:300],
                    "model": "vlm_visual_review",
                    "source": "manual",
                    "is_current": True,
                }).execute().data[0]
                db.table("corrections").insert({
                    "photo_id": photo_id,
                    "classification_id": cls["id"],
                    "action": "confirm",
                    "location_slug": loc_slug,
                    "hse_type_slug": hse_slug,
                    "note": "auto-seeded from visual VLM review",
                }).execute()
                n_photos_created += 1

            if sha not in existing_embeddings:
                vec = embed_image(r["img"])
                db.table("photo_embeddings").upsert({
                    "sha256": sha,
                    "hse_type_slug": hse_slug,
                    "location_slug": loc_slug,
                    "label_source": "manual",
                    "project_code": r["project_code"],
                    "issue_id": r["issue_id"],
                    "source_path": f"visualseed/{r['project_code']}/{r['issue_id']}/{r['original_filename']}",
                    "embedding": vec.tolist(),
                }, on_conflict="sha256").execute()
                existing_embeddings.add(sha)
                n_embeds_upserted += 1
        except Exception as e:  # noqa: BLE001
            log.warning("DB/R2 write failed for %s: %s", sha[:10], e)
            n_errors += 1

    _save_cache(cache)
    log.info("")
    log.info("Done.")
    log.info("  VLM calls made:                  %d", n_classified)
    log.info("  Kept (violation, above threshold): %d", n_kept)
    log.info("  Skipped (non-violation):         %d", n_skipped_nonviolation)
    log.info("  Skipped (low confidence):        %d", n_skipped_lowconf)
    log.info("  Errors:                          %d", n_errors)
    if not args.dry_run:
        log.info("  photos+classifications+corrections created: %d", n_photos_created)
        log.info("  pgvector embeddings upserted:                %d", n_embeds_upserted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
