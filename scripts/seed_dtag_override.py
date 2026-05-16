"""Post-process the seed_validate_aecis cache: any photo the VLM
rejected as `non_violation` whose sha256 is ALREADY in
photo_embeddings with label_source='dtag' is an authoritative
violation — the dtag label was assigned by the prior inspector-
validated pipeline. Override the VLM verdict and inherit the
dtag-side hse_type_slug + location_slug.

Why: the VLM occasionally misreads ambiguous photos (e.g. a
finished bathroom with a real safety hazard tucked in the
corner) as workmanship-only. If the same photo already has a
dtag verdict in the DB, that prior judgement wins.

This script is idempotent: re-running just refreshes the
dtag→cache override and rewrites manifest.validated.jsonl. It
does NOT call OpenRouter — pure DB read + JSON rewrite.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/seed_dtag_override.py
    ./.venv-webapp/Scripts/python.exe scripts/seed_dtag_override.py --dry-run
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

CACHE_PATH    = REPO_ROOT / "scripts" / ".seed_validate_aecis_cache.json"
MANIFEST_IN   = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.jsonl"
MANIFEST_OUT  = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.validated.jsonl"
PHOTOS_ROOT   = REPO_ROOT / "Issue_Gen" / "photos"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_dtag_index() -> dict[str, dict]:
    """Returns {sha256: {hse_type_slug, location_slug}} for all
    photo_embeddings rows with label_source='dtag'. Pages via
    range() so we don't hit Supabase's default 1k row cap."""
    from supabase import create_client
    db = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    out: dict[str, dict] = {}
    page_size = 1000
    offset = 0
    while True:
        rows = (db.table("photo_embeddings")
                  .select("sha256, hse_type_slug, location_slug")
                  .eq("label_source", "dtag")
                  .range(offset, offset + page_size - 1)
                  .execute().data or [])
        for r in rows:
            if r.get("sha256"):
                out[r["sha256"]] = {
                    "hse_type_slug": r.get("hse_type_slug"),
                    "location_slug": r.get("location_slug"),
                }
        if len(rows) < page_size:
            break
        offset += page_size
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report overlap but don't write changes")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="confidence threshold for promoting overrides "
                         "into the validated manifest (same default as "
                         "seed_validate_aecis.py)")
    args = ap.parse_args()

    if not CACHE_PATH.exists():
        log.error("cache missing at %s — run seed_validate_aecis.py first", CACHE_PATH)
        return 2
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    log.info("cache entries: %d", len(cache))

    log.info("loading dtag-labeled sha256s from photo_embeddings…")
    dtag_idx = load_dtag_index()
    log.info("dtag rows in DB: %d", len(dtag_idx))

    non_viol = [sha for sha, v in cache.items()
                if v.get("verdict") == "non_violation"]
    log.info("cache non_violation entries: %d", len(non_viol))

    overlap = [sha for sha in non_viol if sha in dtag_idx]
    log.info("non_violation entries that exist in dtag set: %d", len(overlap))

    if not overlap:
        log.info("no overrides to apply.")
        return 0

    if args.dry_run:
        log.info("DRY RUN — first 10 overrides that would be applied:")
        for sha in overlap[:10]:
            d = dtag_idx[sha]
            log.info("  sha=%s  →  hse=%s  loc=%s",
                     sha[:12], d["hse_type_slug"], d["location_slug"])
        log.info("(total %d)", len(overlap))
        return 0

    # Apply the overrides to cache. Keep VLM verdict in vlm_* fields
    # so the override is auditable, and promote dtag verdict on top.
    for sha in overlap:
        d = dtag_idx[sha]
        prev = cache[sha]
        cache[sha] = {
            **prev,
            "verdict": "dtag_override",
            "passed": True,
            "vlm_hse_type_slug": prev.get("vlm_hse_type_slug"),
            "vlm_location_slug": prev.get("vlm_location_slug"),
            "vlm_hse_conf": prev.get("vlm_hse_conf"),
            "vlm_loc_conf": prev.get("vlm_loc_conf"),
            # Authoritative labels come from the dtag row.
            "dtag_hse_type_slug": d["hse_type_slug"],
            "dtag_location_slug": d["location_slug"],
            "override_reason": "sha256 present in photo_embeddings with label_source='dtag'",
        }

    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)
    log.info("cache rewritten with %d overrides", len(overlap))

    # Rebuild manifest.validated.jsonl from cache + raw manifest.
    # Emit one entry per manifest row whose sha256 verdict is
    # `passed` or `dtag_override`.
    log.info("rebuilding %s…", MANIFEST_OUT.name)
    import hashlib
    def sha_of(fp: str) -> str:
        safe = fp
        for c in ':*?"<>|':
            safe = safe.replace(c, "_")
        p = PHOTOS_ROOT / safe
        if not p.exists():
            return ""
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    n_in = n_kept = 0
    with MANIFEST_OUT.open("w", encoding="utf-8") as out_f, \
         MANIFEST_IN.open(encoding="utf-8") as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            sha = rec.get("sha256") or sha_of(rec.get("filepath", ""))
            if not sha:
                continue
            v = cache.get(sha)
            if not v or not v.get("passed"):
                continue
            # promote dtag label if override
            if v.get("verdict") == "dtag_override":
                hse_slug = v.get("dtag_hse_type_slug")
                loc_slug = v.get("dtag_location_slug")
            else:
                hse_slug = v.get("vlm_hse_type_slug")
                loc_slug = v.get("vlm_location_slug")
            if not hse_slug or not loc_slug:
                continue
            out_rec = {
                **rec,
                "sha256": sha,
                "verdict": v.get("verdict"),
                "vlm_hse_type_slug": hse_slug,
                "vlm_location_slug": loc_slug,
                "vlm_hse_conf": v.get("vlm_hse_conf"),
                "vlm_loc_conf": v.get("vlm_loc_conf"),
                "vlm_reasoning": v.get("vlm_reasoning", ""),
                "passed": True,
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_kept += 1
    log.info("manifest: %d input rows, %d passed/override kept", n_in, n_kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
