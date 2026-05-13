"""VLM-gate validator for the AECIS seed dataset.

Same pattern as scripts/visual_seed_from_disk.py: every candidate
photo goes to Sonnet vision with the issue metadata as auxiliary
context. Sonnet decides:
  1. Is this an HSE violation? (paperwork / portrait / office = skip)
  2. If yes, what's the hse_type + location?
  3. Confidence on both axes (< threshold = skip)

Only photos that pass both gates land in the validated manifest.
The orchestrator (scripts/seed_with_checkpoints.py) reads the
validated manifest instead of the raw downloader output, so the
batches of 200 you upload are *pre-screened* — not 200 random
AECIS rows, 200 photos already confirmed as real violations with
high-confidence labels.

INPUT:  Issue_Gen/photos/manifest.jsonl   (from seed_download_aecis_photos.py)
OUTPUT: Issue_Gen/photos/manifest.validated.jsonl
CACHE:  scripts/.seed_validate_aecis_cache.json   (per-sha256 verdicts)

Usage:
  ./.venv-webapp/Scripts/python.exe scripts/seed_validate_aecis.py \\
      [--limit N]              # cap photos (0 = all)
      [--threshold 0.6]        # min confidence on both axes
      [--workers 4]            # parallel LLM calls
      [--dry-run]              # print plan, no API calls

Cost: ~$0.014 per photo via OpenRouter / Sonnet 4.5 with vision.
2,343 photos -> ~$33 total. With prompt caching warm, ~$0.008/photo.
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MANIFEST_IN = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.jsonl"
MANIFEST_OUT = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.validated.jsonl"
CACHE_PATH = REPO_ROOT / "scripts" / ".seed_validate_aecis_cache.json"
PHOTOS_ROOT = REPO_ROOT / "Issue_Gen" / "photos"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Same prompt skeleton as visual_seed_from_disk.py — kept aligned so
# the AECIS gate produces the same shape of judgement as the older
# disk-based gate. Only the wording tweaks reflect that AECIS issues
# carry an issue_name + description rather than title_en + title_vn.
VALIDATION_SYSTEM_PROMPT = """You are reviewing a photograph from a \
Vietnamese / multi-country construction site to decide whether it
should be added to a safety-violation training dataset.

Your task:
  1. Look at the photo.
  2. Decide if it depicts a VISUAL safety violation on a construction site.
  3. If YES: classify it into one of the provided HSE-type and location slugs.
  4. If NO (paperwork, portraits, office scenes, document scans, meeting
     notes, equipment certificates, blank/unreadable, finishing-defect
     photos that aren't safety hazards like "poor workmanship" / "wrong
     paint color"): return null for both slugs and is_violation=false.
     These photos must be excluded from training.

The issue_name and description come from AECIS's inspection record and
are AUXILIARY context only. Trust the photo more than the metadata —
some issues are mis-filed into the HSE discipline despite being
workmanship complaints.

Output ONE JSON object, no prose, no markdown fences:

{
  "is_violation": true | false,
  "location": {"slug": "<one slug or null>", "confidence": <0..1>},
  "hse_type": {"slug": "<one slug or null>", "confidence": <0..1>},
  "reasoning": "<10-40 words: what made you say yes or no>"
}

If is_violation is false, set both slugs to null and confidences to 0.
Use conservative confidence — < 0.6 means low certainty and the photo
will be excluded from training.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_image(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    media_type = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
    }.get(suffix, "image/jpeg")
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return b64, media_type


def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(c: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _load_taxonomy() -> dict:
    """Pull the same taxonomy the production classifier uses."""
    path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    if path.exists():
        parents = json.loads(path.read_text(encoding="utf-8")).get("parents", {})
    else:
        parents = {}
    # Build the flat hse_types + locations the prompt needs.
    hse_types = [
        {"slug": k, "label_en": k.replace("_", " ").title()}
        for k in sorted(parents.keys())
    ]
    locations = [
        {"slug": "Common_working_area", "label_en": "Common working area"},
        {"slug": "Confined_space",      "label_en": "Confined space"},
        {"slug": "Excavation_or_pit",   "label_en": "Excavation / pit"},
        {"slug": "Height_work",         "label_en": "Work at height"},
        {"slug": "Storage_area",        "label_en": "Storage area"},
        {"slug": "Traffic_route",       "label_en": "Traffic route"},
        {"slug": "Mechanical_zone",     "label_en": "Mechanical zone"},
        {"slug": "Electrical_zone",     "label_en": "Electrical zone"},
        {"slug": "Fire_exit",           "label_en": "Fire exit"},
    ]
    return {"hse_types": hse_types, "locations": locations}


def _call_vlm(img_path: Path, issue_name: str, description: str,
              tax: dict, model: str) -> dict:
    """Send photo + AECIS metadata to the VLM. Return parsed JSON."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": os.environ.get("OPENROUTER_TITLE", "violation-bot-seed"),
        },
    )
    hse_list = "\n".join(f"  - {h['slug']}: {h['label_en']}" for h in tax["hse_types"])
    loc_list = "\n".join(f"  - {l['slug']}: {l['label_en']}" for l in tax["locations"])
    b64, media_type = _encode_image(img_path)
    user_content = [
        {"type": "text",
         "text": f"HSE_TYPES:\n{hse_list}\n\nLOCATIONS:\n{loc_list}"},
        {"type": "image_url",
         "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        {"type": "text",
         "text": (
             f'AECIS issue name: "{issue_name}"\n'
             f'AECIS description: "{description[:400]}"\n\n'
             "Classify the photo above. Return JSON only."
         )},
    ]
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )
    text = resp.choices[0].message.content or ""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(text[s : e + 1])


def _validate_one(rec: dict, tax: dict, model: str,
                  threshold: float, cache: dict) -> dict:
    """Returns the input rec annotated with validation verdict.
    Uses the cache (keyed by sha256) to avoid re-paying for already-
    judged photos."""
    img = PHOTOS_ROOT / rec["filepath"]
    if not img.exists():
        return {**rec, "verdict": "missing_local_file", "passed": False}
    sha = rec.get("sha256") or _sha256(img)
    rec = {**rec, "sha256": sha}

    cached = cache.get(sha)
    if cached is not None:
        return {**rec, **cached, "from_cache": True}

    try:
        v = _call_vlm(img, rec.get("issue_name", ""),
                      rec.get("description", ""), tax, model)
    except Exception as e:  # noqa: BLE001
        verdict = {
            "verdict": "vlm_error",
            "passed": False,
            "error": str(e)[:200],
        }
        cache[sha] = verdict
        return {**rec, **verdict, "from_cache": False}

    hse_conf = float((v.get("hse_type") or {}).get("confidence") or 0)
    loc_conf = float((v.get("location") or {}).get("confidence") or 0)
    is_viol = bool(v.get("is_violation"))
    hse_slug = (v.get("hse_type") or {}).get("slug")
    loc_slug = (v.get("location") or {}).get("slug")
    passed = (is_viol and hse_conf >= threshold and loc_conf >= threshold
              and hse_slug and loc_slug)
    verdict = {
        "verdict": "passed" if passed else (
            "non_violation" if not is_viol
            else "low_confidence"),
        "passed": passed,
        "vlm_hse_type_slug": hse_slug,
        "vlm_location_slug": loc_slug,
        "vlm_hse_conf": hse_conf,
        "vlm_loc_conf": loc_conf,
        "vlm_reasoning": v.get("reasoning", "")[:200],
    }
    cache[sha] = verdict
    return {**rec, **verdict, "from_cache": False}


def _iter_input_records(limit: int = 0):
    """Yield manifest entries, dedup'd by filepath."""
    if not MANIFEST_IN.exists():
        sys.stderr.write(f"ERROR: missing {MANIFEST_IN}\n"
                         "Run scripts/seed_download_aecis_photos.py first.\n")
        sys.exit(2)
    seen, n = set(), 0
    with MANIFEST_IN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = rec.get("filepath")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            yield rec
            n += 1
            if limit and n >= limit:
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel VLM calls; respect OpenRouter RPM")
    ap.add_argument("--model", default=os.environ.get(
        "OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.stderr.write("ERROR: OPENROUTER_API_KEY not set in env / .env\n")
        return 2

    tax = _load_taxonomy()
    cache = _load_cache()
    records = list(_iter_input_records(args.limit))
    sys.stdout.write(f"input: {len(records)} unique photos\n")
    sys.stdout.write(f"model: {args.model}\n")
    sys.stdout.write(f"threshold: {args.threshold}\n")
    sys.stdout.write(f"cache: {len(cache)} prior verdicts\n")
    if args.dry_run:
        for r in records[:5]:
            sys.stdout.write(f"  would judge: {r.get('filepath')}\n")
        return 0

    counts = {"passed": 0, "non_violation": 0, "low_confidence": 0,
              "vlm_error": 0, "missing_local_file": 0}
    fresh_api_calls = 0
    t0 = time.perf_counter()

    out_fp = MANIFEST_OUT.open("w", encoding="utf-8")

    def _task(rec):
        return _validate_one(rec, tax, args.model, args.threshold, cache)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for result in ex.map(_task, records):
                counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
                if not result.get("from_cache", False) and result["verdict"] != "missing_local_file":
                    fresh_api_calls += 1
                if result["passed"]:
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed = sum(counts.values())
                if processed % 25 == 0:
                    sys.stdout.write(
                        f"  {processed}/{len(records)}  "
                        f"pass={counts['passed']} "
                        f"non={counts['non_violation']} "
                        f"low={counts['low_confidence']} "
                        f"err={counts.get('vlm_error', 0)}\n"
                    )
                # Persist cache periodically so a Ctrl-C doesn't lose
                # all the API spend.
                if fresh_api_calls % 50 == 49:
                    _save_cache(cache)
    finally:
        out_fp.close()
        _save_cache(cache)

    elapsed = time.perf_counter() - t0
    pass_rate = counts["passed"] / max(1, len(records))
    sys.stdout.write(
        f"\nDone in {elapsed:.1f}s — "
        f"passed={counts['passed']} ({pass_rate:.1%})  "
        f"non={counts['non_violation']}  low={counts['low_confidence']}  "
        f"err={counts.get('vlm_error', 0)}  missing={counts.get('missing_local_file', 0)}\n"
        f"validated manifest: {MANIFEST_OUT}\n"
        f"fresh API calls: {fresh_api_calls}  (cached: {sum(counts.values()) - fresh_api_calls - counts.get('missing_local_file', 0)})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
