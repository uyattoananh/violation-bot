"""LLM-driven mining for rare HSE classes.

Strategy: scan the AECIS manifest for entries whose text mentions
keywords associated with each rare class. For each candidate photo,
send a TARGETED vision query to Gemini Flash 2.5: "does this photo
depict <class_name>?" If confirmed at confidence >= --min-vision-
confidence, CLIP-embed and upsert as `label_source='aecis_curated_rare_v1'`.

This is the existing vision-verify gate, flipped: instead of
confirming the text-classifier's guess across the 29-class space,
we hunt for ONE specific class at a time. Much higher recall for
rare classes that the multi-class classifier rarely picks.

No human judgement required — the LLM does the visual classification.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/mine_rare_class_seeds.py \\
        --class Welding_unsafe --max-confirms 30
    ./.venv-webapp/Scripts/python.exe scripts/mine_rare_class_seeds.py \\
        --all
    ./.venv-webapp/Scripts/python.exe scripts/mine_rare_class_seeds.py \\
        --class Truck_vehicle_unsafe --dry-run
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from openai import OpenAI

LABEL_SOURCE = "aecis_curated_rare_v1"
MANIFEST_PATH = REPO / "Issue_Gen" / "photos" / "manifest.jsonl"
PHOTOS_ROOT = REPO / "Issue_Gen" / "photos"
CACHE_PATH = REPO / "scripts" / ".rare_mine_cache.json"

# Per-rare-class keyword bank. Pulled from the AECIS taxonomy +
# common Vietnamese construction-site terms. Lower-cased,
# substring match (not whole-word) so plural / inflected forms hit.
RARE_CLASSES: dict[str, list[str]] = {
    "Welding_unsafe":            ["weld", "welding", "hàn", "hàn điện", "hàn hơi"],
    "Truck_vehicle_unsafe":      ["truck", "xe tải", "xe lùi", "reversing", "vehicle"],
    "Mass_piling_unsafe":        ["pile head", "piling", "cọc", "đầu cọc", "ép cọc"],
    "Smoking_area_unsafe":       ["smoke", "smoking", "hút thuốc", "thuốc lá"],
    "Concrete_work_unsafe":      ["concrete", "bê tông", "đổ bê tông", "rebar"],
    "Site_lighting_unsafe":      ["lighting", "ánh sáng", "đèn", "lamp", "lit"],
    "Garbage_waste_unsafe":      ["garbage", "waste", "rác", "rubbish", "thải"],
    "Pressure_equipment_unsafe": ["pressure", "áp lực", "compressor", "boiler"],
    "Formwork_unsafe":           ["formwork", "cốp pha", "shutter"],
    "Common_area_unsafe":        ["common area", "khu chung", "passage", "corridor"],
    "First_aid_kit_unsafe":      ["first aid", "sơ cứu", "y tế", "medical"],
    "Parking_area_unsafe":       ["parking", "bãi đỗ", "bãi xe", "garage"],
    "Confined_space_unsafe":     ["confined", "không gian kín", "tank", "vault"],
    "Chemicals_hazmat_unsafe":   ["chemical", "hóa chất", "hazmat", "toxic", "drum", "phuy"],
    "Ladder_unsafe":             ["ladder", "thang", "stair"],
}


def _safe(fp: str) -> str:
    out = fp
    for c in ':*?"<>|':
        out = out.replace(c, "_")
    return out


VISION_PROMPT = """Look at this construction-site photograph.

Decide: does this photo CLEARLY depict the following safety
violation type?

  category: {slug}
  description: {desc}

Be strict. Only return match=true if the photo unambiguously shows
this specific category — workers, equipment, or site features that
directly correspond to the category. If the photo shows a different
violation, an unrelated scene, paperwork, finished surfaces, or
"general site mess", return match=false.

Return ONE JSON object, no prose, no markdown fences:
{{"match": true | false, "confidence": <0..1>,
  "reasoning": "<10-25 words: what made you decide>"}}
"""

# Short English descriptions per class to anchor the vision-verify
# call (Gemini interprets the slug + description together).
CLASS_DESCRIPTIONS: dict[str, str] = {
    "Welding_unsafe":            "active welding/hot-work without proper protection (sparks, no PPE, flammable materials nearby)",
    "Truck_vehicle_unsafe":      "trucks, lorries, or heavy vehicles operating unsafely (no spotter, reversing alarm missing, overloaded, blocking traffic)",
    "Mass_piling_unsafe":        "pile-driving operations without proper marking, barriers, or worker protection",
    "Smoking_area_unsafe":       "smoking in an unauthorized area, or unsafe smoking near flammables",
    "Concrete_work_unsafe":      "active concrete pouring, casting, or rebar work with safety issues (poor formwork, no protection)",
    "Site_lighting_unsafe":      "inadequate or absent lighting in a work area where lighting is required",
    "Garbage_waste_unsafe":      "accumulated garbage or waste creating a specific hazard (fire risk, attracting pests, blocking egress)",
    "Pressure_equipment_unsafe": "pressure vessels, compressors, gas cylinders without inspection, certification, or proper handling",
    "Formwork_unsafe":           "concrete formwork (shuttering) that is improperly braced, damaged, or unstable",
    "Common_area_unsafe":        "shared circulation areas (passages, corridors, walkways) with hazards blocking safe passage",
    "First_aid_kit_unsafe":      "first aid station that is missing, inaccessible, or improperly stocked",
    "Parking_area_unsafe":       "vehicle parking areas with hazards (no markings, blocking emergency exits, on uneven ground)",
    "Confined_space_unsafe":     "entry to confined spaces (tanks, vaults, pits) without proper permits, ventilation, or rescue equipment",
    "Chemicals_hazmat_unsafe":   "chemical drums, hazmat materials, or fuels stored or handled unsafely",
    "Ladder_unsafe":             "ladders used unsafely (broken, leaning at wrong angle, not secured, exceeding safe height)",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_image(p: Path) -> tuple[str, str]:
    suffix = p.suffix.lower().lstrip(".")
    mt = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png",
          "webp":"image/webp","heic":"image/heic"}.get(suffix, "image/jpeg")
    return base64.standard_b64encode(p.read_bytes()).decode("ascii"), mt


def vision_verify(or_client: OpenAI, img_path: Path, slug: str,
                  description: str, model: str) -> dict:
    """Ask Gemini: does this photo depict the given class?"""
    b64, mt = _encode_image(img_path)
    sys_p = VISION_PROMPT.format(slug=slug, desc=description)
    try:
        r = or_client.chat.completions.create(
            model=model, max_tokens=200,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mt};base64,{b64}"}},
                ]},
            ],
        )
    except Exception as e:  # noqa: BLE001
        return {"match": False, "confidence": 0.0,
                "reasoning": f"vision call failed: {str(e)[:120]}"}
    raw = (r.choices[0].message.content or "").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return {"match": False, "confidence": 0.0,
                "reasoning": f"no JSON: {raw[:80]}"}
    try:
        j = json.loads(raw[s:e+1])
        return {
            "match": bool(j.get("match")),
            "confidence": float(j.get("confidence") or 0),
            "reasoning": (j.get("reasoning") or "")[:200],
        }
    except Exception as ex:  # noqa: BLE001
        return {"match": False, "confidence": 0.0,
                "reasoning": f"parse err: {ex}"}


def keyword_candidates(target_slug: str) -> list[dict]:
    """Scan manifest for rows whose issue_name or description matches
    the target class's keywords. Returns deduped-by-filepath list."""
    if target_slug not in RARE_CLASSES:
        sys.stderr.write(f"unknown rare class: {target_slug}\n")
        sys.exit(2)
    keywords = [k.lower() for k in RARE_CLASSES[target_slug]]
    if not MANIFEST_PATH.exists():
        sys.stderr.write(f"missing manifest: {MANIFEST_PATH}\n")
        sys.exit(2)
    seen: set[str] = set()
    out: list[dict] = []
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = rec.get("filepath", "")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            text = ((rec.get("issue_name", "") or "") + " " +
                    (rec.get("description", "") or "")).lower()
            if any(kw in text for kw in keywords):
                local = PHOTOS_ROOT / _safe(fp)
                if local.exists() and local.stat().st_size >= 1024:
                    out.append({**rec, "_local": str(local)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", type=str, default="")
    ap.add_argument("--all", action="store_true",
                    help="mine all rare classes in sequence")
    ap.add_argument("--max-confirms", type=int, default=30,
                    help="stop after N confirmed embeds per class")
    ap.add_argument("--max-checks", type=int, default=150,
                    help="cap vision calls per class (cost control)")
    ap.add_argument("--min-vision-confidence", type=float, default=0.7)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.cls:
        ap.error("specify --class <slug> or --all")

    targets = list(RARE_CLASSES.keys()) if args.all else [args.cls]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write("OPENROUTER_API_KEY not set\n")
        return 2
    or_client = OpenAI(api_key=api_key,
                       base_url="https://openrouter.ai/api/v1")

    db = create_client(os.environ["SUPABASE_URL"],
                       os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # tenant lookup
    try:
        tenant = (db.table("tenants").select("id").eq("name", "Public Demo")
                  .limit(1).execute().data or [])
        tenant_id = tenant[0]["id"] if tenant else None
    except Exception:  # noqa: BLE001
        tenant_id = None

    # Cache prior verdicts so reruns are free for already-judged photos
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}

    # Lazy-load CLIP only on first real embed
    embed_image = None
    grand_totals = {"checked": 0, "confirmed": 0, "embedded": 0}

    for slug in targets:
        candidates = keyword_candidates(slug)
        print(f"\n=== {slug} ===")
        print(f"  keyword-matched candidates in manifest: {len(candidates)}")

        # Filter out already-embedded shas (any label_source counts)
        # plus already-cache-hit-rejected
        n_checked = n_confirmed = n_embedded = 0
        for rec in candidates:
            if n_checked >= args.max_checks:
                print(f"  reached --max-checks {args.max_checks}, stopping class")
                break
            if n_confirmed >= args.max_confirms:
                print(f"  reached --max-confirms {args.max_confirms}, stopping class")
                break
            local = Path(rec["_local"])
            sha = _sha256(local)

            # Skip if already in DB under any label_source (avoid duplicates)
            existing = (db.table("photo_embeddings").select("id")
                        .eq("sha256", sha).limit(1).execute().data or [])
            if existing:
                continue

            n_checked += 1
            cache_key = f"{sha}|{slug}"
            v = cache.get(cache_key)
            if v is None:
                v = vision_verify(or_client, local, slug,
                                  CLASS_DESCRIPTIONS.get(slug, slug),
                                  args.model)
                cache[cache_key] = v
                # Save cache periodically
                if n_checked % 10 == 0:
                    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    tmp = CACHE_PATH.with_suffix(".tmp")
                    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
                    tmp.replace(CACHE_PATH)

            if not v.get("match") or v.get("confidence", 0) < args.min_vision_confidence:
                continue
            n_confirmed += 1
            if args.dry_run:
                continue

            # Lazy-load CLIP
            if embed_image is None:
                from src.embeddings import embed_image as _embed
                embed_image = _embed
            try:
                emb = embed_image(local).tolist()
                # Use "Common_working_area" as a safe default location
                # (rare classes often photographed in generic locations)
                db.table("photo_embeddings").upsert({
                    "sha256": sha,
                    "hse_type_slug": slug,
                    "location_slug": "Common_working_area",
                    "label_source": LABEL_SOURCE,
                    "project_code": f"P_{rec.get('project_id') or 'AECIS'}",
                    "issue_id": str(rec.get("issue_id") or ""),
                    "source_path": f"aecis_curated_rare/{rec.get('filepath','')}",
                    "embedding": emb,
                    "tenant_id": tenant_id,
                }, on_conflict="sha256").execute()
                n_embedded += 1
                if n_embedded % 5 == 0:
                    print(f"    embedded {n_embedded}…")
            except Exception as e:  # noqa: BLE001
                print(f"    embed err: {e}")

        print(f"  ▸ checked={n_checked}  confirmed={n_confirmed}  embedded={n_embedded}")
        grand_totals["checked"] += n_checked
        grand_totals["confirmed"] += n_confirmed
        grand_totals["embedded"] += n_embedded

    # Final cache flush
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(CACHE_PATH)

    print(f"\n=== grand totals across {len(targets)} classes ===")
    for k, v in grand_totals.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
