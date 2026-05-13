"""Run the seed-validation prompt across multiple candidate models
on the same N AECIS photos and emit a side-by-side comparison.

Use this to decide which model is worth the full $45+ run before
spending it. Cheaper models that broadly agree with Sonnet for
SAY 90%+ of the validation verdicts are the safe budget pick;
ones that disagree on >30% of photos shouldn't be used as judges.

Usage:
  ./.venv-webapp/Scripts/python.exe scripts/compare_seed_validators.py \\
      [--n 10] \\
      [--models sonnet,flash,haiku,pro] \\
      [--threshold 0.6]

Outputs:
  tmp/compare_seed_validators.csv      one row per (model, photo)
  tmp/compare_seed_validators.summary  pass-rate + agreement table

Cost: ~$0.01/photo/model on average. 10 photos × 4 models ≈ $0.40.
"""
from __future__ import annotations
import argparse
import base64
import csv
import json
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

MANIFEST = REPO_ROOT / "Issue_Gen" / "photos" / "manifest.jsonl"
PHOTOS_ROOT = REPO_ROOT / "Issue_Gen" / "photos"
OUT_CSV = REPO_ROOT / "tmp" / "compare_seed_validators.csv"
OUT_SUMMARY = REPO_ROOT / "tmp" / "compare_seed_validators.summary"

MODEL_REGISTRY = {
    # Short alias -> (openrouter slug, label, ~ $/photo with vision @ ~1KB prompt)
    "sonnet":  ("anthropic/claude-sonnet-4.5",   "Sonnet 4.5",   0.0140),
    "haiku":   ("anthropic/claude-haiku-4.5",    "Haiku 4.5",    0.0030),
    "flash":   ("google/gemini-2.5-flash",        "Gemini Flash 2.5", 0.0050),
    "pro":     ("google/gemini-2.5-pro",          "Gemini Pro 2.5",   0.0150),
}


# Mirrors scripts/visual_seed_from_disk.py's VLM gate prompt so that
# the comparison is on the canonical seed-validation task — not
# something invented just for this script.
PROMPT = """You are reviewing a photograph from a Vietnamese / \
multi-country construction site to decide whether it should be added
to a safety-violation training dataset.

Look at the photo. Decide if it depicts a VISUAL safety violation
on a construction site. If YES, classify it into one of the
provided HSE-type and location slugs. If NO (paperwork, portraits,
office scenes, blank/unreadable, finishing-defect like "poor
workmanship" / "wrong paint colour"): return null for both slugs.

The issue_name and description from AECIS's record are AUXILIARY
context — trust the photo more than the metadata.

Output ONE JSON object, no prose, no markdown:

{
  "is_violation": true | false,
  "location": {"slug": "<slug or null>", "confidence": <0..1>},
  "hse_type": {"slug": "<slug or null>", "confidence": <0..1>},
  "reasoning": "<10-30 words>"
}

Use conservative confidence — < 0.6 means low certainty.
"""


def _load_taxonomy() -> dict:
    p = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    parents = {}
    if p.exists():
        parents = json.loads(p.read_text(encoding="utf-8")).get("parents", {})
    hse_types = [{"slug": k, "label_en": k.replace("_", " ").title()} for k in sorted(parents.keys())]
    locations = [
        {"slug": "Common_working_area", "label_en": "Common working area"},
        {"slug": "Excavation_or_pit",   "label_en": "Excavation / pit"},
        {"slug": "Height_work",         "label_en": "Work at height"},
        {"slug": "Storage_area",        "label_en": "Storage area"},
        {"slug": "Traffic_route",       "label_en": "Traffic route"},
        {"slug": "Electrical_zone",     "label_en": "Electrical zone"},
        {"slug": "Mechanical_zone",     "label_en": "Mechanical zone"},
        {"slug": "Confined_space",      "label_en": "Confined space"},
        {"slug": "Fire_exit",           "label_en": "Fire exit"},
    ]
    return {"hse_types": hse_types, "locations": locations}


def _encode(path: Path) -> tuple[str, str]:
    mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        path.suffix.lower().lstrip("."), "image/jpeg")
    return base64.standard_b64encode(path.read_bytes()).decode("ascii"), mt


def _call(model_slug: str, img: Path, issue_name: str, description: str, tax: dict) -> dict:
    """Single VLM call. Returns parsed JSON + the raw response for
    debugging. Never raises — wraps errors in the returned dict."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={"X-Title": "violation-bot-compare"},
    )
    hse_list = "\n".join(f"  - {h['slug']}: {h['label_en']}" for h in tax["hse_types"])
    loc_list = "\n".join(f"  - {l['slug']}: {l['label_en']}" for l in tax["locations"])
    b64, mt = _encode(img)
    user_content = [
        {"type": "text", "text": f"HSE_TYPES:\n{hse_list}\n\nLOCATIONS:\n{loc_list}"},
        {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}},
        {"type": "text",
         "text": f'AECIS issue name: "{issue_name}"\nAECIS description: "{description[:400]}"\n\n'
                 "Classify the photo above. Return JSON only."},
    ]
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_slug,
            max_tokens=300,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e <= s:
            return {"error": f"no JSON: {text[:80]}", "latency_s": latency}
        return {**json.loads(text[s : e + 1]), "latency_s": latency}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"[:200],
                "latency_s": time.perf_counter() - t0}


def _iter_photos(n: int):
    seen = set()
    out = []
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = r.get("filepath")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            img = PHOTOS_ROOT / fp
            if not img.exists():
                continue
            r["img_path"] = img
            out.append(r)
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--models", type=str, default="sonnet,haiku,flash,pro",
                    help="comma-separated aliases from MODEL_REGISTRY")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.stderr.write("ERROR: OPENROUTER_API_KEY not set\n"); return 2

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in chosen if m not in MODEL_REGISTRY]
    if bad:
        sys.stderr.write(f"unknown model aliases: {bad}\n"); return 2

    photos = _iter_photos(args.n)
    tax = _load_taxonomy()
    sys.stdout.write(f"photos: {len(photos)}, models: {chosen}\n\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    by_photo: dict[str, dict[str, dict]] = {}

    for p in photos:
        key = p["filepath"]
        by_photo[key] = {}
        sys.stdout.write(f"--- {key[-50:]}  issue={p.get('issue_id','?')} ---\n")
        for alias in chosen:
            slug, label, est_cost = MODEL_REGISTRY[alias]
            result = _call(slug, p["img_path"], p.get("issue_name", ""),
                           p.get("description", ""), tax)
            hse = ((result.get("hse_type") or {}).get("slug") or "")
            hse_conf = float((result.get("hse_type") or {}).get("confidence") or 0)
            is_viol = bool(result.get("is_violation"))
            passed = (is_viol and hse and hse_conf >= args.threshold)
            verdict = "passed" if passed else (
                "non_violation" if not is_viol else
                ("low_conf" if not result.get("error") else "error"))
            by_photo[key][alias] = {"verdict": verdict, "hse": hse, "conf": hse_conf}
            rows.append({
                "photo": key,
                "issue_id": p.get("issue_id", ""),
                "model_alias": alias,
                "model_label": label,
                "verdict": verdict,
                "hse_slug": hse,
                "hse_conf": round(hse_conf, 2),
                "latency_s": round(result.get("latency_s", 0), 1),
                "est_cost_usd": est_cost,
                "error": result.get("error", ""),
                "reasoning": (result.get("reasoning") or "")[:120],
            })
            sys.stdout.write(
                f"  {label:18s}  verdict={verdict:14s} "
                f"hse={hse[:25]:25s} conf={hse_conf:.2f} "
                f"({result.get('latency_s',0):.1f}s)\n"
            )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── summary: pass rate + agreement matrix ─────────────────
    summary = []
    summary.append(f"COMPARE-SEED-VALIDATORS — {len(photos)} photos, "
                   f"{len(chosen)} models")
    summary.append("")
    summary.append("PER-MODEL PASS RATE")
    summary.append("  model                pass  non-viol  low-conf  err  est-cost")
    summary.append("  -------------------------------------------------------------")
    for alias in chosen:
        slug, label, est = MODEL_REGISTRY[alias]
        c = {"passed": 0, "non_violation": 0, "low_conf": 0, "error": 0}
        for k in by_photo:
            v = by_photo[k][alias]["verdict"]
            c[v] = c.get(v, 0) + 1
        summary.append(
            f"  {label:18s}  {c['passed']:>3} "
            f"     {c['non_violation']:>3}        {c['low_conf']:>3}      {c['error']:>3}   "
            f"${est * len(photos):.2f}"
        )
    summary.append("")
    # Pairwise verdict agreement (treating "passed" vs "not passed" as binary)
    summary.append("PAIRWISE BINARY AGREEMENT  (model-A says pass iff model-B says pass)")
    summary.append("  " + " | ".join(f"{a:>10}" for a in [""] + chosen))
    for a in chosen:
        cells = [f"{a:>10}"]
        for b in chosen:
            if a == b:
                cells.append(f"{100.0:>9.1f}%")
            else:
                agree = sum(1 for k in by_photo
                            if (by_photo[k][a]["verdict"] == "passed") ==
                               (by_photo[k][b]["verdict"] == "passed"))
                cells.append(f"{100 * agree / len(by_photo):>9.1f}%")
        summary.append("  " + " | ".join(cells))
    summary.append("")
    summary.append("HSE-SLUG MATCH RATE  (passed-on-both AND same slug)")
    for a in chosen:
        for b in chosen:
            if a >= b: continue
            both_passed = [k for k in by_photo
                           if by_photo[k][a]["verdict"] == "passed"
                           and by_photo[k][b]["verdict"] == "passed"]
            if not both_passed:
                summary.append(f"  {a:8s} vs {b:8s}  (no overlap of passed photos)")
                continue
            same = sum(1 for k in both_passed
                       if by_photo[k][a]["hse"] == by_photo[k][b]["hse"])
            summary.append(
                f"  {a:8s} vs {b:8s}  {same}/{len(both_passed)} "
                f"= {100 * same / len(both_passed):.0f}% same slug"
            )

    OUT_SUMMARY.write_text("\n".join(summary), encoding="utf-8")
    sys.stdout.write("\n" + "\n".join(summary) + "\n")
    sys.stdout.write(f"\nCSV:     {OUT_CSV}\nSummary: {OUT_SUMMARY}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
