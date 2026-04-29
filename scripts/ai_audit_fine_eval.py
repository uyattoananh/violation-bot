"""AI-judge audit for fine-grained classification eval.

The exact-slug-match metric in evaluate_rag.py is too narrow for the AECIS
fine taxonomy — many photos have multiple valid labels and the dataset
folder picks one specific phrasing as "the" GT. This script bypasses
that by asking a separate LLM to judge each prediction:

  "Given this photo's title (e.g. 'Wearing safety harness but no hooking')
   and the model's predicted fine label (e.g. 'No lifeline'), are these
   describing the same safety violation, different-but-both-valid, or is
   the prediction wrong?"

The judge sees:
  - The original AECIS issue_title (the human-written description from
    metadata.json — RICHER than the slug)
  - The model's predicted parent + fine label as English text
  - The image itself (so the judge can verify physically)

The judge does NOT see which is GT vs prediction — it's a semantic
similarity / validity check.

Output: a JSON file + a printed summary with breakdown:
  exact_match     : pred and GT are the same canonical slug
  semantic_match  : different slugs but judge says they describe the same violation
  different_valid : different but both are valid descriptions of the photo
  wrong           : pred doesn't apply to this photo

The "useful" rate (exact + semantic + different_valid) is the real
shippability metric.

Usage:
  python scripts/ai_audit_fine_eval.py evaluation_flash_finegrained.json
  python scripts/ai_audit_fine_eval.py <eval.json> --judge-model anthropic/claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

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
log = logging.getLogger("audit")

DEFAULT_DATASET_ROOT = Path.home() / "Desktop" / "aecis-violations"


_FINE_LOOKUP_CACHE: dict[str, dict] | None = None


def _fine_lookup() -> dict[str, dict]:
    """Map fine_slug -> {label_en, label_vn, parent}."""
    global _FINE_LOOKUP_CACHE
    if _FINE_LOOKUP_CACHE is not None:
        return _FINE_LOOKUP_CACHE
    p = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for parent, items in (data.get("parents") or {}).items():
        for it in items:
            out[it["slug"]] = {**it, "parent": parent}
    _FINE_LOOKUP_CACHE = out
    return out


def _hse_lookup() -> dict[str, str]:
    """Map parent_slug -> human label_en."""
    p = REPO_ROOT / "taxonomy.json"
    tax = json.loads(p.read_text(encoding="utf-8"))
    return {h["slug"]: h.get("label_en", h["slug"]) for h in tax.get("hse_types", [])}


_JUDGE_PROMPT = """You are a construction safety auditor reviewing an AI's
violation classification. Your job: judge whether the AI's label is a
valid description of the safety violation visible in the photo.

You'll see:
  1. The photo
  2. The GROUND TRUTH label (from a human inspector's original report)
  3. The AI's PREDICTED label

Both are short English sentences from the AECIS canonical safety taxonomy.

Possible verdicts:
  - SEMANTIC_MATCH: the AI's label and the human's label describe the
    SAME violation, even if worded differently. (e.g. "no lifeline"
    vs "no fall arrest cable" = same thing)
  - DIFFERENT_VALID: the AI picked a DIFFERENT violation than the
    human, but the AI's pick IS a real violation visible in the photo.
    Common when a photo shows multiple safety issues.
  - WRONG: the AI's label does not describe anything actually visible
    in the photo. The AI hallucinated or got confused.
  - UNCLEAR: the photo is too poor / mislabeled / ambiguous to decide.

Output ONE JSON object, no prose, no markdown fences:

{
  "verdict": "SEMANTIC_MATCH" | "DIFFERENT_VALID" | "WRONG" | "UNCLEAR",
  "reason": "<one short sentence>"
}
"""


def _encode_image(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    media = "image/jpeg"
    if path.suffix.lower() in (".png",):
        media = "image/png"
    elif path.suffix.lower() in (".webp",):
        media = "image/webp"
    return base64.b64encode(raw).decode(), media


def _call_judge(image_b64: str, media_type: str,
                gt_text: str, pred_text: str,
                model_id: str) -> dict:
    """One LLM call. Returns the parsed verdict dict."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    user_text = (
        f"GROUND TRUTH (human inspector's label):\n  \"{gt_text}\"\n\n"
        f"AI PREDICTED:\n  \"{pred_text}\"\n\n"
        "Look at the photo. What is your verdict?"
    )
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=400,
        messages=[
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
    )
    text = resp.choices[0].message.content or ""
    # Strip fences if present, find JSON object
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return {"verdict": "UNCLEAR", "reason": f"judge non-JSON: {text[:80]}"}
    try:
        return json.loads(text[s:e + 1])
    except Exception:  # noqa: BLE001
        return {"verdict": "UNCLEAR", "reason": "judge JSON-parse failed"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", type=str)
    ap.add_argument("--root", type=str, default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--judge-model", type=str,
                    default="anthropic/claude-sonnet-4.5",
                    help="Use a stronger model than the one being audited "
                         "for an unbiased read. Default: Sonnet.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap photos audited (0 = all)")
    ap.add_argument("--include-alternatives", action="store_true",
                    help="Also judge each top-K alternative (Stage 2 emits up to "
                         "2 alts). Cost: ~3x judge calls. Lets us answer 'is the "
                         "right label in the top-3?' as a useful-rate.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    in_path = Path(args.eval_json).resolve()
    out_path = Path(args.out).resolve() if args.out else in_path.with_suffix(".aijudge.json")
    root = Path(args.root).resolve()

    eval_data = json.loads(in_path.read_text(encoding="utf-8"))
    rows = eval_data.get("results", [])
    if args.limit:
        rows = rows[:args.limit]

    fine = _fine_lookup()
    hse = _hse_lookup()

    judgments: list[dict] = []
    started = time.time()
    for i, r in enumerate(rows, 1):
        # Skip rows that are eval errors (no image, no prediction). The eval
        # script records them with an "error" key and minimal other fields.
        if "error" in r or not r.get("image"):
            judgments.append({
                "image": r.get("image", "(unknown)"),
                "verdict": "ERROR",
                "reason": r.get("error") or "missing image field",
            })
            continue
        if not r.get("fine_emitted"):
            # Stage 2 abstained — nothing to audit.
            judgments.append({
                "image": r["image"],
                "verdict": "ABSTAINED",
                "reason": "Stage 2 did not commit a fine label",
            })
            continue
        # Resolve human-readable GT title via metadata.json. Falls back
        # to the slug if metadata is missing.
        img_path = root / r["image"]
        meta_path = img_path.parent / "metadata.json"
        gt_title = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                gt_title = (meta.get("issue_title_en") or "").strip()
            except Exception:  # noqa: BLE001
                pass
        if not gt_title:
            gt_slug = r.get("gt_fine") or ""
            gt_title = fine.get(gt_slug, {}).get("label_en", gt_slug or "(unknown)")

        # Resolve prediction text from the slug.
        pred_slug = r.get("pred_fine") or ""
        pred_label = fine.get(pred_slug, {}).get("label_en", pred_slug)
        pred_parent = hse.get(r.get("pred_hse") or "", r.get("pred_hse"))
        pred_text = f"{pred_label} (under category: {pred_parent})"

        if not img_path.exists():
            judgments.append({
                "image": r["image"], "verdict": "UNCLEAR",
                "reason": "image file missing",
            })
            continue

        try:
            b64, media = _encode_image(img_path)
            res = _call_judge(b64, media, gt_title, pred_text, args.judge_model)
        except Exception as e:  # noqa: BLE001
            log.warning("judge call failed for %s: %s", r["image"], e)
            res = {"verdict": "UNCLEAR", "reason": f"call failed: {e}"}

        log.info("[%3d/%d] %-22s gt='%s' pred='%s'",
                 i, len(rows), res.get("verdict", "?"),
                 gt_title[:40], pred_label[:40])
        j = {
            "image": r["image"],
            "gt_title": gt_title,
            "gt_fine_slug": r.get("gt_fine"),
            "pred_fine_slug": pred_slug,
            "pred_fine_label": pred_label,
            "pred_parent": pred_parent,
            "fine_conf": r.get("fine_conf"),
            "exact_match": bool(r.get("fine_match")),
            "verdict": res.get("verdict", "UNCLEAR"),
            "reason": res.get("reason", ""),
        }
        # Optionally judge each alternative against the same photo so we can
        # compute "any-of-top-K useful" rate. Each alt gets its own verdict.
        if args.include_alternatives:
            alt_slugs = (r.get("pred_fine_alternatives") or [])[:2]
            alt_results: list[dict] = []
            for alt_slug in alt_slugs:
                alt_label = fine.get(alt_slug, {}).get("label_en", alt_slug)
                alt_text = f"{alt_label} (under category: {pred_parent})"
                try:
                    alt_res = _call_judge(b64, media, gt_title, alt_text, args.judge_model)
                except Exception as e:  # noqa: BLE001
                    alt_res = {"verdict": "UNCLEAR", "reason": f"alt call failed: {e}"}
                alt_results.append({
                    "slug": alt_slug, "label": alt_label,
                    "verdict": alt_res.get("verdict", "UNCLEAR"),
                    "reason": alt_res.get("reason", ""),
                })
                log.info("        alt: %-18s '%s'",
                         alt_res.get("verdict", "?"), alt_label[:40])
            j["alternatives"] = alt_results
            # "any of top-K useful" = primary OR any alt is SEMANTIC_MATCH or DIFFERENT_VALID
            useful_set = {"SEMANTIC_MATCH", "DIFFERENT_VALID"}
            j["top3_any_useful"] = (
                j["verdict"] in useful_set
                or any(a["verdict"] in useful_set for a in alt_results)
            )
        judgments.append(j)

    elapsed = time.time() - started
    counts = Counter(j["verdict"] for j in judgments)
    n = len(judgments)
    n_emitted = sum(1 for j in judgments if j["verdict"] != "ABSTAINED")
    exact = sum(1 for j in judgments if j.get("exact_match"))
    useful = sum(1 for j in judgments
                 if j["verdict"] in ("SEMANTIC_MATCH", "DIFFERENT_VALID"))
    wrong = sum(1 for j in judgments if j["verdict"] == "WRONG")

    # Top-3 useful rate: did ANY of the top-K (primary + 2 alts) get a
    # SEMANTIC_MATCH or DIFFERENT_VALID verdict? Only meaningful when
    # --include-alternatives was set; otherwise equals top-1 useful rate.
    top3_judged = [j for j in judgments if "top3_any_useful" in j]
    n_top3_useful = sum(1 for j in top3_judged if j["top3_any_useful"])
    top3_rate = (n_top3_useful / len(top3_judged)) if top3_judged else None

    summary = {
        "input": str(in_path.name),
        "judge_model": args.judge_model,
        "n_total": n,
        "n_emitted": n_emitted,
        "n_abstained": counts.get("ABSTAINED", 0),
        "n_exact_match": exact,
        "n_semantic_match": counts.get("SEMANTIC_MATCH", 0),
        "n_different_valid": counts.get("DIFFERENT_VALID", 0),
        "n_wrong": counts.get("WRONG", 0),
        "n_unclear": counts.get("UNCLEAR", 0),
        "useful_rate_when_emitted": round(useful / max(n_emitted, 1), 3),
        "useful_rate_overall": round(useful / max(n, 1), 3),
        "wrong_rate_when_emitted": round(wrong / max(n_emitted, 1), 3),
        "top3_any_useful_rate": round(top3_rate, 3) if top3_rate is not None else None,
        "top3_n_judged": len(top3_judged),
        "elapsed_seconds": round(elapsed, 1),
        "judgments": judgments,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"AI-JUDGE AUDIT — {n} photos, judge={args.judge_model}")
    print("=" * 60)
    print(f"Stage 2 emitted:        {n_emitted}/{n}")
    print(f"  Abstained:            {counts.get('ABSTAINED', 0)}")
    print()
    print(f"Of the {n_emitted} that emitted:")
    print(f"  exact slug match:     {exact:>3}  ({exact/max(n_emitted,1)*100:.0f}%)")
    print(f"  semantic match:       {counts.get('SEMANTIC_MATCH',0):>3}  "
          f"({counts.get('SEMANTIC_MATCH',0)/max(n_emitted,1)*100:.0f}%)")
    print(f"  different but valid:  {counts.get('DIFFERENT_VALID',0):>3}  "
          f"({counts.get('DIFFERENT_VALID',0)/max(n_emitted,1)*100:.0f}%)")
    print(f"  WRONG:                {counts.get('WRONG',0):>3}  "
          f"({counts.get('WRONG',0)/max(n_emitted,1)*100:.0f}%)")
    print(f"  unclear:              {counts.get('UNCLEAR',0):>3}")
    print()
    print(f"USEFUL rate when emitted: {summary['useful_rate_when_emitted']*100:.1f}%")
    print(f"USEFUL rate overall:      {summary['useful_rate_overall']*100:.1f}%")
    print(f"WRONG rate when emitted:  {summary['wrong_rate_when_emitted']*100:.1f}%")
    if summary.get("top3_any_useful_rate") is not None:
        print()
        print(f"Top-3 ANY useful rate:    {summary['top3_any_useful_rate']*100:.1f}%  "
              f"(over {summary['top3_n_judged']} photos with alternatives)")
        print(f"  (i.e. share of photos where the primary OR an alternative")
        print(f"   would be a valid label — what the inspector picks from)")
    print()
    print(f"JSON: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
