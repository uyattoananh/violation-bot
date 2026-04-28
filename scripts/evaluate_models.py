"""Compare classification accuracy + cost across multiple models on the
SAME sample of photos.

Why: we only switch the production model if it BOTH lowers cost AND raises
accuracy. A single-model eval can't show the trade-off; this one runs the
same sample through every candidate and prints a side-by-side table.

The sample is drawn once with a fixed seed, then re-used for every model so
results are directly comparable (no different-photo, different-difficulty
confound).

Each model gets:
  - Top-1 hse accuracy (and top-3)
  - Top-1 location accuracy (and top-3)
  - Mean confidence on correct + on wrong
  - Total input + output tokens
  - Estimated cost in USD (per public list price; OpenRouter invoice is truth)
  - Wall-clock seconds
  - Per-class F1 of the worst 5 classes (so we can see if the new model
    fixes the F1=0 failure modes the prompt update was aimed at)

Usage:
  python scripts/evaluate_models.py --n 50
  python scripts/evaluate_models.py --n 100 --models anthropic/claude-sonnet-4.5,anthropic/claude-opus-4.5,google/gemini-2.5-pro
  python scripts/evaluate_models.py --n 50 --no-rag

NOTE: cost numbers below are MODEL_PRICING — public OpenRouter list prices
at the time of writing. Update if rates change. The OpenRouter dashboard
is always authoritative.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Reuse the existing eval infrastructure rather than fork it — guarantees
# the GT mapping + sampling logic stays identical to the single-model eval.
from scripts.evaluate_rag import (  # noqa: E402
    _source_to_consolidated_maps, _sample, _temporarily_hide, _restore,
    DEFAULT_ROOT,
)
from src.zero_shot import classify_image, load_taxonomy  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval-models")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# Public OpenRouter list prices (USD per 1M tokens). Keep this dict updated
# when rates move. If a model isn't here, cost falls back to N/A.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.5":   (3.0, 15.0),
    "anthropic/claude-opus-4.5":     (15.0, 75.0),
    "anthropic/claude-haiku-4.5":    (1.0, 5.0),
    "google/gemini-2.5-pro":         (1.25, 5.0),
    "google/gemini-2.5-flash":       (0.30, 2.50),
    "openai/gpt-4o":                 (2.5, 10.0),
    "openai/gpt-4o-mini":            (0.15, 0.60),
}

DEFAULT_MODELS = [
    "anthropic/claude-sonnet-4.5",   # current production
    "anthropic/claude-opus-4.5",     # smarter, more expensive
    "google/gemini-2.5-pro",         # cheaper alt; unknown vision quality on this domain
]


def _evaluate_one_model(model_id: str, sample: list[dict[str, Any]],
                        use_rag: bool, db) -> dict[str, Any]:
    """Run `sample` through `model_id` and return aggregate stats. Identical
    eval shape to evaluate_rag.evaluate, but parametrised by model_id and
    re-uses the SAME sample for every model so results are directly
    comparable."""
    tax = load_taxonomy()
    k = 5 if use_rag else 0
    started = time.time()
    total_in_tok = 0
    total_out_tok = 0
    results: list[dict[str, Any]] = []

    for i, s in enumerate(sample, 1):
        log.info("[%s] [%d/%d] %s GT=(%s, %s)",
                 model_id.split("/")[-1], i, len(sample),
                 s["image_path"].name, s["gt_hse"], s["gt_loc"])
        original = _temporarily_hide(db, s["sha256"]) if db else None
        try:
            cls = classify_image(s["image_path"], taxonomy=tax,
                                 rag_neighbours=k, model=model_id)
        except Exception as e:  # noqa: BLE001
            log.exception("classify failed for %s on %s", s["image_path"], model_id)
            results.append({"image": s["image_path"].name, "error": repr(e)})
            _restore(db, original)
            continue
        _restore(db, original)

        total_in_tok += cls.input_tokens
        total_out_tok += cls.output_tokens
        hse_alts = [a.slug for a in (cls.hse_type_alternatives or [])]
        loc_alts = [a.slug for a in (cls.location_alternatives or [])]
        results.append({
            "image": s["image_path"].name,
            "gt_hse": s["gt_hse"],
            "gt_loc": s["gt_loc"],
            "pred_hse": cls.hse_type.slug,
            "pred_loc": cls.location.slug,
            "hse_conf": cls.hse_type.confidence,
            "loc_conf": cls.location.confidence,
            "hse_match": cls.hse_type.slug == s["gt_hse"],
            "loc_match": cls.location.slug == s["gt_loc"],
            "hse_top3_match": s["gt_hse"] in {cls.hse_type.slug, *hse_alts},
            "loc_top3_match": s["gt_loc"] in {cls.location.slug, *loc_alts},
            "input_tokens": cls.input_tokens,
            "output_tokens": cls.output_tokens,
        })

    ok = [r for r in results if "error" not in r]
    n_ok = len(ok)
    hse_correct = sum(1 for r in ok if r["hse_match"])
    loc_correct = sum(1 for r in ok if r["loc_match"])
    hse_top3 = sum(1 for r in ok if r["hse_top3_match"])
    loc_top3 = sum(1 for r in ok if r["loc_top3_match"])

    # Per-class F1 — identifies whether the new model fixes the F1=0
    # failure modes (Garbage_waste, Site_access, Edge_protection) that
    # the prompt restructure was supposed to address.
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in ok:
        if r["hse_match"]:
            tp[r["gt_hse"]] += 1
        else:
            fp[r["pred_hse"]] += 1
            fn[r["gt_hse"]] += 1
    classes = sorted(set(list(tp) + list(fp) + list(fn)))
    per_class = {}
    for c in classes:
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec  = tp[c] / max(tp[c] + fn[c], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[c] = {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                        "precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "support": tp[c] + fn[c]}

    rate = MODEL_PRICING.get(model_id)
    if rate:
        in_rate, out_rate = rate
        cost = (total_in_tok / 1e6) * in_rate + (total_out_tok / 1e6) * out_rate
    else:
        cost = None

    return {
        "model": model_id,
        "elapsed_seconds": round(time.time() - started, 1),
        "sample_size": n_ok,
        "errors": len(results) - n_ok,
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "estimated_cost_usd": round(cost, 4) if cost is not None else None,
        "hse_accuracy": round(hse_correct / max(n_ok, 1), 3),
        "location_accuracy": round(loc_correct / max(n_ok, 1), 3),
        "hse_top3_accuracy": round(hse_top3 / max(n_ok, 1), 3),
        "location_top3_accuracy": round(loc_top3 / max(n_ok, 1), 3),
        "mean_conf_correct": round(
            sum(r["hse_conf"] for r in ok if r["hse_match"]) / max(hse_correct, 1), 3),
        "mean_conf_wrong": round(
            sum(r["hse_conf"] for r in ok if not r["hse_match"]) / max(n_ok - hse_correct, 1), 3),
        "per_class": per_class,
        "results": results,
    }


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 96)
    print("MODEL COMPARISON (same sample for every model)")
    print("=" * 96)
    # Header
    h = f"{'Model':<36} {'HSE':>7} {'HSE-T3':>7} {'LOC':>7} {'Tokens (in/out)':>22} {'Cost':>9} {'Time':>7}"
    print(h)
    print("-" * 96)
    base = rows[0]
    for r in rows:
        cost = f"${r['estimated_cost_usd']}" if r['estimated_cost_usd'] is not None else "n/a"
        delta_acc = ""
        delta_cost = ""
        if r is not base:
            delta_acc = f" ({(r['hse_accuracy'] - base['hse_accuracy'])*100:+.1f}pp)"
            if r['estimated_cost_usd'] is not None and base['estimated_cost_usd']:
                delta_cost = f" ({(r['estimated_cost_usd'] / base['estimated_cost_usd'] - 1)*100:+.0f}%)"
        print(f"{r['model']:<36} "
              f"{r['hse_accuracy']*100:>6.1f}% "
              f"{r['hse_top3_accuracy']*100:>6.1f}% "
              f"{r['location_accuracy']*100:>6.1f}% "
              f"{r['total_input_tokens']:>10,}/{r['total_output_tokens']:<10,} "
              f"{cost:>9} "
              f"{r['elapsed_seconds']:>6}s"
              + (delta_acc or "") + (delta_cost or ""))
    print()
    print("Decision rule: only switch from the baseline (first row) if a candidate")
    print("BOTH lowers cost AND raises HSE accuracy. Anything else is a regression.")
    print()
    print("Worst classes per model (lowest F1, support>=2):")
    for r in rows:
        bad = [(c, st) for c, st in r["per_class"].items() if st["support"] >= 2]
        bad.sort(key=lambda kv: kv[1]["f1"])
        print(f"  {r['model']}:")
        for c, st in bad[:5]:
            print(f"    {c[:45]:<45}  F1={st['f1']:.2f}  P={st['precision']:.2f}  "
                  f"R={st['recall']:.2f}  support={st['support']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="Comma-separated model IDs (OpenRouter format)")
    ap.add_argument("--no-rag", action="store_true",
                    help="Disable retrieval — pure-vision comparison")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models specified", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    hse_map, loc_map = _source_to_consolidated_maps()
    tax = load_taxonomy()
    sample = _sample(root, args.n, args.seed, hse_map, loc_map, tax)
    log.info("Sampled %d photos — running them through %d models",
             len(sample), len(models))

    db = None
    if not args.no_rag:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            from supabase import create_client
            db = create_client(url, key)

    rows: list[dict[str, Any]] = []
    for m in models:
        log.info("=" * 60)
        log.info("Evaluating: %s", m)
        log.info("=" * 60)
        rows.append(_evaluate_one_model(m, sample, use_rag=not args.no_rag, db=db))

    _print_comparison(rows)

    out = Path(args.out).expanduser().resolve() if args.out else (
        REPO_ROOT / f"evaluation_models_{int(time.time())}.json"
    )
    out.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_size": len(sample),
        "use_rag": not args.no_rag,
        "models": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
