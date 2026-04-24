"""Measure zero-shot classifier agreement with AECIS-provided labels.

Walks a sample of photos with structured ground truth (DTag-labeled SVN data),
runs each through the current classifier, and reports:
  - Top-1 agreement on hse_type
  - Top-1 agreement on location
  - Mean confidence by axis
  - Per-photo details (ground truth vs. predicted) for the lowest-confidence
    / disagreement cases

Usage:
  python scripts/measure_agreement.py --n 20
  python scripts/measure_agreement.py --n 50 --seed 7 --min-class-size 10
  python scripts/measure_agreement.py --dataset-root <path> --out results.json

Only includes issues where label_source == 'dtag' (clean structured labels).
Samples stratified so rare classes get representation proportional to their
actual frequency.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot import classify_image, load_taxonomy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("agreement")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_DATASET = Path.home() / "Desktop" / "aecis-violations"


def _collect_candidates(root: Path, min_class_size: int) -> list[dict[str, Any]]:
    """Return one record per photo that has DTag-sourced ground truth."""
    # First pass: count per class
    per_class: dict[str, int] = {}
    for meta in root.glob("*/*/metadata.json"):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        src_field = d.get("label_source", "dtag")  # legacy files assumed DTag
        if src_field and src_field != "dtag":
            continue
        hse_folder = meta.parent.parent.name
        per_class[hse_folder] = per_class.get(hse_folder, 0) + 1

    ok_classes = {c for c, n in per_class.items() if n >= min_class_size}

    candidates: list[dict[str, Any]] = []
    for meta in root.glob("*/*/metadata.json"):
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        src_field = d.get("label_source", "dtag")
        if src_field and src_field != "dtag":
            continue
        hse_folder = meta.parent.parent.name
        if hse_folder not in ok_classes:
            continue
        for ph in (d.get("photos") or []):
            fname = ph.get("file")
            if not fname:
                continue
            img = meta.parent / fname
            if not img.exists():
                continue
            candidates.append({
                "image_path": img,
                "gt_hse_type_slug": hse_folder,
                "gt_location_en": d.get("location_en", ""),
                "issue_id": d.get("issue_id", ""),
            })
    return candidates


def _slug_for_location(loc_en: str, tax: dict[str, Any]) -> str:
    for l in tax["locations"]:
        if l["label_en"] == loc_en:
            return l["slug"]
    return loc_en.replace(" ", "_")


def run(dataset_root: Path, n: int, seed: int, min_class_size: int) -> dict[str, Any]:
    tax = load_taxonomy()
    candidates = _collect_candidates(dataset_root, min_class_size)
    log.info("Candidate pool: %d photos across %d classes",
             len(candidates),
             len({c["gt_hse_type_slug"] for c in candidates}))

    if not candidates:
        raise RuntimeError("No DTag-labeled candidates found.")

    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n, len(candidates)))
    log.info("Running %d classifications...", len(sample))

    results: list[dict[str, Any]] = []
    started = time.time()
    total_input_toks = 0
    total_output_toks = 0

    for i, c in enumerate(sample, 1):
        log.info("[%d/%d] %s", i, len(sample), c["image_path"].name)
        try:
            cls = classify_image(c["image_path"], taxonomy=tax)
        except Exception as e:  # noqa: BLE001
            log.exception("classify failed on %s", c["image_path"])
            results.append({
                "image_path": str(c["image_path"]),
                "gt_hse_type_slug": c["gt_hse_type_slug"],
                "error": repr(e),
            })
            continue
        gt_loc_slug = _slug_for_location(c["gt_location_en"], tax)
        results.append({
            "image_path": str(c["image_path"]),
            "issue_id": c["issue_id"],
            "gt_hse_type_slug": c["gt_hse_type_slug"],
            "gt_location_slug": gt_loc_slug,
            "gt_location_en": c["gt_location_en"],
            "pred_hse_type_slug": cls.hse_type.slug,
            "pred_location_slug": cls.location.slug,
            "hse_confidence": cls.hse_type.confidence,
            "loc_confidence": cls.location.confidence,
            "hse_match": cls.hse_type.slug == c["gt_hse_type_slug"],
            "loc_match": cls.location.slug == gt_loc_slug,
            "rationale": cls.rationale,
            "input_tokens": cls.input_tokens,
            "output_tokens": cls.output_tokens,
        })
        total_input_toks += cls.input_tokens
        total_output_toks += cls.output_tokens

    elapsed = time.time() - started

    # Compute stats
    ok_runs = [r for r in results if "error" not in r]
    n_ok = len(ok_runs)
    hse_correct = sum(1 for r in ok_runs if r["hse_match"])
    loc_correct = sum(1 for r in ok_runs if r["loc_match"])
    mean_hse_conf = sum(r["hse_confidence"] for r in ok_runs) / max(n_ok, 1)
    mean_loc_conf = sum(r["loc_confidence"] for r in ok_runs) / max(n_ok, 1)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": ok_runs[0].get("_model", "?") if ok_runs else "?",
        "sample_size": n_ok,
        "errors": len(results) - n_ok,
        "elapsed_seconds": round(elapsed, 1),
        "total_input_tokens": total_input_toks,
        "total_output_tokens": total_output_toks,
        "hse_agreement": round(hse_correct / max(n_ok, 1), 3),
        "location_agreement": round(loc_correct / max(n_ok, 1), 3),
        "mean_hse_confidence": round(mean_hse_conf, 3),
        "mean_loc_confidence": round(mean_loc_conf, 3),
        "results": results,
    }
    return summary


def print_summary(s: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print("AGREEMENT REPORT")
    print("=" * 60)
    print(f"Sample size:            {s['sample_size']}  (errors: {s['errors']})")
    print(f"Elapsed:                {s['elapsed_seconds']}s")
    print(f"Tokens (total):         in={s['total_input_tokens']}  out={s['total_output_tokens']}")
    print()
    print(f"HSE-type agreement:     {s['hse_agreement']*100:.1f}%  "
          f"(mean conf {s['mean_hse_confidence']:.2f})")
    print(f"Location agreement:     {s['location_agreement']*100:.1f}%  "
          f"(mean conf {s['mean_loc_confidence']:.2f})")
    print()
    # Disagreements with highest confidence (most worrying)
    disagree = [r for r in s["results"]
                if not r.get("hse_match", True) and "error" not in r]
    disagree.sort(key=lambda r: -r.get("hse_confidence", 0))
    if disagree:
        print("Top disagreements (hse_type), by predicted confidence:")
        for r in disagree[:10]:
            print(f"  conf {r['hse_confidence']:.2f}  "
                  f"GT  {r['gt_hse_type_slug'][:40]:<40}  "
                  f"PRED {r['pred_hse_type_slug'][:40]}")
        if len(disagree) > 10:
            print(f"  ... and {len(disagree) - 10} more")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--n", type=int, default=20, help="sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-class-size", type=int, default=5,
                    help="Only sample from classes with >=N photos")
    ap.add_argument("--out", type=str, default=None,
                    help="Path for the JSON results file")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else DEFAULT_DATASET
    if not root.exists():
        print(f"Dataset root not found: {root}", file=sys.stderr)
        return 1

    s = run(root, n=args.n, seed=args.seed, min_class_size=args.min_class_size)
    print_summary(s)

    out = Path(args.out).expanduser().resolve() if args.out else (
        REPO_ROOT / f"agreement_{int(time.time())}.json"
    )
    out.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
