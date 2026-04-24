"""Evaluate RAG-assisted classifier against ground-truth labels.

Samples N photos with DTag-sourced labels, runs each through the live
classifier (which now uses pgvector k-NN retrieval + Sonnet), and computes:

  - Top-1 accuracy for hse_type (against the consolidated cluster)
  - Top-1 accuracy for location (against the consolidated cluster)
  - Per-class precision / recall / F1
  - Mean confidence (agreement vs. disagreement)
  - Token + wall-clock cost
  - A confusion-pair table for the worst misclassifications

Ground-truth mapping uses taxonomy_merges.json — we compare the model's
consolidated output against the consolidated class of the photo's source DTag.

Leak guard: when retrieving k-NN neighbours, any neighbour whose sha256
matches the query photo itself is filtered out (we don't want the model
"finding" itself in the reference index).

Usage:
  python scripts/evaluate_rag.py --n 100
  python scripts/evaluate_rag.py --n 100 --no-rag   # compare against pure zero-shot
  python scripts/evaluate_rag.py --n 100 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.zero_shot import classify_image, load_taxonomy  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _source_to_consolidated_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (hse_map, loc_map) that maps raw source slugs to consolidated
    slugs. Merges taxonomy_merges.json with the LLM-mapping cache written by
    the auto-seed script so eval GT matches the vocabulary that got embedded.
    """
    merges = json.loads((REPO_ROOT / "taxonomy_merges.json").read_text(encoding="utf-8"))
    hse_map: dict[str, str] = {}
    for c in merges.get("hse_type_clusters", []):
        for src in c["absorbs"]:
            hse_map[src] = c["slug"]
    loc_map: dict[str, str] = {}
    for c in merges.get("location_clusters", []):
        for src in c["absorbs"]:
            loc_map[src] = c["slug"]

    # Merge in LLM mappings from auto-seed so ground-truth slugs the LLM
    # decided on are reflected at eval time.
    cache_path = REPO_ROOT / "scripts" / ".title_mapping_cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cache = {}
        for k, v in cache.items():
            if not k.startswith("slug::"):
                continue
            pseudo_title = k[len("slug::"):]
            # reconstruct the likely slug form (spaces -> underscores)
            src_slug = pseudo_title.replace(" ", "_")
            if v.get("hse"):
                hse_map.setdefault(src_slug, v["hse"])
    return hse_map, loc_map


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample(root: Path, n: int, seed: int,
            hse_map: dict[str, str], loc_map: dict[str, str]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for meta_path in root.glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if (meta.get("label_source") or "dtag") != "dtag":
            continue
        src_hse = meta_path.parent.parent.name
        src_loc_en = meta.get("location_en") or ""
        src_loc = src_loc_en.replace(" ", "_") if src_loc_en else ""
        gt_hse = hse_map.get(src_hse, src_hse)
        gt_loc = loc_map.get(src_loc, src_loc) if src_loc else ""
        for ph in (meta.get("photos") or []):
            fname = ph.get("file")
            if not fname:
                continue
            img = meta_path.parent / fname
            if not img.exists():
                continue
            pool.append({
                "image_path": img,
                "sha256": ph.get("sha256"),
                "gt_hse": gt_hse,
                "gt_loc": gt_loc,
                "issue_id": meta.get("issue_id"),
                "project_code": meta.get("project_code") or "SVN",
                "src_hse": src_hse,
                "src_loc": src_loc,
            })
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _temporarily_hide(db, sha: str) -> dict | None:
    """Move a row out of photo_embeddings by nulling its embedding, then restore.
    Prevents the query photo from appearing as its own neighbour.
    Returns the original row so we can restore it.
    """
    if not sha:
        return None
    try:
        existing = db.table("photo_embeddings").select("*").eq("sha256", sha).execute().data
        if not existing:
            return None
        # Null the embedding so the RPC filter `where embedding is not null` skips it
        db.table("photo_embeddings").update({"embedding": None}).eq("sha256", sha).execute()
        return existing[0]
    except Exception as e:  # noqa: BLE001
        log.warning("leak-guard hide failed for %s: %s", sha, e)
        return None


def _restore(db, row: dict | None) -> None:
    if not row or not row.get("embedding"):
        return
    try:
        db.table("photo_embeddings").update({"embedding": row["embedding"]}).eq("sha256", row["sha256"]).execute()
    except Exception as e:  # noqa: BLE001
        log.warning("leak-guard restore failed for %s: %s", row.get("sha256"), e)


def evaluate(sample: list[dict[str, Any]], use_rag: bool) -> dict[str, Any]:
    tax = load_taxonomy()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    db = None
    if use_rag and url and key:
        from supabase import create_client
        db = create_client(url, key)

    k = 5 if use_rag else 0
    results: list[dict[str, Any]] = []
    started = time.time()
    total_in_tok = 0
    total_out_tok = 0

    for i, s in enumerate(sample, 1):
        log.info("[%d/%d] %s  GT=(%s, %s)", i, len(sample),
                 s["image_path"].name, s["gt_hse"], s["gt_loc"])
        # Leak-guard: hide this photo's own embedding from retrieval
        original = _temporarily_hide(db, s["sha256"]) if db else None
        try:
            cls = classify_image(s["image_path"], taxonomy=tax, rag_neighbours=k)
        except Exception as e:  # noqa: BLE001
            log.exception("classify failed for %s", s["image_path"])
            results.append({**{k2: v for k2, v in s.items() if k2 != "image_path"},
                            "error": repr(e)})
            _restore(db, original)
            continue
        _restore(db, original)

        total_in_tok += cls.input_tokens
        total_out_tok += cls.output_tokens
        hse_alt_slugs = [a.slug for a in (cls.hse_type_alternatives or [])]
        loc_alt_slugs = [a.slug for a in (cls.location_alternatives or [])]
        hse_top3 = {cls.hse_type.slug, *hse_alt_slugs}
        loc_top3 = {cls.location.slug, *loc_alt_slugs}
        results.append({
            "image": str(s["image_path"].relative_to(DEFAULT_ROOT)).replace("\\", "/"),
            "issue_id": s["issue_id"],
            "project_code": s["project_code"],
            "gt_hse": s["gt_hse"],
            "gt_loc": s["gt_loc"],
            "src_hse": s["src_hse"],
            "src_loc": s["src_loc"],
            "pred_hse": cls.hse_type.slug,
            "pred_loc": cls.location.slug,
            "pred_hse_alternatives": hse_alt_slugs,
            "pred_loc_alternatives": loc_alt_slugs,
            "hse_conf": cls.hse_type.confidence,
            "loc_conf": cls.location.confidence,
            "hse_match": cls.hse_type.slug == s["gt_hse"],
            "loc_match": cls.location.slug == s["gt_loc"],
            "hse_top3_match": s["gt_hse"] in hse_top3,
            "loc_top3_match": s["gt_loc"] in loc_top3,
            "rationale": cls.rationale,
            "input_tokens": cls.input_tokens,
            "output_tokens": cls.output_tokens,
        })

    elapsed = time.time() - started
    ok = [r for r in results if "error" not in r]
    n_ok = len(ok)
    hse_correct = sum(1 for r in ok if r["hse_match"])
    loc_correct = sum(1 for r in ok if r["loc_match"])
    hse_top3_correct = sum(1 for r in ok if r.get("hse_top3_match"))
    loc_top3_correct = sum(1 for r in ok if r.get("loc_top3_match"))

    # Per-class precision / recall / F1 on hse
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for r in ok:
        gt, pred = r["gt_hse"], r["pred_hse"]
        if gt == pred:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1
    classes = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    per_class = {}
    for c in classes:
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[c] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    # Confusion pairs (top misses)
    miss_pairs: Counter = Counter()
    for r in ok:
        if not r["hse_match"]:
            miss_pairs[(r["gt_hse"], r["pred_hse"])] += 1

    mean_hse_conf_correct = (
        sum(r["hse_conf"] for r in ok if r["hse_match"]) / max(hse_correct, 1)
    )
    mean_hse_conf_wrong = (
        sum(r["hse_conf"] for r in ok if not r["hse_match"]) / max(n_ok - hse_correct, 1)
    )

    # Estimated cost (Haiku or Sonnet — rough; OpenRouter invoices are authoritative)
    model_id = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    if "haiku" in model_id:
        in_rate, out_rate = 1.0, 5.0   # $/M tokens
    else:
        in_rate, out_rate = 3.0, 15.0
    est_cost_usd = (total_in_tok / 1e6) * in_rate + (total_out_tok / 1e6) * out_rate

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_size": n_ok,
        "errors": len(results) - n_ok,
        "use_rag": use_rag,
        "model": model_id,
        "elapsed_seconds": round(elapsed, 1),
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "estimated_cost_usd": round(est_cost_usd, 4),
        "hse_accuracy": round(hse_correct / max(n_ok, 1), 3),
        "location_accuracy": round(loc_correct / max(n_ok, 1), 3),
        "hse_top3_accuracy": round(hse_top3_correct / max(n_ok, 1), 3),
        "location_top3_accuracy": round(loc_top3_correct / max(n_ok, 1), 3),
        "mean_conf_when_correct": round(mean_hse_conf_correct, 3),
        "mean_conf_when_wrong": round(mean_hse_conf_wrong, 3),
        "per_class": per_class,
        "top_confusions": [
            {"gt": gt, "pred": pred, "count": n}
            for (gt, pred), n in miss_pairs.most_common(15)
        ],
        "results": results,
    }
    return summary


def print_summary(s: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print(f"EVALUATION — RAG={'ON' if s['use_rag'] else 'OFF'} · model={s['model']}")
    print("=" * 70)
    print(f"Sample:                 {s['sample_size']} photos  (errors {s['errors']})")
    print(f"Elapsed:                {s['elapsed_seconds']}s")
    print(f"Tokens:                 in={s['total_input_tokens']:,}  out={s['total_output_tokens']:,}")
    print(f"Estimated OpenRouter:   ${s['estimated_cost_usd']}  (authoritative figure in dashboard)")
    print()
    print(f"HSE-type accuracy:      {s['hse_accuracy']*100:.1f}%  "
          f"(top-3: {s.get('hse_top3_accuracy', 0)*100:.1f}%)")
    print(f"Location accuracy:      {s['location_accuracy']*100:.1f}%  "
          f"(top-3: {s.get('location_top3_accuracy', 0)*100:.1f}%)")
    print(f"Mean confidence — correct: {s['mean_conf_when_correct']:.2f}")
    print(f"Mean confidence — wrong:   {s['mean_conf_when_wrong']:.2f}")
    print()
    print("Per-class F1 (top 10 by support):")
    rows = sorted(s["per_class"].items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"]))
    for cls, st in rows[:10]:
        support = st["tp"] + st["fn"]
        print(f"  {cls[:45]:<45}  support={support:>3}  "
              f"P={st['precision']:.2f}  R={st['recall']:.2f}  F1={st['f1']:.2f}")
    print()
    print("Top confusions (ground-truth -> predicted, count):")
    for c in s["top_confusions"]:
        print(f"  {c['count']:>3}  {c['gt'][:35]:<35} -> {c['pred']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rag", action="store_true",
                    help="Disable retrieval (compare against pure zero-shot)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    hse_map, loc_map = _source_to_consolidated_maps()
    sample = _sample(root, args.n, args.seed, hse_map, loc_map)
    log.info("Sampled %d photos for evaluation", len(sample))

    s = evaluate(sample, use_rag=not args.no_rag)
    print_summary(s)

    mode_tag = "nofag" if args.no_rag else "rag"
    out = Path(args.out).expanduser().resolve() if args.out else (
        REPO_ROOT / f"evaluation_{mode_tag}_{int(time.time())}.json"
    )
    out.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
