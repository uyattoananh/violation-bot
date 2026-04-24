"""Classify a sample of scraped photos and render a side-by-side HTML page.

Produces a static review page showing each photo with:
  - Ground-truth label (AECIS folder / DTag)
  - AI-predicted location + hse_type + confidence
  - Rationale
  - Match indicator (exact-match vs. sibling vs. disagree)

The HTML can be opened straight in Chrome to eyeball classifier quality
without setting up the full webapp.

Usage:
  python scripts/classify_and_render_html.py --n 30
  python scripts/classify_and_render_html.py --n 100 --model anthropic/claude-haiku-4.5
  python scripts/classify_and_render_html.py --include-slpxa        # include SLPXA title-labeled data
  python scripts/classify_and_render_html.py --out predictions.html
"""
from __future__ import annotations

import argparse
import html
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
log = logging.getLogger("render")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _sample_photos(root: Path, n: int, seed: int, include_slpxa: bool) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for meta_path in root.glob("*/*/metadata.json"):
        try:
            d = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        src_field = d.get("label_source", "dtag")
        if not include_slpxa and src_field and src_field != "dtag":
            continue
        hse_folder = meta_path.parent.parent.name
        for ph in (d.get("photos") or []):
            fname = ph.get("file")
            if not fname:
                continue
            img = meta_path.parent / fname
            if not img.exists():
                continue
            pool.append({
                "image_path": img,
                "relpath": str(img.relative_to(root)).replace("\\", "/"),
                "gt_hse_type_slug": hse_folder,
                "gt_location_en": d.get("location_en", ""),
                "issue_id": d.get("issue_id", ""),
                "project_code": d.get("project_code", ""),
                "label_source": src_field or "dtag",
                "issue_title_en": d.get("issue_title_en", ""),
                "issue_title_vn": d.get("issue_title_vn", ""),
            })
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _render(predictions: list[dict[str, Any]], root: Path, model_name: str) -> str:
    esc = html.escape
    n_exact = sum(1 for p in predictions if p.get("hse_match"))
    n_loc_exact = sum(1 for p in predictions if p.get("loc_match"))

    cards = []
    for p in predictions:
        img = esc(p["relpath"])
        border_class = "border-emerald-500" if p.get("hse_match") else "border-amber-500" if p.get("pred_hse_type_slug") else "border-rose-500"
        conf_hse = f"{(p.get('hse_confidence') or 0)*100:.0f}%"
        conf_loc = f"{(p.get('loc_confidence') or 0)*100:.0f}%"
        match_tag = ""
        if p.get("hse_match"):
            match_tag = '<span class="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 text-xs">exact</span>'
        elif "error" in p:
            match_tag = '<span class="px-2 py-0.5 rounded bg-rose-100 text-rose-800 text-xs">error</span>'
        else:
            match_tag = '<span class="px-2 py-0.5 rounded bg-amber-100 text-amber-800 text-xs">disagree</span>'

        cards.append(f"""
<article class="bg-white rounded-lg border-2 {border_class} overflow-hidden">
  <a href="{img}" target="_blank">
    <img src="{img}" loading="lazy" class="w-full h-64 object-cover bg-slate-100">
  </a>
  <div class="p-3 text-sm space-y-1">
    <div class="flex items-center justify-between">
      <div class="text-xs text-slate-500">{esc(p['project_code'])} · issue {esc(p['issue_id'])} · {esc(p['label_source'])}</div>
      {match_tag}
    </div>
    <div><b>GT hse:</b> {esc(p['gt_hse_type_slug'])}</div>
    <div><b>GT loc:</b> {esc(p['gt_location_en']) or '<i class="text-slate-400">none</i>'}</div>
    <hr class="my-1">
    <div><b>AI hse:</b> {esc(p.get('pred_hse_type_slug') or 'ERROR')} <span class="text-slate-500">({conf_hse})</span></div>
    <div><b>AI loc:</b> {esc(p.get('pred_location_slug') or '?')} <span class="text-slate-500">({conf_loc})</span></div>
    <div class="italic text-slate-600 text-xs">{esc(p.get('rationale', '') or p.get('error', ''))}</div>
  </div>
</article>
""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AI classifier — spot-check</title>
<script src="https://cdn.tailwindcss.com"></script>
</head><body class="bg-slate-50">
<header class="bg-[#1f4e79] text-white px-6 py-3">
  <div class="max-w-6xl mx-auto">
    <h1 class="text-lg font-semibold">AI classifier spot-check</h1>
    <div class="text-sm opacity-80">
      {len(predictions)} photos · model <code>{esc(model_name)}</code> ·
      top-1 hse {n_exact}/{len(predictions)} ({n_exact/max(len(predictions),1)*100:.0f}%) ·
      top-1 loc {n_loc_exact}/{len(predictions)} ({n_loc_exact/max(len(predictions),1)*100:.0f}%) ·
      generated {time.strftime("%Y-%m-%d %H:%M")}
    </div>
  </div>
</header>
<main class="max-w-6xl mx-auto p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
{''.join(cards)}
</main>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--include-slpxa", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="HTML output path. Default: <root>/_spot_check.html")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    tax = load_taxonomy()
    loc_lookup = {l["slug"]: l for l in tax["locations"]}

    samples = _sample_photos(root, args.n, args.seed, args.include_slpxa)
    log.info("Classifying %d photos...", len(samples))

    predictions = []
    for i, s in enumerate(samples, 1):
        log.info("[%d/%d] %s", i, len(samples), s["relpath"])
        try:
            cls = classify_image(s["image_path"], taxonomy=tax, model=args.model)
        except Exception as e:  # noqa: BLE001
            log.exception("classify failed")
            predictions.append({**s, "error": repr(e)})
            continue
        # Resolve GT location slug for match check
        gt_loc_slug = s["gt_location_en"].replace(" ", "_") if s["gt_location_en"] else ""
        for l in tax["locations"]:
            if l["label_en"] == s["gt_location_en"]:
                gt_loc_slug = l["slug"]
                break
        predictions.append({
            **s,
            "pred_hse_type_slug": cls.hse_type.slug,
            "pred_location_slug": cls.location.slug,
            "hse_confidence": cls.hse_type.confidence,
            "loc_confidence": cls.location.confidence,
            "hse_match": cls.hse_type.slug == s["gt_hse_type_slug"],
            "loc_match": cls.location.slug == gt_loc_slug,
            "rationale": cls.rationale,
        })

    model_name = args.model or predictions[0].get("model", "?") if predictions else "?"
    # Use the real model name from the first successful classification
    for p in predictions:
        if "error" not in p and p.get("rationale"):
            # model field is stored in Classification.model but we didn't copy it above; use env or arg
            break

    html_out = _render(predictions, root, args.model or "default")
    # Output must be inside root for relative image paths to work
    out_path = Path(args.out).expanduser().resolve() if args.out else root / "_spot_check.html"
    out_path.write_text(html_out, encoding="utf-8")
    log.info("Spot-check page: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
