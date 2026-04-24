"""Extract the two prepared vocabularies (location + hse_type) from the
scraped violations dataset. Writes `taxonomy.json` + a markdown summary.

The AI classifier at inference time must pick from these exact vocabularies,
so this is the single source of truth for the model's label set.

Usage:
  python scripts/extract_taxonomy.py
  python scripts/extract_taxonomy.py --src <path>  --out taxonomy.json
  python scripts/extract_taxonomy.py --min-count 3   # prune very rare items

Outputs:
  taxonomy.json  — machine-readable: {locations: [...], hse_types: [...]}
  taxonomy.md    — human-readable summary with counts + bilingual labels
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("taxonomy")

DEFAULT_SRC = Path.home() / "Desktop" / "aecis-violations"
REPO_ROOT = Path(__file__).resolve().parents[1]


def collect(src: Path, *, dtag_only: bool = False) -> tuple[Counter, Counter, dict, dict, list[str]]:
    """Scan metadata.json files; return location + hse counts, bilingual maps,
    and the list of raw primary DTag strings seen (useful for debugging).

    If dtag_only=True, skip issues whose label_source != 'dtag'. Use this to
    build a clean structured vocabulary from SVN-style projects and exclude
    free-text title classes from older projects (SLPXA, etc.).
    """
    loc_counts: Counter = Counter()
    hse_counts: Counter = Counter()
    loc_vn: dict[str, str] = {}
    hse_vn: dict[str, str] = {}
    primary_dtags: list[str] = []

    for meta_path in src.glob("*/*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("broken metadata %s: %s", meta_path, e)
            continue
        if dtag_only:
            src_field = meta.get("label_source", "")
            # Legacy files scraped before label_source existed: treat those as
            # DTag-sourced (all original SVN data was DTag-based).
            if src_field and src_field != "dtag":
                continue
        hse_folder = meta_path.parent.parent.name  # the slugified hse_type
        loc_en = (meta.get("location_en") or "").strip()
        loc_vn_txt = (meta.get("location_vn") or "").strip()
        primary_raw = (meta.get("primary_dtag_raw") or "").strip()

        n_photos = len(meta.get("photos") or [])

        if loc_en:
            loc_counts[loc_en] += n_photos
            if loc_vn_txt and loc_en not in loc_vn:
                loc_vn[loc_en] = loc_vn_txt
        if hse_folder and hse_folder != "unlabeled":
            hse_counts[hse_folder] += n_photos
        # Parse hse_type EN/VN out of primary_dtag_raw for the bilingual map.
        # Pattern: "SVN | HSE | Location/VN | HSE type/VN"
        if primary_raw:
            primary_dtags.append(primary_raw)
            parts = [p.strip() for p in primary_raw.split("|")]
            if len(parts) >= 4:
                hse_raw = parts[3]
                # Split EN/VN on first "/ " (slash-space)
                idx = hse_raw.find("/ ")
                if idx >= 0:
                    en = hse_raw[:idx].strip()
                    vn = hse_raw[idx + 2:].strip()
                    if hse_folder and hse_folder not in hse_vn:
                        hse_vn[hse_folder] = vn
                    # Also keep the EN label keyed by folder
                    hse_vn[f"__en__{hse_folder}"] = en

    return loc_counts, hse_counts, loc_vn, hse_vn, primary_dtags


def build(
    src: Path,
    *,
    min_count: int,
    dtag_only: bool = False,
) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Source root not found: {src}")

    loc_counts, hse_counts, loc_vn, hse_vn, primary_dtags = collect(src, dtag_only=dtag_only)

    def locs_above(n: int) -> list[tuple[str, int]]:
        return sorted(((k, v) for k, v in loc_counts.items() if v >= n), key=lambda x: -x[1])

    def hse_above(n: int) -> list[tuple[str, int]]:
        return sorted(((k, v) for k, v in hse_counts.items() if v >= n), key=lambda x: -x[1])

    locations = [
        {
            "slug": name.replace(" ", "_"),
            "label_en": name,
            "label_vn": loc_vn.get(name, ""),
            "photo_count": count,
        }
        for name, count in locs_above(min_count)
    ]

    hse_types = [
        {
            "slug": slug,
            "label_en": hse_vn.get(f"__en__{slug}", slug.replace("_", " ")),
            "label_vn": hse_vn.get(slug, ""),
            "photo_count": count,
        }
        for slug, count in hse_above(min_count)
    ]

    total_locs = sum(c for _, c in loc_counts.items())
    total_hse = sum(c for _, c in hse_counts.items())
    total_issues = len(list(src.glob("*/*/metadata.json")))

    taxonomy = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_root": str(src),
        "min_count_filter": min_count,
        "totals": {
            "issues_scanned": total_issues,
            "location_total_photos": total_locs,
            "hse_type_total_photos": total_hse,
            "location_classes_kept": len(locations),
            "hse_type_classes_kept": len(hse_types),
            "location_classes_dropped": len(loc_counts) - len(locations),
            "hse_type_classes_dropped": len(hse_counts) - len(hse_types),
        },
        "locations": locations,
        "hse_types": hse_types,
    }
    return taxonomy


def render_md(tax: dict[str, Any]) -> str:
    t = tax["totals"]
    lines = [
        f"# Violation taxonomy (auto-generated)",
        "",
        f"Generated: {tax['generated_at']}  ·  Source: `{tax['source_root']}`  ·  "
        f"Min photo count: {tax['min_count_filter']}",
        "",
        f"- Issues scanned: **{t['issues_scanned']}**",
        f"- Location classes kept: **{t['location_classes_kept']}** "
        f"(dropped {t['location_classes_dropped']} below threshold)",
        f"- HSE-type classes kept: **{t['hse_type_classes_kept']}** "
        f"(dropped {t['hse_type_classes_dropped']} below threshold)",
        "",
        "## Locations",
        "",
        "| Photos | English | Vietnamese |",
        "|---:|---|---|",
    ]
    for loc in tax["locations"]:
        lines.append(f"| {loc['photo_count']} | {loc['label_en']} | {loc['label_vn']} |")
    lines += ["", "## HSE types", "", "| Photos | English | Vietnamese |", "|---:|---|---|"]
    for hse in tax["hse_types"]:
        lines.append(f"| {hse['photo_count']} | {hse['label_en']} | {hse['label_vn']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None,
                    help=f"Scraped dataset root. Default: {DEFAULT_SRC}")
    ap.add_argument("--out", type=str, default=None,
                    help=f"Taxonomy JSON output. Default: {REPO_ROOT}/taxonomy.json")
    ap.add_argument("--md", type=str, default=None,
                    help=f"Human-readable markdown. Default: {REPO_ROOT}/taxonomy.md")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Drop classes with fewer than N photos. Default: 1 (keep all).")
    ap.add_argument("--dtag-only", action="store_true",
                    help="Only include issues with label_source='dtag' (skip free-text "
                         "title-labeled data from older projects). Recommended for "
                         "building the prepared vocabulary used at inference time.")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve() if args.src else DEFAULT_SRC
    out = Path(args.out).expanduser().resolve() if args.out else REPO_ROOT / "taxonomy.json"
    md_out = Path(args.md).expanduser().resolve() if args.md else REPO_ROOT / "taxonomy.md"

    log.info("Source: %s", src)
    log.info("Out:    %s", out)
    log.info("Min:    %d", args.min_count)
    log.info("DTag only: %s", args.dtag_only)

    tax = build(src, min_count=args.min_count, dtag_only=args.dtag_only)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tax, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(render_md(tax), encoding="utf-8")

    t = tax["totals"]
    print()
    print(f"Locations kept:  {t['location_classes_kept']} "
          f"(dropped {t['location_classes_dropped']})")
    print(f"HSE types kept:  {t['hse_type_classes_kept']} "
          f"(dropped {t['hse_type_classes_dropped']})")
    print(f"JSON: {out}")
    print(f"MD:   {md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
