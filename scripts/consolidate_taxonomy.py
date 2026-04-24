"""Apply taxonomy_merges.json to taxonomy_source.json ->writes taxonomy.json.

Source taxonomy = the raw 71 hse_types / 19 locations auto-extracted from
scraped AECIS DTag data.
Merged taxonomy = the consolidated vocabulary the AI model sees at inference
time (13 hse_types / 9 locations in the default merges).

Also emits taxonomy.md for human readability.

Usage:
  python scripts/consolidate_taxonomy.py
  python scripts/consolidate_taxonomy.py --strict   # fail if any source slug isn't covered
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _merge_axis(source_axis: list[dict], clusters: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply merge clusters to a single taxonomy axis.

    Returns (new_entries, unmapped_source_slugs).
    """
    by_source_slug = {e["slug"]: e for e in source_axis}
    consumed: set[str] = set()
    out: list[dict] = []

    for cluster in clusters:
        photo_total = 0
        component_labels_en: list[str] = []
        for src_slug in cluster["absorbs"]:
            e = by_source_slug.get(src_slug)
            if not e:
                continue
            consumed.add(src_slug)
            photo_total += int(e.get("photo_count", 0) or 0)
            if e.get("label_en"):
                component_labels_en.append(e["label_en"])
        out.append({
            "slug": cluster["slug"],
            "label_en": cluster["label_en"],
            "label_vn": cluster.get("label_vn", ""),
            "photo_count": photo_total,
            "absorbs": cluster["absorbs"],
            "component_labels_en": component_labels_en,
        })

    unmapped = [e["slug"] for e in source_axis if e["slug"] not in consumed]
    # Carry unmapped as solo clusters
    for slug in unmapped:
        src = by_source_slug[slug]
        out.append({
            "slug": slug,
            "label_en": src["label_en"],
            "label_vn": src.get("label_vn", ""),
            "photo_count": src.get("photo_count", 0),
            "absorbs": [slug],
            "component_labels_en": [src["label_en"]],
        })

    # Sort by photo count desc
    out.sort(key=lambda x: -x["photo_count"])
    return out, unmapped


def render_md(tax: dict) -> str:
    lines = [
        "# Violation taxonomy (consolidated)",
        "",
        f"Generated: {tax['generated_at']}  ·  "
        f"{len(tax['locations'])} locations · {len(tax['hse_types'])} hse_types",
        "",
        "## HSE types",
        "",
        "| Photos | Slug | English | Vietnamese | Merged from |",
        "|---:|---|---|---|---|",
    ]
    for h in tax["hse_types"]:
        absorbs = ", ".join(h.get("absorbs", [])[:4])
        if len(h.get("absorbs", [])) > 4:
            absorbs += f", +{len(h['absorbs']) - 4} more"
        lines.append(
            f"| {h['photo_count']} | `{h['slug']}` | {h['label_en']} | "
            f"{h.get('label_vn','')} | {absorbs} |"
        )
    lines += ["", "## Locations", "",
              "| Photos | Slug | English | Vietnamese | Merged from |",
              "|---:|---|---|---|---|"]
    for l in tax["locations"]:
        absorbs = ", ".join(l.get("absorbs", [])[:4])
        lines.append(
            f"| {l['photo_count']} | `{l['slug']}` | {l['label_en']} | "
            f"{l.get('label_vn','')} | {absorbs} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=str(REPO_ROOT / "taxonomy_source.json"))
    ap.add_argument("--merges", type=str, default=str(REPO_ROOT / "taxonomy_merges.json"))
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "taxonomy.json"))
    ap.add_argument("--md-out", type=str, default=str(REPO_ROOT / "taxonomy.md"))
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any source slug is unmapped (instead of carrying solo).")
    args = ap.parse_args()

    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    merges = json.loads(Path(args.merges).read_text(encoding="utf-8"))

    hse_clusters = merges.get("hse_type_clusters", [])
    loc_clusters = merges.get("location_clusters", [])

    new_hse, unmapped_hse = _merge_axis(src["hse_types"], hse_clusters)
    new_loc, unmapped_loc = _merge_axis(src["locations"], loc_clusters)

    if args.strict and (unmapped_hse or unmapped_loc):
        print(f"Unmapped hse_type slugs: {unmapped_hse}", file=sys.stderr)
        print(f"Unmapped location slugs: {unmapped_loc}", file=sys.stderr)
        return 1

    out_tax = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_root": src.get("source_root", ""),
        "source_taxonomy_file": args.source,
        "merges_file": args.merges,
        "hse_types": new_hse,
        "locations": new_loc,
        "stats": {
            "source_hse_type_count": len(src["hse_types"]),
            "source_location_count": len(src["locations"]),
            "consolidated_hse_type_count": len(new_hse),
            "consolidated_location_count": len(new_loc),
            "unmapped_hse_types": unmapped_hse,
            "unmapped_locations": unmapped_loc,
        },
    }
    Path(args.out).write_text(json.dumps(out_tax, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.md_out).write_text(render_md(out_tax), encoding="utf-8")

    print(f"Consolidated HSE types:  {len(src['hse_types'])} ->{len(new_hse)}")
    print(f"Consolidated locations:  {len(src['locations'])} ->{len(new_loc)}")
    if unmapped_hse:
        print(f"Unmapped (carried solo) hse: {unmapped_hse}")
    if unmapped_loc:
        print(f"Unmapped (carried solo) loc: {unmapped_loc}")
    print(f"JSON: {args.out}")
    print(f"MD:   {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
