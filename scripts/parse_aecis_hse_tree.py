"""Parse the AECIS HSE tree text dump (data/aecis_hse_tree_raw.txt) into a
structured JSON taxonomy.

Input format: alternating lines of (label, owner_name). Owner names are a
known small set. Labels look like "EN/ VN" or "EN / VN" — split on the first
slash that's not in a parenthesised aside.

Some top-level entries (Safety training, TBM) are administrative and the
photographer can't capture them in a site photo — those are excluded so the
classifier doesn't get a class it can't see.

Output: data/aecis_hse_tree.json with the shape:
    {
      "version": "aecis-1.0",
      "source": "AECIS web app HSE tree",
      "locations": [
        {
          "slug": "...", "label_en": "...", "label_vn": "...",
          "hse_types": [{"slug": "...", "label_en": "...", "label_vn": "..."}]
        }
      ]
    }
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Owner names that should be filtered out — they appear after each label.
OWNERS = {"Phan Thanh Tung", "Nguyen Toan Tri", "Phan Hữu Ánh"}

# The exact EN-prefix of each top-level location, in order. Anything not in
# this set that appears between owner lines is treated as an HSE-type child
# of the most recent location.
LOCATION_HEADINGS_EN = [
    "Mass piling work",
    "Housekeeping",                            # "Housekeeping / Công tác vệ sinh" — note leading-space variant
    "Steel structures, precast concrete installation",
    "Lifting work",
    "Digging/Deep hole",
    "Electricity and Electrical Equipment",
    "Confined space",
    "Construction equipment (Gondola, Suspended scaffold, Hoist, Lifting equipment",
    "Trucks",
    "Electric welding - Gas welding",
    "Working at height",
    "Gate and safety walkway",
    "Fire Prevention",
    "Gas and chemicals storage",
    "Scaffolding and Platform",
    "Pressure equipment",
    "Hot work",
    "Warehouse Area",
    "Formwork",
    "Workshop area",
    "Smoke Area",
    "Floor opening",
    "PPEs",
    "Mechanical equipment for construction",
    "Garbage area - Waste",
    "Emergency case",
    "Temporary lightning pole",
    "Concrete work",
    "Material area",
    "Drinking water for worker",
    "Common area",
    "Signal board, warning tape",
    "Parking area",
    "Ladder",
    "First-aid kit",
    "Common working area",
]

# Excluded — the user said skip "items that cannot be photographed" like
# training and toolbox-meeting branches.
EXCLUDED_HEADINGS = [
    "Safety training",
    "TBM",                                     # appears as bare "TBM" and as "TBM/ Họp giao ban đầu ca"
]


def _slugify(s: str, maxlen: int = 80) -> str:
    """alpha/num/_/- only, collapse runs of separators, truncate."""
    s = s.strip()
    # Replace runs of non-alphanumeric with a single underscore.
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:maxlen] or "unknown"


def _split_bilingual(label: str) -> tuple[str, str]:
    """A label looks like 'EN text/ VN text' or 'EN text / VN text'.
    The boundary marker is a slash IMMEDIATELY followed by a space — the EN
    side may itself contain slashes (e.g. 'Digging/Deep hole/ Đào đất').
    We also skip slashes inside parens.

    Returns (en, vn). If no boundary found, returns (label, '')."""
    depth = 0
    for i, c in enumerate(label):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "/" and depth == 0:
            # Only treat this slash as the boundary if the next char is a space.
            if i + 1 < len(label) and label[i + 1] == " ":
                en = label[:i].strip()
                vn = label[i + 1 :].strip()
                return en, vn
    return label.strip(), ""


def _is_location_heading(en: str) -> bool:
    en = en.strip()
    for heading in LOCATION_HEADINGS_EN:
        if en == heading or en == heading + " ":          # tolerate trailing space
            return True
        # Some headings have ellipsis or close-paren that varies — accept prefix match
        # only for the long Construction-equipment one.
        if heading.startswith("Construction equipment") and en.startswith(heading):
            return True
    return False


def _is_excluded_heading(en: str) -> bool:
    en = en.strip()
    for heading in EXCLUDED_HEADINGS:
        if en == heading or en == heading + " " or en.startswith(heading + "/") or en.startswith(heading + " "):
            return True
    return False


def parse_tree(raw: str) -> dict:
    # Drop blank lines + owner lines + the root "HSE" line.
    lines: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s in OWNERS:
            continue
        if s == "HSE":
            continue
        lines.append(s)

    locations: list[dict] = []
    cur_location: dict | None = None
    skipping = False

    seen_loc_slugs: set[str] = set()
    seen_hse_in_loc: dict[str, set[str]] = {}

    for ln in lines:
        en, vn = _split_bilingual(ln)
        if _is_excluded_heading(en):
            cur_location = None
            skipping = True
            continue
        if _is_location_heading(en):
            skipping = False
            slug = _slugify(en)
            # Disambiguate duplicate slugs (Housekeeping vs Housekeeping_general etc.)
            base = slug
            i = 2
            while slug in seen_loc_slugs:
                slug = f"{base}_{i}"
                i += 1
            seen_loc_slugs.add(slug)

            cur_location = {
                "slug": slug,
                "label_en": en,
                "label_vn": vn,
                "hse_types": [],
            }
            seen_hse_in_loc[slug] = set()
            locations.append(cur_location)
            continue
        # Anything else is an HSE-type under the current location.
        if skipping or cur_location is None:
            continue
        slug = _slugify(en)
        if slug in seen_hse_in_loc[cur_location["slug"]]:
            # Same hse name appearing twice under one location (typo in source) — skip duplicate.
            continue
        seen_hse_in_loc[cur_location["slug"]].add(slug)
        cur_location["hse_types"].append({
            "slug": slug,
            "label_en": en,
            "label_vn": vn,
        })

    return {
        "version": "aecis-1.0",
        "source": "AECIS web app HSE tree (DOM dump, manually curated)",
        "excluded_branches": EXCLUDED_HEADINGS,
        "stats": {
            "location_count": len(locations),
            "hse_type_total": sum(len(loc["hse_types"]) for loc in locations),
            "hse_type_unique_en": len({h["label_en"] for loc in locations for h in loc["hse_types"]}),
        },
        "locations": locations,
    }


def main() -> int:
    in_path = REPO_ROOT / "data" / "aecis_hse_tree_raw.txt"
    out_path = REPO_ROOT / "data" / "aecis_hse_tree.json"

    if not in_path.exists():
        print(f"missing {in_path}", file=sys.stderr)
        return 1

    raw = in_path.read_text(encoding="utf-8")
    tree = parse_tree(raw)
    out_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  locations:        {tree['stats']['location_count']}")
    print(f"  hse_types total:  {tree['stats']['hse_type_total']}")
    print(f"  unique EN labels: {tree['stats']['hse_type_unique_en']}")
    print()
    print("locations breakdown:")
    for loc in tree["locations"]:
        print(f"  {len(loc['hse_types']):3d}  {loc['slug']:<48}  {loc['label_en']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
