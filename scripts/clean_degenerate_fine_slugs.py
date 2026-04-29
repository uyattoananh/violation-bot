"""Remove degenerate fine sub-type slugs from data/fine_hse_types_by_parent.json.

Background: the AECIS consolidation script that built fine_hse_types_by_parent.json
slugified some labels poorly, leaving entries whose slug AND label are
single context-free words like "Damaged", "Unreinforced", "No". These
have no GT photos in the dataset (verified) and serve only to give the
two-stage classifier a meaningless candidate to pick from. Found via
manual audit when Stage 2 selected `Damaged` under Electrical_unsafe
twice on photos that obviously meant something else.

This script removes ONLY the slugs that:
  1. Are degenerate by heuristic (single word, slug == label, length < 12)
  2. AND have zero photos in the local dataset (~/Desktop/aecis-violations)

Slugs that have photos (No_roof, Unsorted, Not_clean) are LEFT ALONE —
they map to real GT data and need human input on what to rename them to,
or whether to keep them as-is.

Idempotent. Default dry-run; pass --apply to write. Backup to .bak.

Usage:
  python scripts/clean_degenerate_fine_slugs.py            # dry-run
  python scripts/clean_degenerate_fine_slugs.py --apply
  python scripts/clean_degenerate_fine_slugs.py --revert
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINE_PATH = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
DATASET_ROOT = Path.home() / "Desktop" / "aecis-violations"


# Hand-confirmed list of slugs to remove. The audit step below verifies
# each still has zero photos before applying — if a future re-scrape
# adds photos, we abort and report. Slugs that DO have photos are not
# in this list (No_roof, Unsorted, Not_clean) and stay untouched.
SLUGS_TO_REMOVE: dict[str, str] = {
    "Damaged":      "Electrical_unsafe",
    "Unreinforced": "Site_general_unsafe",
    "No":           "Workshop_area_unsafe",
    "Not_fixed":    "Formwork_unsafe",
}


def _has_photos(slug: str) -> int:
    folder = DATASET_ROOT / slug
    if not folder.exists():
        return 0
    return sum(
        1 for issue_dir in folder.iterdir() if issue_dir.is_dir()
        for _ in issue_dir.glob("*.jp*g")
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    bak = FINE_PATH.with_suffix(FINE_PATH.suffix + ".bak")
    if args.revert:
        if not bak.exists():
            print("No backup to revert from")
            return 1
        shutil.copy2(bak, FINE_PATH)
        print(f"restored {FINE_PATH.name} from {bak.name}")
        return 0

    data = json.loads(FINE_PATH.read_text(encoding="utf-8"))
    parents = data["parents"]

    print("=" * 60)
    print("Degenerate slug cleanup")
    print("=" * 60)

    removed = 0
    skipped = 0
    for slug, parent in SLUGS_TO_REMOVE.items():
        items = parents.get(parent, [])
        idx = next((i for i, it in enumerate(items) if it["slug"] == slug), None)
        if idx is None:
            print(f"  (skip) {slug!r} not in {parent} — already removed?")
            skipped += 1
            continue
        n_photos = _has_photos(slug)
        if n_photos > 0:
            print(f"  ⚠ ABORT: {slug!r} has {n_photos} photos in dataset — "
                  f"no longer safe to remove. Decide a rename target instead.")
            skipped += 1
            continue
        print(f"  remove: {parent}/{slug}  (0 photos in dataset)")
        del items[idx]
        removed += 1

    # Update stats
    if "stats" in data and isinstance(data["stats"], dict):
        total_fine = sum(len(v) for v in parents.values())
        old = data["stats"].get("total_fine_types")
        data["stats"]["total_fine_types"] = total_fine
        print(f"  total_fine_types: {old} -> {total_fine}")

    print()
    print(f"Removed: {removed}  Skipped: {skipped}")

    if not args.apply:
        print("DRY-RUN. Pass --apply to write.")
        return 0

    if not bak.exists():
        shutil.copy2(FINE_PATH, bak)
        print(f"  backed up to {bak.name}")
    FINE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FINE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
