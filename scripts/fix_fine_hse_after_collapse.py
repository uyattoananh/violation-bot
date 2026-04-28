"""Migrate fine_hse_types_by_parent.json after the parent collapse.

When apply_taxonomy_dedup.py collapsed 4 parent classes:
  Garbage_waste_unsafe   -> Housekeeping_general
  Smoking_area_unsafe    -> Housekeeping_general
  Site_lighting_unsafe   -> Site_general_unsafe
  Welding_unsafe         -> Hot_work_hazard

it left fine_hse_types_by_parent.json out of date — those parent keys
still exist there with their sub-type lists, which means:
  - The webapp inspector form would never reach those sub-types
    (the parent dropdown no longer lists them)
  - Inspectors lose granularity (e.g., "Garbage_outside_designated_areas"
    used to be a sub-type of Garbage_waste_unsafe; should now appear
    under Housekeeping_general)

This script:
  1. Reads data/fine_hse_types_by_parent.json
  2. Moves sub-types from each collapsed parent into the target parent
  3. Deduplicates by sub-type slug (in case both parents had the same sub-type)
  4. Removes the 4 collapsed parent entries
  5. Updates the parent_count stat
  6. Writes back, with .bak

Usage:
  python scripts/fix_fine_hse_after_collapse.py            # dry-run
  python scripts/fix_fine_hse_after_collapse.py --apply
  python scripts/fix_fine_hse_after_collapse.py --revert
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINE_PATH = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"

# Same collapse map as apply_taxonomy_dedup.py — keep in sync.
COLLAPSE_INTO: dict[str, str] = {
    "Garbage_waste_unsafe":  "Housekeeping_general",
    "Smoking_area_unsafe":   "Housekeeping_general",
    "Site_lighting_unsafe":  "Site_general_unsafe",
    "Welding_unsafe":        "Hot_work_hazard",
}


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backed up {p.name} -> {bak.name}")


def _revert() -> int:
    bak = FINE_PATH.with_suffix(FINE_PATH.suffix + ".bak")
    if not bak.exists():
        print(f"  no backup at {bak.name}, nothing to revert")
        return 1
    shutil.copy2(bak, FINE_PATH)
    print(f"  restored from {bak.name}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        return _revert()

    data = json.loads(FINE_PATH.read_text(encoding="utf-8"))
    # File structure: {"version": "...", "parents": {...}, "stats": {...}}
    # The parents dict is keyed by parent-class slug -> list of fine sub-types.
    parents: dict[str, list[dict]] = data["parents"]

    log: list[str] = []
    for src, dst in COLLAPSE_INTO.items():
        if src not in parents:
            log.append(f"  (skip) {src} not present in fine map")
            continue
        if dst not in parents:
            log.append(f"  WARNING: target {dst} not present — skipping {src}")
            continue
        existing_slugs = {f["slug"] for f in parents[dst]}
        moved = 0
        for ft in parents[src]:
            if ft["slug"] not in existing_slugs:
                parents[dst].append(ft)
                moved += 1
            else:
                # already a sub-type of dst, drop the dup
                pass
        log.append(f"  moved {moved}/{len(parents[src])} sub-types: "
                   f"{src} -> {dst}")
        del parents[src]

    # Update stats if present
    if "stats" in data and isinstance(data["stats"], dict):
        old_pc = data["stats"].get("parent_count")
        data["stats"]["parent_count"] = len(parents)
        total_fine = sum(len(v) for v in parents.values())
        data["stats"]["total_fine_types"] = total_fine
        log.append(f"  stats updated: parent_count {old_pc} -> {len(parents)}, "
                   f"total_fine_types -> {total_fine}")

    print("=" * 80)
    print("Fine sub-type migration plan")
    print("=" * 80)
    for line in log:
        print(line)
    print()

    if not args.apply:
        print("DRY-RUN: pass --apply to write")
        return 0

    _backup(FINE_PATH)
    FINE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FINE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
