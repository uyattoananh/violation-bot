"""Apply the dedup + collapse plan to taxonomy_merges.json AND taxonomy.json.

Plan (per the audit + greenlit recommendation):

  Phase 1 — 24 mechanical "smaller cluster wins" duplicate resolutions
  Phase 2 — Manhole tiebreaker -> Edge_protection_missing
  Phase 3 — Collapse 4 redundant sub-classes:
              Garbage_waste_unsafe   -> Housekeeping_general
              Smoking_area_unsafe    -> Housekeeping_general
              Site_lighting_unsafe   -> Site_general_unsafe
              Welding_unsafe         -> Hot_work_hazard

After: 33 -> 29 hse_types. The 4 collapsed classes are not load-bearing
distinctions on the safety hierarchy of controls; they're sub-granularities
of existing main-tree branches.

Both files get a .bak backup before any change. --revert restores from .bak.
Default is --dry-run; pass --apply to actually write.

Usage:
  python scripts/apply_taxonomy_dedup.py            # dry-run (default)
  python scripts/apply_taxonomy_dedup.py --apply
  python scripts/apply_taxonomy_dedup.py --revert
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGES_PATH = REPO_ROOT / "taxonomy_merges.json"
TAX_PATH = REPO_ROOT / "taxonomy.json"


# ----- Phase 1+2: per-source-slug resolution table -----
# For each duplicated source slug, the cluster it should belong to.
# These come from the audit's auto-resolution (smaller cluster wins) plus
# the human pick on the manhole tiebreaker.
DUPLICATE_RESOLUTIONS: dict[str, str] = {
    "CO_CQ_record_inspection_or_maintenance_records_not_maintained":
        "Equipment_machinery_unsafe",
    "Do_not_pump_water_regularly_leaving_stagnant_water_causes_unsanitary_conditions":
        "Excavation_unsafe",
    "Garbage_outside_designated_areas":
        "Garbage_waste_unsafe",   # gets collapsed into Housekeeping in Phase 3
    "Gas_cylinder_compressed_air_not_fixed_on_shelf_trolley":
        "Welding_unsafe",         # gets collapsed into Hot_work in Phase 3
    "Lack_of_guarding_on_moving_parts":
        "Equipment_machinery_unsafe",
    "Lack_or_missing_of_lighting_for_walkway_or_working_areas":
        "Site_lighting_unsafe",   # gets collapsed into Site_general in Phase 3
    "Materials_are_not_arranged_neatly":
        "Materials_storage_unsafe",
    "Materials_are_risk_of_falling_from_height":
        "Site_general_unsafe",
    "Materials_falling":
        "Truck_vehicle_unsafe",
    "Missing_inspection_stamp_or_certificates":
        "Pressure_equipment_unsafe",
    "No_access_unsafe_access":
        "Site_access_unsafe",
    "No_fire_extinguisher":
        "Welding_unsafe",         # gets collapsed -> Hot_work
    "No_lack_of_installation_area_signs":
        "Housekeeping_general",
    "No_roof":
        "Site_general_unsafe",
    "No_signs_to_define_the_material_storage_area_such_as_warning_tape_signal_board":
        "Materials_storage_unsafe",
    "Not_covering_the_manhole":
        "Edge_protection_missing",   # PHASE 2 human pick (engineering control)
    "Not_or_lacking_Signal_board_warning_tape":
        "Warning_signs_missing",
    "Not_transportation_on_time":
        "Garbage_waste_unsafe",   # gets collapsed -> Housekeeping
    "Smoking_or_cigarette_butts_at_the_workplace":
        "Smoking_area_unsafe",    # gets collapsed -> Housekeeping
    "The_material_and_equipment_trans_lifting_area_must_not_be_isolated_without_warni":
        "Site_general_unsafe",
    "Unsafe_standing_position_weak_ground_no_padding_uneven_surface":
        "Site_general_unsafe",
    "Unsafe_walkway_Materials_falling_from_above_obstructive_objects_risk_of_falling_":
        "Site_access_unsafe",
    "Wearing_safety_harness_but_no_hooking":
        "Fall_protection_personal",
    "Working_at_heights_without_safety_harness":
        "Fall_protection_personal",
    "Working_below_without_protection":
        "Fall_protection_personal",
}

# ----- Phase 3: cluster -> parent collapse -----
COLLAPSE_INTO: dict[str, str] = {
    "Garbage_waste_unsafe":  "Housekeeping_general",
    "Smoking_area_unsafe":   "Housekeeping_general",
    "Site_lighting_unsafe":  "Site_general_unsafe",
    "Welding_unsafe":        "Hot_work_hazard",
}


def _backup(p: Path) -> Path:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backed up: {p.name} -> {bak.name}")
    else:
        print(f"  backup exists, leaving as-is: {bak.name}")
    return bak


def _revert() -> int:
    for p in (MERGES_PATH, TAX_PATH):
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            print(f"  no backup found for {p.name}, skipping")
            continue
        shutil.copy2(bak, p)
        print(f"  restored: {bak.name} -> {p.name}")
    return 0


def _apply_dedup_to_merges(merges: dict) -> tuple[dict, list[str]]:
    """Apply Phase 1+2 (dedup) and Phase 3 (collapse) to taxonomy_merges.

    Returns (new_merges, change_log).
    """
    log: list[str] = []
    clusters = merges.get("hse_type_clusters", [])

    # ---- Phase 1+2: keep each source slug only in the cluster that
    # the resolution table picks; remove from all others ----
    for c in clusters:
        new_absorbs = []
        for s in c["absorbs"]:
            assigned = DUPLICATE_RESOLUTIONS.get(s)
            if assigned is None:
                # Not a duplicate — keep wherever it currently sits.
                new_absorbs.append(s)
            elif assigned == c["slug"]:
                new_absorbs.append(s)
                # Will only log once per slug below, after both passes.
            else:
                log.append(f"  removed {s!r} from cluster {c['slug']!r} "
                           f"(assigned to {assigned!r})")
        c["absorbs"] = new_absorbs

    # ---- Phase 3: collapse the 4 redundant clusters into their parents
    # by transferring their absorbed sources, then deleting them ----
    parents_by_slug = {c["slug"]: c for c in clusters}
    new_clusters: list[dict] = []
    for c in clusters:
        target = COLLAPSE_INTO.get(c["slug"])
        if target is None:
            new_clusters.append(c)
            continue
        if target not in parents_by_slug:
            log.append(f"  WARNING: collapse target {target!r} not found "
                       f"for cluster {c['slug']!r} — skipping")
            new_clusters.append(c)
            continue
        # Move sources, also add the dead cluster's own slug as a source
        # of the parent (so legacy data with that slug auto-maps to the parent)
        parent = parents_by_slug[target]
        for s in c["absorbs"]:
            if s not in parent["absorbs"]:
                parent["absorbs"].append(s)
        if c["slug"] not in parent["absorbs"]:
            parent["absorbs"].append(c["slug"])
        log.append(f"  collapsed cluster {c['slug']!r} "
                   f"({len(c['absorbs'])} sources) -> {target!r}")
    merges["hse_type_clusters"] = new_clusters

    # Final dedup of absorbs[] within each cluster (in case a slug was
    # both a cluster-name AND already a source in the parent)
    for c in new_clusters:
        seen = set()
        unique = []
        for s in c["absorbs"]:
            if s not in seen:
                unique.append(s)
                seen.add(s)
        c["absorbs"] = unique

    return merges, log


def _apply_collapse_to_taxonomy(tax: dict) -> tuple[dict, list[str]]:
    """Remove the 4 collapsed classes from taxonomy.json's hse_types list."""
    log: list[str] = []
    keep = [h for h in tax.get("hse_types", []) if h["slug"] not in COLLAPSE_INTO]
    removed = [h["slug"] for h in tax.get("hse_types", []) if h["slug"] in COLLAPSE_INTO]
    for slug in removed:
        log.append(f"  removed {slug!r} from taxonomy.hse_types "
                   f"(collapsed into {COLLAPSE_INTO[slug]!r})")
    tax["hse_types"] = keep
    if "stats" in tax and "hse_types" in tax.get("stats", {}):
        tax["stats"]["hse_types"] = len(keep)
    return tax, log


def _diff_summary(old: dict, new: dict, kind: str) -> None:
    """Print a brief diff of cluster counts before/after for clarity."""
    if kind == "merges":
        old_n = len(old.get("hse_type_clusters", []))
        new_n = len(new.get("hse_type_clusters", []))
        print(f"  hse_type_clusters: {old_n} -> {new_n}  (-{old_n - new_n})")
    elif kind == "tax":
        old_n = len(old.get("hse_types", []))
        new_n = len(new.get("hse_types", []))
        print(f"  hse_types: {old_n} -> {new_n}  (-{old_n - new_n})")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default is dry-run)")
    ap.add_argument("--revert", action="store_true",
                    help="Restore both files from their .bak siblings")
    args = ap.parse_args()

    if args.revert:
        print("Reverting from backups...")
        return _revert()

    # Load both files
    merges = json.loads(MERGES_PATH.read_text(encoding="utf-8"))
    tax = json.loads(TAX_PATH.read_text(encoding="utf-8"))
    merges_orig = json.loads(json.dumps(merges))   # deep copy for diff
    tax_orig = json.loads(json.dumps(tax))

    print("=" * 80)
    print("Applying taxonomy dedup + collapse")
    print("=" * 80)
    print()

    # Apply changes in memory
    merges_new, merges_log = _apply_dedup_to_merges(merges)
    tax_new, tax_log = _apply_collapse_to_taxonomy(tax)

    print(f"taxonomy_merges.json — {len(merges_log)} changes")
    for line in merges_log[:25]:
        print(line)
    if len(merges_log) > 25:
        print(f"  ... ({len(merges_log) - 25} more)")
    print()
    _diff_summary(merges_orig, merges_new, "merges")
    print()

    print(f"taxonomy.json — {len(tax_log)} changes")
    for line in tax_log:
        print(line)
    print()
    _diff_summary(tax_orig, tax_new, "tax")
    print()

    if not args.apply:
        print("DRY-RUN: pass --apply to write the changes")
        print("         (--revert will restore from .bak after an apply)")
        return 0

    print("Writing changes...")
    _backup(MERGES_PATH)
    _backup(TAX_PATH)
    MERGES_PATH.write_text(
        json.dumps(merges_new, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    TAX_PATH.write_text(
        json.dumps(tax_new, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Done.")
    print()
    print("Next: run `python scripts/evaluate_rag.py --n 100 --seed 7` to")
    print("measure F1 recovery on the previously-zero classes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
