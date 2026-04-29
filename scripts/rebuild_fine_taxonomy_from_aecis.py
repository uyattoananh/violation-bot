"""Rebuild data/fine_hse_types_by_parent.json from the canonical AECIS source.

The previous fine taxonomy (294 items) was generated from photos seen
during auto-seeding. Items the seeder hadn't seen were dropped — so
Stage 2's vocab was incomplete. Symptom: a "two workers on a ladder"
photo couldn't get the canonical AECIS slug
`Only_one_person_is_allowed_to_work_on_the_ladder` because that slug
wasn't in the candidate list.

This rebuild uses the AUTHORITATIVE AECIS source (data/aecis_hse_tree.json,
460 items across 36 source-locations) and maps each item to one of the
29 consolidated parent classes via a hand-curated LOCATION_TO_PARENT
table.

Output: writes data/fine_hse_types_by_parent.json with the full set,
preserving the file's existing schema:
  {
    "version": "...",
    "source": "...",
    "parents": {
      "<parent_slug>": [
        {"slug": "...", "label_en": "...", "label_vn": "...",
         "primary_work_zone": "<source_location_label>"},
        ...
      ]
    },
    "stats": {"parent_count": N, "total_fine_types": M}
  }

Idempotent. Default dry-run; --apply writes. Backup to .bak.

Usage:
  python scripts/rebuild_fine_taxonomy_from_aecis.py
  python scripts/rebuild_fine_taxonomy_from_aecis.py --apply
  python scripts/rebuild_fine_taxonomy_from_aecis.py --revert
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "aecis_hse_tree.json"
TARGET_PATH = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
TAXONOMY_PATH = REPO_ROOT / "taxonomy.json"


# Hand-curated mapping: each AECIS source location -> the consolidated
# parent class it belongs to. This is the "human knowledge" the
# automatic merge missed. Reviewed against the user-pasted AECIS
# canonical list and the 29 parents in taxonomy.json.
#
# Where ambiguity exists (a source location's items could go in 2
# parents), we pick the parent whose name is most semantically aligned
# with the location's WORK CONTEXT.
LOCATION_TO_PARENT: dict[str, str] = {
    # Direct name matches (clean 1:1)
    "Mass_piling_work":                                  "Mass_piling_unsafe",
    "Housekeeping":                                      "Housekeeping_general",
    "Confined_space":                                    "Confined_space_unsafe",
    "Trucks":                                            "Truck_vehicle_unsafe",
    "Working_at_height":                                 "Fall_protection_personal",
    "Gate_and_safety_walkway":                           "Site_access_unsafe",
    "Fire_Prevention":                                   "Fire_prevention_unsafe",
    "Gas_and_chemicals_storage":                         "Chemicals_hazmat_unsafe",
    "Scaffolding_and_Platform":                          "Scaffolding_unsafe",
    "Pressure_equipment":                                "Pressure_equipment_unsafe",
    "Hot_work":                                          "Hot_work_hazard",
    "Warehouse_Area":                                    "Warehouse_unsafe",
    "Formwork":                                          "Formwork_unsafe",
    "Workshop_area":                                     "Workshop_area_unsafe",
    "Floor_opening":                                     "Edge_protection_missing",
    "PPEs":                                              "PPE_missing",
    "Mechanical_equipment_for_construction":             "Equipment_machinery_unsafe",
    "Concrete_work":                                     "Concrete_work_unsafe",
    "Material_area":                                     "Materials_storage_unsafe",
    "Drinking_water_for_worker":                         "Drinking_water_unsafe",
    "Common_area":                                       "Common_area_unsafe",
    "Signal_board_warning_tape":                         "Warning_signs_missing",
    "Parking_area":                                      "Parking_area_unsafe",
    "Ladder":                                            "Ladder_unsafe",
    "First_aid_kit":                                     "First_aid_kit_unsafe",

    # Consolidations (multiple sources -> one parent)
    "Steel_structures_precast_concrete_installation":    "Lifting_unsafe",   # all about lifting steel/precast
    "Lifting_work":                                      "Lifting_unsafe",
    "Construction_equipment_Gondola_Suspended_scaffold_Hoist_Lifting_equipment":
                                                         "Lifting_unsafe",
    "Digging_Deep_hole":                                 "Excavation_unsafe",
    "Electricity_and_Electrical_Equipment":              "Electrical_unsafe",
    "Electric_welding_Gas_welding":                      "Hot_work_hazard",  # collapsed Welding into Hot_work
    "Smoke_Area":                                        "Housekeeping_general",  # collapsed Smoking_area
    "Garbage_area_Waste":                                "Housekeeping_general",  # collapsed Garbage_waste

    # Catch-alls (admin / generic categories)
    "Common_working_area":                               "Site_general_unsafe",
    "Emergency_case":                                    "Site_general_unsafe",
    "Temporary_lightning_pole":                          "Site_general_unsafe",
}


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backed up {p.name} -> {bak.name}")


def _revert(p: Path) -> int:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        print(f"no backup at {bak.name}")
        return 1
    shutil.copy2(bak, p)
    print(f"restored {p.name} from {bak.name}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        return _revert(TARGET_PATH)

    src = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    valid_parents = {h["slug"] for h in
                     json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["hse_types"]}

    # Verify mapping covers all source locations + maps to valid parents
    missing_mapping: list[str] = []
    invalid_parent: list[tuple[str, str]] = []
    for loc in src["locations"]:
        loc_slug = loc["slug"]
        if loc_slug not in LOCATION_TO_PARENT:
            missing_mapping.append(loc_slug)
        else:
            tgt = LOCATION_TO_PARENT[loc_slug]
            if tgt not in valid_parents:
                invalid_parent.append((loc_slug, tgt))
    if missing_mapping:
        print("ERROR: source locations with no parent mapping:")
        for s in missing_mapping:
            print(f"  - {s}")
        return 2
    if invalid_parent:
        print("ERROR: parent slugs not in current taxonomy.json:")
        for s, t in invalid_parent:
            print(f"  - {s} -> {t}")
        return 2

    # Build the new fine map. De-dup by slug (a slug appearing under
    # multiple source-locations gets the FIRST parent assignment;
    # alternatives are recorded for inspection but not as duplicates).
    by_parent: dict[str, list[dict]] = defaultdict(list)
    seen_slugs: dict[str, str] = {}   # slug -> parent (first-assigned)
    duplicates: list[tuple[str, str, str]] = []   # (slug, first_parent, attempted_parent)
    for loc in src["locations"]:
        parent = LOCATION_TO_PARENT[loc["slug"]]
        for ft in loc.get("hse_types", []):
            slug = ft.get("slug")
            if not slug:
                continue
            if slug in seen_slugs:
                if seen_slugs[slug] != parent:
                    duplicates.append((slug, seen_slugs[slug], parent))
                continue
            seen_slugs[slug] = parent
            by_parent[parent].append({
                "slug": slug,
                "label_en": ft.get("label_en", slug),
                "label_vn": ft.get("label_vn", ""),
                "primary_work_zone": loc.get("label_en", loc["slug"]),
            })

    # Stats
    n_total = sum(len(v) for v in by_parent.values())
    print(f"Source locations:   {len(src['locations'])}")
    print(f"Source fine items:  {sum(len(l.get('hse_types', [])) for l in src['locations'])}")
    print(f"Output parents:     {len(by_parent)}")
    print(f"Output fine items:  {n_total}  (after de-dup)")
    if duplicates:
        print(f"Cross-parent dup slugs: {len(duplicates)}")
        for slug, first, attempted in duplicates[:10]:
            print(f"  {slug}: kept under {first}, also seen as {attempted}")
        if len(duplicates) > 10:
            print(f"  ... ({len(duplicates) - 10} more)")
    print()
    print("Per-parent fine counts (new vs old):")
    old = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    old_parents = old.get("parents", {})
    parents_sorted = sorted(set(list(by_parent.keys()) + list(old_parents.keys())))
    for p in parents_sorted:
        new_n = len(by_parent.get(p, []))
        old_n = len(old_parents.get(p, []))
        delta = new_n - old_n
        marker = "  " if delta == 0 else (f" +{delta}" if delta > 0 else f" {delta}")
        print(f"  {p:<35} {old_n:>4} -> {new_n:>4} {marker}")

    # Diagnostic: show the items added under Ladder_unsafe (the user's case)
    if "Ladder_unsafe" in by_parent:
        old_ladder = {it["slug"] for it in old_parents.get("Ladder_unsafe", [])}
        new_ladder = by_parent["Ladder_unsafe"]
        added = [it for it in new_ladder if it["slug"] not in old_ladder]
        if added:
            print(f"\nNew Ladder_unsafe items added by this rebuild ({len(added)}):")
            for it in added:
                print(f"  - {it['slug']}")

    # Sanity: every parent in taxonomy.json should still have at least
    # one fine item (otherwise the inspector edit-form dropdown is empty).
    empty_parents = [p for p in valid_parents if not by_parent.get(p)]
    if empty_parents:
        print(f"\nWARNING: {len(empty_parents)} parent(s) have ZERO fine items:")
        for p in empty_parents:
            print(f"  - {p}")
        print("These are valid (some parents are pure catch-alls) but flagged for review.")

    out = {
        "version": "aecis-fine-2.0",
        "source": "data/aecis_hse_tree.json (full canonical AECIS taxonomy)",
        "rebuild_method": "scripts/rebuild_fine_taxonomy_from_aecis.py",
        "parents": dict(by_parent),
        "stats": {
            "parent_count": len(by_parent),
            "total_fine_types": n_total,
        },
    }

    if not args.apply:
        print()
        print("DRY-RUN. Pass --apply to write.")
        return 0

    _backup(TARGET_PATH)
    TARGET_PATH.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {TARGET_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
