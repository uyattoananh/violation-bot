"""Roll back the new (33-class consolidated) hse_type_slug values in
photo_embeddings, classifications, and corrections to the old 13-class
vocabulary. Use this if the new branch is failing eval and you want to
restore the prior known-good state.

Strategy: every NEW slug maps to exactly ONE OLD slug (defined in the
NEW_33_TO_OLD_13 table below). Identity mappings (Housekeeping_general
-> Housekeeping_general, Scaffolding_unsafe -> Scaffolding_unsafe, etc.)
are still listed for completeness — they're no-ops at runtime but make
the table self-documenting.

WARNING — when not to use:
  - If you've run auto_seed_from_disk.py with the new taxonomy and added
    rows that have slugs like Pressure_equipment_unsafe / Welding_unsafe /
    Confined_space_unsafe (slugs that don't exist in the old 13-class
    vocab), this revert WILL coerce them into the closest old equivalent.
    That's lossy. To prevent that loss you should either:
      (a) accept the lossy coercion and run this script normally, OR
      (b) DELETE the auto_seed-added rows first (any row added after the
          forward migration timestamp), then run revert.

Idempotent: running twice is safe.

Usage on the VPS or local Windows:
  python scripts/revert_pgvector_to_old_hse.py            # dry-run
  python scripts/revert_pgvector_to_old_hse.py --apply    # actually update
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


# Comprehensive table: every NEW slug -> closest OLD equivalent.
# Identity entries kept for clarity. Where multiple new slugs collapse
# into one old slug (e.g. Welding_unsafe + Hot_work_hazard + Fire_prevention_unsafe
# all -> Fire_hot_work_hazard), we accept the lossy collapse.
NEW_33_TO_OLD_13 = {
    # Identity (no rename in forward migration)
    "Housekeeping_general":       "Housekeeping_general",
    "Scaffolding_unsafe":         "Scaffolding_unsafe",
    "Edge_protection_missing":    "Edge_protection_missing",
    "Equipment_machinery_unsafe": "Equipment_machinery_unsafe",
    "Fall_protection_personal":   "Fall_protection_personal",
    "Lifting_unsafe":             "Lifting_unsafe",
    "Electrical_unsafe":          "Electrical_unsafe",
    "Chemicals_hazmat_unsafe":    "Chemicals_hazmat_unsafe",
    "PPE_missing":                "PPE_missing",
    "Ladder_unsafe":              "Ladder_unsafe",
    # Renamed in forward migration — invert
    "Site_access_unsafe":         "Access_walkway_unsafe",
    "Site_general_unsafe":        "Site_conditions_unsafe",
    "Hot_work_hazard":            "Fire_hot_work_hazard",
    # New parents that didn't exist in the old vocab — map to closest equivalent
    "Welding_unsafe":             "Fire_hot_work_hazard",
    "Fire_prevention_unsafe":     "Fire_hot_work_hazard",
    "Pressure_equipment_unsafe":  "Equipment_machinery_unsafe",
    "Truck_vehicle_unsafe":       "Equipment_machinery_unsafe",
    "Mass_piling_unsafe":         "Equipment_machinery_unsafe",
    "Concrete_work_unsafe":       "Equipment_machinery_unsafe",
    "Excavation_unsafe":          "Site_conditions_unsafe",
    "Confined_space_unsafe":      "Site_conditions_unsafe",
    "Smoking_area_unsafe":        "Site_conditions_unsafe",
    "First_aid_kit_unsafe":       "Site_conditions_unsafe",
    "Workshop_area_unsafe":       "Site_conditions_unsafe",
    "Site_lighting_unsafe":       "Site_conditions_unsafe",
    "Warning_signs_missing":      "Site_conditions_unsafe",
    "Parking_area_unsafe":        "Site_conditions_unsafe",
    "Drinking_water_unsafe":      "Housekeeping_general",
    "Common_area_unsafe":         "Housekeeping_general",
    "Materials_storage_unsafe":   "Housekeeping_general",
    "Garbage_waste_unsafe":       "Housekeeping_general",
    "Warehouse_unsafe":           "Housekeeping_general",
    "Formwork_unsafe":            "Equipment_machinery_unsafe",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually run the UPDATEs (default = dry-run)")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        print("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required", file=sys.stderr)
        return 2

    from supabase import create_client
    db = create_client(url, key)

    print(f"=== HSE REVERT — {'APPLY' if args.apply else 'DRY-RUN'} ===")
    print(f"  {len(NEW_33_TO_OLD_13)} new->old slug mappings in the revert table")
    print(f"  IDENTITIES are no-ops at runtime; only NON-IDENTITIES will UPDATE rows.")
    print()

    grand_total = 0
    for table in ("photo_embeddings", "classifications", "corrections"):
        print(f"--- table: {table} ---")
        per_slug = {}
        for new, old in NEW_33_TO_OLD_13.items():
            if new == old:
                continue   # identity — nothing to do
            try:
                res = (
                    db.table(table).select("*", count="exact")
                      .eq("hse_type_slug", new).limit(1).execute()
                )
                n = res.count or 0
            except Exception as e:  # noqa: BLE001
                print(f"  COUNT FAILED  {new}: {e}")
                continue
            per_slug[new] = (n, old)
            if n:
                print(f"  {n:5d}  {new:<32} -> {old}")
        total = sum(c[0] for c in per_slug.values())
        print(f"  TOTAL rows to revert: {total}")
        grand_total += total

        if args.apply and total:
            for new, (n, old) in per_slug.items():
                if n == 0:
                    continue
                try:
                    db.table(table).update({"hse_type_slug": old}).eq("hse_type_slug", new).execute()
                    print(f"  applied: {new} -> {old}  ({n} rows)")
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED applying {new} -> {old}: {e}")
        print()

    print(f"Grand total rows that would be reverted: {grand_total}")
    if not args.apply:
        print("Re-run with --apply to actually write the updates.")
        print()
        print("After --apply, also revert taxonomy.json + taxonomy_merges.json by either:")
        print("  - git checkout main -- taxonomy.json taxonomy_merges.json")
        print("  - or switch the whole repo back: git checkout main && bash scripts/restart-vps.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
