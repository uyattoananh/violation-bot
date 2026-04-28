"""Generate the new taxonomy.json + taxonomy_merges.json from
data/aecis_consolidation.json.

Inputs:
  data/aecis_consolidation.json   — 33 parent classes, each with absorbs[]
                                     listing the AECIS-canonical slugs
  taxonomy.json (existing)         — kept for the location vocabulary AND
                                     to extract the OLD 13-class slug→parent
                                     migration so old labels resolve
  taxonomy_merges.json (existing)  — kept to preserve scraped-DTag absorbs

Outputs:
  taxonomy.json (REWRITTEN)        — new 33-class hse_types + 9 locations
  taxonomy_merges.json (REWRITTEN) — old 13 slugs + 294 AECIS slugs +
                                     legacy DTag clusters → 33 parents
  data/old_to_new_hse_map.json     — explicit migration map for the
                                     pgvector + classifications + corrections
                                     UPDATEs (run on the VPS after merge)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# Manual map: each OLD 13-class slug (from existing taxonomy.json) -> new
# consolidated parent. These are all 1:1 except Fire_hot_work_hazard which
# split into Fire_prevention_unsafe + Hot_work_hazard; we map the legacy
# rolled-up class to Hot_work_hazard since the spark/welding interpretation
# was the dominant pattern in seeded data.
OLD_13_TO_NEW_33 = {
    "Housekeeping_general":       "Housekeeping_general",
    "Scaffolding_unsafe":         "Scaffolding_unsafe",
    "Edge_protection_missing":    "Edge_protection_missing",
    "Equipment_machinery_unsafe": "Equipment_machinery_unsafe",
    "Access_walkway_unsafe":      "Site_access_unsafe",
    "Fall_protection_personal":   "Fall_protection_personal",
    "Lifting_unsafe":             "Lifting_unsafe",
    "Electrical_unsafe":          "Electrical_unsafe",
    "Site_conditions_unsafe":     "Site_general_unsafe",
    "Chemicals_hazmat_unsafe":    "Chemicals_hazmat_unsafe",
    "PPE_missing":                "PPE_missing",
    "Fire_hot_work_hazard":       "Hot_work_hazard",
    "Ladder_unsafe":              "Ladder_unsafe",
}


def main() -> int:
    cons_path = REPO_ROOT / "data" / "aecis_consolidation.json"
    tax_path  = REPO_ROOT / "taxonomy.json"
    merges_path = REPO_ROOT / "taxonomy_merges.json"

    if not cons_path.exists():
        print("missing aecis_consolidation.json — run consolidate_aecis.py", file=sys.stderr)
        return 1

    cons = json.loads(cons_path.read_text(encoding="utf-8"))
    old_tax = json.loads(tax_path.read_text(encoding="utf-8"))
    old_merges = json.loads(merges_path.read_text(encoding="utf-8"))

    # Locations stay as-is — the user said location is for internal classifier
    # only; not changing that vocabulary in this consolidation pass.
    new_locations = old_tax["locations"]

    # Build the 33 new hse_types. Each parent inherits its label_en/_vn from
    # the consolidation file; absorbs lists EVERY thing that maps to it:
    #   - the OLD 13-class slug if applicable
    #   - the 294 AECIS-canonical slugs that were absorbed
    #   - any legacy raw DTag slugs from the old taxonomy_merges.json that
    #     belonged to the old class (so existing scraped labels still resolve)
    new_hse_types = []
    new_merge_clusters = []
    for parent in cons["parents"]:
        slug = parent["slug"]
        absorbs: list[str] = []

        # OLD 13-class slug, if this new parent has one
        for old, new in OLD_13_TO_NEW_33.items():
            if new == slug:
                absorbs.append(old)

        # AECIS canonical slugs absorbed by this parent
        for child in parent["absorbs"]:
            absorbs.append(child["aecis_slug"])

        # Legacy raw DTag absorbs from old taxonomy_merges.json — find every
        # OLD-13 cluster that maps to this NEW-33 parent and inherit its
        # source slug list.
        for old_cluster in old_merges.get("hse_type_clusters", []):
            old_slug = old_cluster["slug"]
            new_parent = OLD_13_TO_NEW_33.get(old_slug)
            if new_parent == slug:
                absorbs.extend(old_cluster.get("absorbs", []))

        # Dedup while preserving order
        seen = set()
        absorbs_dedup = []
        for a in absorbs:
            if a not in seen:
                seen.add(a)
                absorbs_dedup.append(a)

        new_hse_types.append({
            "slug": slug,
            "label_en": parent["label_en"],
            "label_vn": parent["label_vn"],
            "photo_count": 0,  # populated by future re-aggregation, not critical
            "absorbs": absorbs_dedup,
            "component_labels_en": [c["label_en"] for c in parent["absorbs"]],
        })

        new_merge_clusters.append({
            "slug": slug,
            "label_en": parent["label_en"],
            "label_vn": parent["label_vn"],
            "absorbs": absorbs_dedup,
        })

    # Write new taxonomy.json
    new_tax = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "source_root": old_tax.get("source_root", ""),
        "source_taxonomy_file": old_tax.get("source_taxonomy_file", ""),
        "merges_file": str(merges_path),
        "consolidation_source": str(cons_path),
        "hse_types": new_hse_types,
        "locations": new_locations,
        "stats": {
            "consolidated_hse_type_count": len(new_hse_types),
            "consolidated_location_count": len(new_locations),
            "absorbed_aecis_canonical_count": cons["absorbed_aecis_count"],
            "old_13_class_migrated": len(OLD_13_TO_NEW_33),
        },
    }
    tax_path.write_text(json.dumps(new_tax, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write new taxonomy_merges.json
    new_merges = {
        "version": 2,
        "notes": (
            "Consolidation v2: maps the 294 photographable AECIS-canonical "
            "hse_types AND the legacy 13-class slugs into 33 parent classes. "
            "Source: data/aecis_consolidation.json. Edit that file (or the "
            "PARENTS rules in scripts/consolidate_aecis.py) and re-run "
            "scripts/apply_aecis_consolidation.py to regenerate."
        ),
        "hse_type_clusters": new_merge_clusters,
        "location_clusters": old_merges.get("location_clusters", []),
    }
    merges_path.write_text(json.dumps(new_merges, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write the explicit old→new map (for the live data migration)
    map_path = REPO_ROOT / "data" / "old_to_new_hse_map.json"
    map_path.write_text(json.dumps({
        "old_to_new_hse_type_slug": OLD_13_TO_NEW_33,
        "notes": (
            "Run this on the VPS after deploying the new taxonomy: it updates "
            "all photo_embeddings, classifications, and corrections rows so "
            "the existing 1108 labels keep working under the new vocab. See "
            "scripts/migrate_pgvector_to_new_hse.py."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== APPLIED ===")
    print(f"taxonomy.json:           {len(new_hse_types)} hse_types, {len(new_locations)} locations")
    print(f"taxonomy_merges.json:    {len(new_merge_clusters)} clusters, "
          f"{sum(len(c['absorbs']) for c in new_merge_clusters)} total absorbs")
    print(f"old_to_new_hse_map.json: {len(OLD_13_TO_NEW_33)} legacy slugs migrated")
    print(f"")
    print(f"Each new parent inherits:")
    for h in new_hse_types[:5]:
        print(f"  {h['slug']:<32} <- {len(h['absorbs'])} absorbs (incl. AECIS + legacy)")
    print(f"  ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
