"""Build data/fine_hse_types_by_parent.json — a lookup the webapp uses to
populate the optional 'AECIS fine sub-type' dropdown that appears beside
the coarse hse_type when the inspector clicks Edit.

Source: data/aecis_consolidation.json (the 33 parents with their absorbed
AECIS canonical slugs, EN/VN labels). We strip the legacy 13-class slugs
since those aren't AECIS-canonical and would just clutter the dropdown.

Output shape:
    {
      "Lifting_unsafe": [
        {"slug": "Cable_hooks_used_to_lift_objects_are_unsafe",
         "label_en": "Cable hooks used to lift objects are unsafe",
         "label_vn": "Cáp móc để nâng vật không đảm bảo an toàn",
         "primary_work_zone": "Lifting work"},
        ...
      ],
      ...
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cons_path = REPO_ROOT / "data" / "aecis_consolidation.json"
    if not cons_path.exists():
        print("missing aecis_consolidation.json", file=sys.stderr)
        return 1
    cons = json.loads(cons_path.read_text(encoding="utf-8"))

    by_parent: dict[str, list[dict]] = {}
    for parent in cons["parents"]:
        children = []
        for c in parent["absorbs"]:
            children.append({
                "slug": c["aecis_slug"],
                "label_en": c["label_en"],
                "label_vn": c.get("label_vn", ""),
                "primary_work_zone": c.get("primary_work_zone", ""),
            })
        # Sort alphabetically by EN label so the dropdown is browsable
        children.sort(key=lambda x: x["label_en"])
        by_parent[parent["slug"]] = children

    out_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    out_path.write_text(json.dumps({
        "version": "aecis-fine-1.0",
        "source": "data/aecis_consolidation.json",
        "parents": by_parent,
        "stats": {
            "parent_count": len(by_parent),
            "total_fine_types": sum(len(v) for v in by_parent.values()),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  {len(by_parent)} parents")
    print(f"  {sum(len(v) for v in by_parent.values())} total fine sub-types")
    print(f"  largest parent: ", end="")
    biggest = max(by_parent.items(), key=lambda kv: len(kv[1]))
    print(f"{biggest[0]} ({len(biggest[1])} fine types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
