"""Consolidate the 294 photographable AECIS hse_types into ~30 coarse classes.

Rationale: 294 classes against 1,108 training photos = 3.8 photos/class
average — the long tail will be unpredictable. Grouping similar violations
under coarse parents brings avg/class up to ~30+ which matches what the
classifier can actually learn from.

Mapping rules are keyword-based and applied in priority order. The first
rule that matches wins. Each AECIS slug ends up in exactly one parent.

Outputs:
  data/aecis_consolidation.json  — full mapping (parent + absorbs[]) for review
  taxonomy.json                  — REPLACES the existing one with the new parents
  taxonomy_merges.json           — REPLACES the existing one (old 13-class +
                                     new AECIS canonical → consolidated)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----- consolidation rules: (parent_slug, label_en, label_vn, predicate) -----
# `predicate(en_lower, parent_zone_lower)` returns True if the AECIS entry
# belongs to this parent. Ordered by priority — first match wins.

PARENTS: list[tuple[str, str, str, callable]] = [

    # ---------- PPE (cross-cutting, highest priority) ----------
    ("PPE_missing",
     "Missing or inadequate personal protective equipment",
     "Thiếu hoặc không đầy đủ trang bị bảo hộ cá nhân",
     lambda en, z: (
         "ppe" in en or "ppes" in en or
         "safety helmet" in en or "no welding mask" in en or
         "welding gloves" in en or "hearing protection" in en or
         "safety glasses" in en or "safety shoes" in en or "safety harness" in en or
         "apron" in en or "metal without gloves" in en or
         "no wearing safety vest" in en or "without hard hat" in en
         or z == "ppes"
     )),

    # ---------- Fall protection (worker visibly at risk) ----------
    ("Fall_protection_personal",
     "Working at heights without personal fall protection",
     "Làm việc trên cao không có dây an toàn cá nhân",
     lambda en, z: (
         "without safety harness" in en or "no hooking" in en or "not hooking" in en or
         "no lifeline" in en or "lifeline damaged" in en or
         "lifeline has not been inspected" in en or "lifeline must" in en or
         "need to have a separate lifeline" in en or
         "no 3-square safety net" in en or "safety net" in en and "not been inspected" in en or
         "no safety net" in en or "safety net or damaged" in en
     )),

    # ---------- Edge / opening / floor hole protection ----------
    ("Edge_protection_missing",
     "Missing handrails, edge or opening protection",
     "Thiếu lan can bảo vệ mép, lỗ mở",
     lambda en, z: (
         "no safety handrail" in en or "no handrail" in en or
         "lack of hard handrails" in en or "no guardrail" in en or
         "openings not properly covered" in en or "floor opening" in en or
         z == "floor_opening" or
         "elevator door" in en or "elevator shaft" in en or
         "edge warning" in en or "no signs, hard handrail" in en or
         "standing on handrails" in en
     )),

    # ---------- Scaffolding ----------
    ("Scaffolding_unsafe",
     "Scaffolding unsafe or missing components",
     "Giàn giáo không an toàn hoặc thiếu chi tiết",
     lambda en, z: (
         "scaffold" in en or "i-beam" in en or "scaffolder" in en or
         z == "scaffolding_and_platform"
     )),

    # ---------- Ladder ----------
    ("Ladder_unsafe",
     "Ladder damaged, unsafe or used incorrectly",
     "Thang hỏng / dùng không an toàn",
     lambda en, z: "ladder" in en or z == "ladder"),

    # ---------- Lifting (cranes, hoists, cables) ----------
    ("Lifting_unsafe",
     "Unsafe lifting / rigging / crane operation",
     "Nâng hạ / cẩu / cáp không an toàn",
     lambda en, z: (
         "crane" in en or "cable hook" in en or "cable hooks" in en or
         "load test" in en or "lifting work is unsafe" in en or
         "load lifting" in en or "no signalman" in en or "no signalman for crane" in en or
         "no guide rope" in en or "no guide wire" in en or
         "rigger" in en or "cable, shackle" in en or "cables and shackles" in en or
         "no load display" in en or "below the object lifting" in en or
         "lifting equipment" in en or "operator dit not have" in en or
         "no isolation" in en and ("install" in en or "isolated install" in en) or
         z in ("lifting_work",
               "construction_equipment_gondola_suspended_scaffold_hoist_lifting_equipment",
               "steel_structures_precast_concrete_installation")
     )),

    # ---------- Excavation ----------
    ("Excavation_unsafe",
     "Excavation / pit / deep hole hazard",
     "Đào đất / hố sâu nguy hiểm",
     lambda en, z: (
         "excavator" in en or "excavation" in en or
         "pit" in en or "manhole" in en or
         "edge of the pit" in en or "landslide" in en or
         "underground" in en and "scanning" in en or
         z == "digging_deep_hole"
     )),

    # ---------- Electrical ----------
    ("Electrical_unsafe",
     "Electrical hazard (wiring, panel, equipment)",
     "Nguy cơ điện (dây / tủ / thiết bị)",
     lambda en, z: (
         "electrical" in en or "electric " in en or
         "wire" in en or "wiring" in en or "rcd" in en or
         "msb" in en or "dbs" in en or "elcb" in en or "rccb" in en or
         "grounding" in en or "ground clamp" in en or
         "submersible pump" in en or "cb and switches" in en or
         "emergency stop button" in en or "single-line" in en or
         "underground cable" in en or "transformer" in en or "transfomer" in en or
         "cutting wheel" in en or
         z == "electricity_and_electrical_equipment"
     )),

    # ---------- Welding (electric/gas) ----------
    ("Welding_unsafe",
     "Welding / cutting hazard",
     "Hàn / cắt không an toàn",
     lambda en, z: (
         ("welder" in en or "welding" in en or "weld" in en) or
         z == "electric_welding_gas_welding"
     )),

    # ---------- Hot work + sparks (when not specifically welding) ----------
    ("Hot_work_hazard",
     "Hot work spark or fire hazard",
     "Công việc sinh nhiệt / nguy cơ tia lửa",
     lambda en, z: (
         "hot work" in en or "spark" in en or "tia lửa" in en or
         "non-fire resistant" in en or "fire-resistant tarp" in en or
         z == "hot_work"
     )),

    # ---------- Pressure equipment / gas cylinders ----------
    ("Pressure_equipment_unsafe",
     "Pressure vessel / gas cylinder / pneumatic hazard",
     "Thiết bị áp lực / bình khí",
     lambda en, z: (
         "pressure" in en or "gas cylinder" in en or
         "flashback arrestor" in en or "non-return valve" in en or
         "compressed air" in en or "safety valve" in en or "gas cylinder cap" in en or
         z == "pressure_equipment"
     )),

    # ---------- Fire prevention (extinguishers, exit, drills) ----------
    ("Fire_prevention_unsafe",
     "Fire prevention deficiency",
     "Thiếu sót phòng cháy chữa cháy",
     lambda en, z: (
         "fire extinguisher" in en or "exit signs" in en or
         "emergency exit" in en or "assembly point" in en or
         "firefighting" in en or "fire prevention" in en or
         "fire and fighting" in en or
         z == "fire_prevention"
     )),

    # ---------- Chemicals / spills ----------
    ("Chemicals_hazmat_unsafe",
     "Hazardous chemicals / waste / spills / improper storage",
     "Hóa chất / chất thải nguy hại / rò rỉ / lưu trữ không an toàn",
     lambda en, z: (
         "chemical" in en or "msds" in en or "spill" in en or "spilled" in en or
         "paint" in en and "spill" in en or
         "hazardous waste" in en or
         z == "gas_and_chemicals_storage"
     )),

    # ---------- Confined space ----------
    ("Confined_space_unsafe",
     "Confined space hazard",
     "Không gian hạn chế",
     lambda en, z: z == "confined_space" or "confined space" in en),

    # ---------- Trucks / vehicles ----------
    ("Truck_vehicle_unsafe",
     "Unsafe truck or vehicle operation on site",
     "Xe tải / phương tiện không an toàn",
     lambda en, z: (
         "truck" in en or "vehicle" in en or "speed limit" in en or
         "designated route" in en or "driver" in en or
         "moving in violation" in en or
         z == "trucks"
     )),

    # ---------- Concrete pump / concrete work ----------
    ("Concrete_work_unsafe",
     "Concrete pump / pour hazard",
     "Đổ bê tông / bơm bê tông không an toàn",
     lambda en, z: (
         "concrete pipeline" in en or "concrete pump" in en or "pump truck" in en or
         "pump boom" in en or
         z == "concrete_work"
     )),

    # ---------- Formwork ----------
    ("Formwork_unsafe",
     "Formwork install or strip hazard",
     "Cốp pha / tháo cốp pha không an toàn",
     lambda en, z: (
         z == "formwork" or "formwork" in en or
         "rebar" in en
     )),

    # ---------- Pile pressing ----------
    ("Mass_piling_unsafe",
     "Pile pressing / unprotected pile heads",
     "Ép cọc / đầu cọc không che chắn",
     lambda en, z: (
         "pile head" in en or "pile pressing" in en or "danger area when the machine" in en or
         z == "mass_piling_work"
     )),

    # ---------- Workshop area ----------
    ("Workshop_area_unsafe",
     "Workshop / fabrication area hazard",
     "Khu vực gia công không an toàn",
     lambda en, z: (
         z == "workshop_area" or "painting area" in en or
         ("tray" in en and "machine" in en)
     )),

    # ---------- Warehouse ----------
    ("Warehouse_unsafe",
     "Warehouse storage hazard",
     "Khu vực kho không an toàn",
     lambda en, z: z == "warehouse_area"),

    # ---------- Material storage ----------
    ("Materials_storage_unsafe",
     "Material storage hazard (stacking, falling, blocking access)",
     "Tập kết vật tư không an toàn",
     lambda en, z: (
         "materials at risk of collapse" in en or "materials are not arranged" in en or
         "no signs to define the material storage" in en or "materials at risk of falling" in en or
         "materials temporarily stored unsafely" in en or "too many materials" in en or
         z == "material_area"
     )),

    # ---------- Mechanical equipment (general site machinery) ----------
    ("Equipment_machinery_unsafe",
     "Machinery / equipment unsafe or damaged",
     "Máy móc / thiết bị không an toàn",
     lambda en, z: (
         "machinery" in en or "equipment" in en and "leaking" in en or
         "lack of guarding" in en or "moving parts" in en or
         z == "mechanical_equipment_for_construction"
     )),

    # ---------- Smoking area ----------
    ("Smoking_area_unsafe",
     "Designated smoking area issue",
     "Khu vực hút thuốc không đảm bảo",
     lambda en, z: z == "smoke_area" or "smoking" in en or "cigarette" in en),

    # ---------- First aid kit ----------
    ("First_aid_kit_unsafe",
     "First-aid station deficiency",
     "Khu vực sơ cấp cứu thiếu sót",
     lambda en, z: (
         "first aid" in en or "medicines" in en or
         z == "first_aid_kit"
     )),

    # ---------- Parking ----------
    ("Parking_area_unsafe",
     "Parking area deficiency",
     "Khu vực đậu xe không đảm bảo",
     lambda en, z: z == "parking_area"),

    # ---------- Drinking water ----------
    ("Drinking_water_unsafe",
     "Drinking water area unhygienic",
     "Khu vực nước uống không vệ sinh",
     lambda en, z: z == "drinking_water_for_worker"),

    # ---------- Common area (eating / resting) ----------
    ("Common_area_unsafe",
     "Common eating / rest area unhygienic",
     "Khu vực ăn uống / nghỉ ngơi không vệ sinh",
     lambda en, z: (
         z == "common_area" or
         "eating and drinking at the wrong place" in en or
         "food, water bottles" in en or
         "area with many insects" in en
     )),

    # ---------- Site warning signs / barriers ----------
    ("Warning_signs_missing",
     "Site warning signs / barricades missing or damaged",
     "Biển báo / cảnh báo / hàng rào thiếu hoặc hỏng",
     lambda en, z: (
         z == "signal_board_warning_tape" or
         "signal board" in en or "warning tape" in en or
         "signs to define" in en or "no warning for work area" in en or
         "no warning, isolation" in en
     )),

    # ---------- Garbage / waste ----------
    ("Garbage_waste_unsafe",
     "Garbage / waste handling deficiency",
     "Thu gom / xử lý rác không đúng",
     lambda en, z: (
         z == "garbage_area_waste" or
         "unsorted" in en or "stored neatly" in en and "rác" in en or
         "hazardous waste" in en or "garbage outside designated areas" in en
     )),

    # ---------- Site access / gate / walkway ----------
    ("Site_access_unsafe",
     "Site access / gate / walkway hazard",
     "Cổng / lối đi không an toàn",
     lambda en, z: (
         z == "gate_and_safety_walkway" or
         "entrance to site" in en or "fence" in en and "security gate" in en or
         "unsafe walkway" in en or "no safe walkway" in en or
         "no safe passage" in en or "no safety ways" in en or
         "obstructive objects" in en or "obstacles, too narrow" in en or
         "no access" in en or "unsafe access" in en
     )),

    # ---------- Site lighting ----------
    ("Site_lighting_unsafe",
     "Site lighting deficiency",
     "Thiếu chiếu sáng",
     lambda en, z: "lighting" in en),

    # ---------- General housekeeping (catch-all) ----------
    ("Housekeeping_general",
     "General housekeeping / 5S / cleanliness",
     "Vệ sinh chung / 5S",
     lambda en, z: (
         "cleaning" in en or "maintenance inadequate" in en or
         "5s" in en or "implement 5s" in en or
         "stagnant water" in en or "vacuum cleaner" in en or
         "isolate" in en and ("dust" in en or "water" in en) or
         "garbage" in en or "waste materials" in en or
         "concrete or waste" in en or "iron rods are not covered" in en or
         "machinery and equipment leaking oil" in en or
         z == "housekeeping" or
         "not clean" in en or "dust" in en
     )),

    # ---------- Site general fallback ----------
    ("Site_general_unsafe",
     "Other unsafe site condition",
     "Điều kiện công trường không an toàn (khác)",
     lambda en, z: True),  # catch-all
]


def consolidate(en: str, parent_zone_slug: str) -> str:
    en_lower = en.lower()
    z_lower = parent_zone_slug.lower()
    for slug, _, _, predicate in PARENTS:
        try:
            if predicate(en_lower, z_lower):
                return slug
        except Exception:
            continue
    return "Site_general_unsafe"


def main() -> int:
    proposed_path = REPO_ROOT / "data" / "aecis_hse_proposed.json"
    if not proposed_path.exists():
        print("missing aecis_hse_proposed.json — run audit_aecis_hse_types.py", file=sys.stderr)
        return 1

    proposed = json.loads(proposed_path.read_text(encoding="utf-8"))
    entries = proposed["hse_types"]

    # Build absorbs map per parent
    absorbs: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        # Find the parent_zone slug from the original tree (we kept parent_label_en)
        parent_zone_label = e.get("primary_work_zone", "") or ""
        # Parent_zone slug normalisation matches what parse_aecis_hse_tree.py emits
        parent_zone = re.sub(r"[^A-Za-z0-9]+", "_", parent_zone_label).strip("_").lower()
        parent = consolidate(e["label_en"], parent_zone)
        absorbs[parent].append({
            "label_en": e["label_en"],
            "label_vn": e.get("label_vn", ""),
            "aecis_slug": e["slug"],
            "primary_work_zone": parent_zone_label,
        })

    # Build the parent metadata
    consolidated = []
    for slug, label_en, label_vn, _ in PARENTS:
        children = absorbs.get(slug, [])
        if not children:
            continue   # don't emit empty parents
        consolidated.append({
            "slug": slug,
            "label_en": label_en,
            "label_vn": label_vn,
            "size": len(children),
            "absorbs": children,
        })

    # Sort biggest-first for a more readable file
    consolidated.sort(key=lambda c: -c["size"])

    out_path = REPO_ROOT / "data" / "aecis_consolidation.json"
    out_path.write_text(json.dumps({
        "version": "aecis-consolidated-1.0",
        "source": "data/aecis_hse_proposed.json grouped by keyword rules in scripts/consolidate_aecis.py",
        "parent_class_count": len(consolidated),
        "absorbed_aecis_count": sum(c["size"] for c in consolidated),
        "parents": consolidated,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== AECIS CONSOLIDATION ===")
    print(f"Parent classes:    {len(consolidated)}")
    print(f"AECIS slugs absorbed: {sum(c['size'] for c in consolidated)}")
    print(f"")
    print(f"Per-parent count (sorted high-to-low):")
    for c in consolidated:
        print(f"  {c['size']:3d}  {c['slug']:<32}  {c['label_en']}")
    print(f"")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
