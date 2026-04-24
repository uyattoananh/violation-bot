"""
Construction Site Safety Violation Taxonomy — Vietnamese Regulations

Based on:
  - QCVN 18:2021/BXD  (National Technical Regulation on Safety in Construction)
  - Law 84/2015/QH13   (Law on Occupational Safety and Hygiene)
  - Decree 06/2021/ND-CP (Construction safety requirements)
  - Decree 12/2022/ND-CP (Administrative penalties — labour/OHS)
  - Decree 16/2022/ND-CP (Administrative penalties — construction sector)

Each violation is mapped to its governing regulation section, penalty article,
and fine range (VND). Only visually identifiable violations are included.
"""

# ---------------------------------------------------------------------------
# Category 1 — Site Layout & Infrastructure (QCVN 18:2021/BXD §2.1)
# ---------------------------------------------------------------------------
# Penalty: Decree 12/2022 Art.21 Cl.3 (VND 20–25M) for site safety equipment;
#          Decree 12/2022 Art.22 Cl.5 (VND 10–15M) for missing warning signs.

# ---------------------------------------------------------------------------
# Category 2 — Structural Fall Protection (QCVN 18:2021/BXD §2.2, §2.7)
# ---------------------------------------------------------------------------
# Penalty: Decree 12/2022 Art.22 Cl.9 (VND 25–40M) for QCVN violations;
#          Decree 12/2022 Art.22 Cl.10 (VND 30–40M) for compelling unprotected work.

# ---------------------------------------------------------------------------
# Category 3 — Electrical & Fire Hazards (QCVN 18:2021/BXD §2.16, §2.17)
# ---------------------------------------------------------------------------
# Penalty: Decree 12/2022 Art.22 Cl.7 (VND 20–25M);
#          Decree 16/2022 Art.32 (construction-sector penalties).

# ---------------------------------------------------------------------------
# Category 4 — Excavation & Earthwork (QCVN 18:2021/BXD §2.8)
# ---------------------------------------------------------------------------
# Penalty: Decree 12/2022 Art.22 Cl.9 (VND 25–40M);
#          Decree 16/2022 Art.32.

# ---------------------------------------------------------------------------
# Category 5 — Environmental & Hygiene Controls (QCVN 18:2021/BXD §2.18, §2.20)
# ---------------------------------------------------------------------------
# Penalty: Decree 12/2022 Art.21 Cl.3 (VND 20–25M).

VIOLATION_CLASSES = {
    # ===== Category 1: Site Layout & Infrastructure =====
    0: {
        "name": "non_compliant_fencing",
        "label": "Hàng rào không đạt chuẩn",
        "label_en": "Missing / Non-Compliant Perimeter Fencing",
        "regulation": "QCVN 18:2021/BXD §2.1",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Missing or non-compliant perimeter fencing (height < 2.0m in urban areas)",
        "severity": "serious",
        "category": "site_layout",
    },
    1: {
        "name": "inadequate_lighting",
        "label": "Thiếu chiếu sáng",
        "label_en": "Inadequate Lighting",
        "regulation": "QCVN 18:2021/BXD §2.1, §2.20",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Inadequate lighting for nighttime operations or hazardous zones (min 50 Lux enclosed areas)",
        "severity": "serious",
        "category": "site_layout",
    },
    2: {
        "name": "missing_safety_signs",
        "label": "Thiếu biển báo an toàn",
        "label_en": "Missing / Illegible Safety Warning Signs",
        "regulation": "QCVN 18:2021/BXD §2.1; Law 84/2015",
        "penalty": "Decree 12/2022 Art.22 Cl.5",
        "fine_vnd": "10,000,000 – 15,000,000",
        "description": "Missing, illegible, or incorrectly placed safety warning signs (must be in Vietnamese)",
        "severity": "moderate",
        "category": "site_layout",
    },
    3: {
        "name": "blocked_egress",
        "label": "Chặn lối thoát hiểm",
        "label_en": "Obstructed Emergency Egress Routes",
        "regulation": "QCVN 18:2021/BXD §2.1",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Obstruction of emergency egress routes and passageways",
        "severity": "serious",
        "category": "site_layout",
    },
    4: {
        "name": "unauthorized_access",
        "label": "Xâm nhập khu vực cấm",
        "label_en": "Unauthorized Entry to Restricted Zones",
        "regulation": "QCVN 18:2021/BXD §2.1",
        "penalty": "Decree 12/2022 Art.22 Cl.5",
        "fine_vnd": "10,000,000 – 15,000,000",
        "description": "Unauthorized entry or uncontrolled access points to restricted zones",
        "severity": "moderate",
        "category": "site_layout",
    },

    # ===== Category 2: Structural Fall Protection =====
    5: {
        "name": "missing_guardrails",
        "label": "Thiếu lan can bảo vệ",
        "label_en": "Missing / Incomplete Guardrail Systems",
        "regulation": "QCVN 18:2021/BXD §2.2, §2.7",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Missing or incomplete guardrail systems (top rail, mid-rail, toe-board); min 1.1m height at 2.0m+",
        "severity": "serious",
        "category": "fall_protection",
    },
    6: {
        "name": "uncovered_openings",
        "label": "Lỗ mở không che chắn",
        "label_en": "Uncovered Floor / Wall / Shaft Openings",
        "regulation": "QCVN 18:2021/BXD §2.7",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Uncovered or unsecured floor, wall, and elevator shaft openings",
        "severity": "serious",
        "category": "fall_protection",
    },
    7: {
        "name": "unstable_scaffolding",
        "label": "Giàn giáo không ổn định",
        "label_en": "Unstable Scaffolding",
        "regulation": "QCVN 18:2021/BXD §2.2",
        "penalty": "Decree 12/2022 Art.22 Cl.9; Decree 16/2022 Art.32",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Unstable scaffolding — missing base plates, cross-bracing, or wall tie-ins; safety factor min 4x",
        "severity": "serious",
        "category": "fall_protection",
    },
    8: {
        "name": "insecure_platforms",
        "label": "Sàn công tác không an toàn",
        "label_en": "Insecure Temporary Work Platforms / Walkways",
        "regulation": "QCVN 18:2021/BXD §2.2, §2.7",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Insecure temporary work platforms or walkways (min width 50cm)",
        "severity": "serious",
        "category": "fall_protection",
    },
    9: {
        "name": "no_debris_netting",
        "label": "Thiếu lưới chống rơi",
        "label_en": "Lack of Overhead Protection / Debris Netting",
        "regulation": "QCVN 18:2021/BXD §2.1, §2.7",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Lack of overhead protection or debris netting in high-rise zones (danger zone: 5m <20m height, 7m 20-70m, 10m 70-120m)",
        "severity": "serious",
        "category": "fall_protection",
    },

    # ===== Category 3: Electrical & Fire Hazards =====
    10: {
        "name": "exposed_wiring",
        "label": "Dây điện hở",
        "label_en": "Exposed Electrical Wiring / Live Cables on Ground",
        "regulation": "QCVN 18:2021/BXD §2.16; QCVN 01:2020/BCT",
        "penalty": "Decree 12/2022 Art.22 Cl.7",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Exposed electrical wiring or live cables lying on the ground (safe voltage max 42V AC)",
        "severity": "serious",
        "category": "electrical_fire",
    },
    11: {
        "name": "no_elcb",
        "label": "Thiếu ELCB trên tủ điện tạm",
        "label_en": "Temporary Power Board Lacking ELCB",
        "regulation": "QCVN 18:2021/BXD §2.16",
        "penalty": "Decree 12/2022 Art.22 Cl.7",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Temporary power distribution boards lacking Earth Leakage Circuit Breakers (ELCB)",
        "severity": "serious",
        "category": "electrical_fire",
    },
    12: {
        "name": "unprotected_cables",
        "label": "Cáp điện không bảo vệ",
        "label_en": "Cables Crossing Vehicle Paths Without Conduit",
        "regulation": "QCVN 18:2021/BXD §2.16",
        "penalty": "Decree 12/2022 Art.22 Cl.7",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Lack of protective conduits for cables crossing vehicle paths",
        "severity": "moderate",
        "category": "electrical_fire",
    },
    13: {
        "name": "flammable_storage_violation",
        "label": "Lưu trữ vật liệu cháy nổ sai quy định",
        "label_en": "Improper Flammable / Explosive Material Storage",
        "regulation": "QCVN 18:2021/BXD §2.17; QCVN 06:2021/BXD",
        "penalty": "Decree 16/2022 Art.32",
        "fine_vnd": "varies",
        "description": "Improper storage or segregation of flammable and explosive materials",
        "severity": "serious",
        "category": "electrical_fire",
    },
    14: {
        "name": "no_fire_equipment",
        "label": "Thiếu thiết bị chữa cháy",
        "label_en": "Absent / Non-Functioning Firefighting Equipment",
        "regulation": "QCVN 06:2021/BXD; QCVN 18:2021/BXD §2.17",
        "penalty": "Decree 16/2022 Art.32",
        "fine_vnd": "varies",
        "description": "Absence of functioning firefighting equipment at designated stations",
        "severity": "serious",
        "category": "electrical_fire",
    },

    # ===== Category 4: Excavation & Earthwork =====
    15: {
        "name": "unshored_excavation",
        "label": "Hố đào không chống đỡ",
        "label_en": "Excavation Lacking Shoring / Bracing (>1.5m)",
        "regulation": "QCVN 18:2021/BXD §2.8",
        "penalty": "Decree 12/2022 Art.22 Cl.9; Decree 16/2022 Art.32",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Lack of shoring, bracing, or benching in excavations deeper than 1.5m",
        "severity": "serious",
        "category": "excavation",
    },
    16: {
        "name": "surcharge_loading",
        "label": "Chất tải gần mép hố đào",
        "label_en": "Surcharge Loading Near Excavation Edge",
        "regulation": "QCVN 18:2021/BXD §2.8",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Surcharge loading (soil/materials) placed within 1.0m of excavation edges",
        "severity": "serious",
        "category": "excavation",
    },
    17: {
        "name": "no_vehicle_barriers",
        "label": "Thiếu rào chắn xe tại hố đào",
        "label_en": "Missing Vehicle Stop Blocks at Pit Edges",
        "regulation": "QCVN 18:2021/BXD §2.8",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Missing physical stop blocks or barriers for vehicles at pit edges",
        "severity": "serious",
        "category": "excavation",
    },
    18: {
        "name": "waterlogged_trench",
        "label": "Hố đào ngập nước",
        "label_en": "Water Accumulation in Trenches",
        "regulation": "QCVN 18:2021/BXD §2.8",
        "penalty": "Decree 12/2022 Art.22 Cl.9",
        "fine_vnd": "25,000,000 – 40,000,000",
        "description": "Accumulation of water in trenches compromising soil stability",
        "severity": "moderate",
        "category": "excavation",
    },

    # ===== Category 5: Environmental & Hygiene Controls =====
    19: {
        "name": "no_confined_space_ventilation",
        "label": "Thiếu thông gió không gian kín",
        "label_en": "Lack of Ventilation in Confined Spaces",
        "regulation": "QCVN 18:2021/BXD §2.18, §2.20",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Lack of mechanical ventilation in confined spaces or underground basements",
        "severity": "serious",
        "category": "environmental",
    },
    20: {
        "name": "dust_noise_vibration",
        "label": "Bụi / tiếng ồn / rung vượt ngưỡng",
        "label_en": "Uncontrolled Dust, Noise, or Vibration",
        "regulation": "QCVN 18:2021/BXD §2.18",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Uncontrolled dust, noise, or vibration exceeding technical thresholds",
        "severity": "moderate",
        "category": "environmental",
    },
    21: {
        "name": "improper_waste_discharge",
        "label": "Xả thải xây dựng sai quy định",
        "label_en": "Improper Construction Waste Discharge",
        "regulation": "QCVN 18:2021/BXD §2.20",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Improper discharge of liquid construction waste into public sewage",
        "severity": "moderate",
        "category": "environmental",
    },
    22: {
        "name": "debris_stagnant_water",
        "label": "Mảnh vỡ sắc nhọn / nước đọng",
        "label_en": "Sharp Debris, Nails, or Stagnant Water in Work Areas",
        "regulation": "QCVN 18:2021/BXD §2.1, §2.20",
        "penalty": "Decree 12/2022 Art.21 Cl.3",
        "fine_vnd": "20,000,000 – 25,000,000",
        "description": "Accumulation of sharp debris, nails, or stagnant water in work areas",
        "severity": "moderate",
        "category": "environmental",
    },
}

NUM_CLASSES = len(VIOLATION_CLASSES)

# Category groupings for reporting
CATEGORIES = {
    "site_layout": "Mặt bằng & Hạ tầng công trường",
    "fall_protection": "Chống rơi ngã",
    "electrical_fire": "Điện & Phòng cháy",
    "excavation": "Đào đất & Công trình ngầm",
    "environmental": "Môi trường & Vệ sinh",
}

CATEGORY_EN = {
    "site_layout": "Site Layout & Infrastructure",
    "fall_protection": "Structural Fall Protection",
    "electrical_fire": "Electrical & Fire Hazards",
    "excavation": "Excavation & Earthwork",
    "environmental": "Environmental & Hygiene Controls",
}

CLASS_NAMES = [VIOLATION_CLASSES[i]["name"] for i in range(NUM_CLASSES)]
CLASS_LABELS = [VIOLATION_CLASSES[i]["label"] for i in range(NUM_CLASSES)]
CLASS_LABELS_EN = [VIOLATION_CLASSES[i]["label_en"] for i in range(NUM_CLASSES)]


def get_class_index(name: str) -> int:
    """Get class index by violation name."""
    for idx, info in VIOLATION_CLASSES.items():
        if info["name"] == name:
            return idx
    raise ValueError(f"Unknown violation class: {name}")


def get_regulation(class_idx: int) -> str:
    """Get Vietnamese regulation reference for a given class index."""
    return VIOLATION_CLASSES[class_idx]["regulation"]


def get_penalty_ref(class_idx: int) -> str:
    """Get penalty decree article for a given class index."""
    return VIOLATION_CLASSES[class_idx]["penalty"]


def get_classes_by_category(category: str) -> list[int]:
    """Return all class indices belonging to a category."""
    return [
        idx for idx, info in VIOLATION_CLASSES.items()
        if info["category"] == category
    ]


def format_prediction(class_idx: int, confidence: float) -> dict:
    """Format a single prediction into a structured result."""
    info = VIOLATION_CLASSES[class_idx]
    return {
        "class_index": class_idx,
        "violation": info["name"],
        "label_vi": info["label"],
        "label_en": info["label_en"],
        "regulation": info["regulation"],
        "penalty": info["penalty"],
        "fine_vnd": info["fine_vnd"],
        "severity": info["severity"],
        "category": info["category"],
        "confidence": round(confidence, 4),
    }
