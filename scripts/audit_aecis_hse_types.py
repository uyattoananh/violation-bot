"""Audit the 387 unique AECIS hse_types and propose a cleaned vocabulary.

Three checks:
  1. EXACT duplicates — same EN label appearing under multiple work zones.
     The AECIS tree has these because the same violation (e.g. "Inadequate
     PPEs", "No PTW/JSA") is listed under every relevant work zone. The AI
     doesn't care about location; we want one canonical entry.

  2. NEAR-duplicates — labels that say the same thing with slightly different
     wording. Detected via Jaccard similarity on word tokens.

  3. NOT-PHOTOGRAPHABLE filter — labels describing administrative items
     (paperwork, certificates, checklists, training records) or behaviors
     (fighting, alone-working, alcohol, phone use while walking) that can't
     be reliably classified from a single still photo. Identified by keyword
     pattern.

Outputs:
  data/aecis_hse_audit.json — full audit report (every label with its tags)
  data/aecis_hse_proposed.json — recommended cleaned vocabulary (the
      site-condition classes that survived all three filters, deduped by
      EN label).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- 1. Patterns for "not photographable" classification ----------

# Admin / paperwork — the violation IS that a document is missing or expired.
# In principle you could photograph the absence of a sticker on a machine,
# but in practice the AI can't reliably tell "this machine has no inspection
# stamp" from a photo. These belong on a checklist, not a vision classifier.
ADMIN_PATTERNS = [
    r"\bno (?:inspection )?certificate\b",
    r"\bno (?:safety )?card\b",
    r"\bgroup 3 safety card\b",
    r"\bsafety card or other document\b",
    r"\bno ptw\b",
    r"\bno jsa\b",
    r"\bworking without ptw\b",
    r"\bno permit\b",
    r"\bno checklist\b",
    r"\bdo not? have a checklist\b",
    r"\bno history\b",
    r"\bno declaration\b",
    r"\bno (?:maintenance|technical|insurance|periodic|electrical safety) "
        r"(?:record|inspection|certificate|safety inspection|cabinet safety)\b",
    r"\bno co/?cq\b",
    r"\bno qa[/-]qc\b",
    r"\bnot have qa/qc\b",
    r"\bno documentation\b",
    r"\bno documents?\b",
    r"\bnot have technical documents?\b",
    r"\bno (?:operator|technical|electrical) (?:document|information|inspection)\b",
    r"\bno shift handover\b",
    r"\bno entry and exit record\b",
    r"\bno insurance certificate\b",
    r"\bno operator information\b",
    r"\bno layout of temporary\b",
    r"\bno single-line\b",
    r"\bno safety instructions\b",
    r"\bno periodic\b",
    r"\bnot posted on notice board\b",
    r"\bworker is not safety trained\b",
    r"\bno confirmation of safety training\b",
    r"\bcrane not inspected daily\b",
    r"\bnot inspected daily\b",
    r"\bnot inspected initially\b",
    r"\bnot mos\b",
    r"\bmos.*not inspected\b",
    r"\bmos when digging\b",
    r"doesn['’]t have group 3\b",
    r"di[dt]n['’]t have certificate\b",
    r"\boperator not degree\b",
    r"\bwelder without\b",
    r"\brigger\b.*certificate",
    r"\belectric welder without a certificate\b",
    r"\bscaffolder without certificate\b",
    r"\bexcavator drivers? don['’]t have a group 3\b",
    r"\bcrane driver doesn['’]t have group 3\b",
    r"\boperator dit not have certificate\b",
    r"\brigger dit not have certificate\b",
    r"\bnot updated emergency number\b",
    r"\bcontact list is blurred\b",
    r"\bfirefighting team lacks certification\b",
    r"\bno fire drills\b",
    r"\bno fire prevention team\b",
    r"\bno fire prevention\b",
    r"\bno emergency respond measures\b",
    r"\bno emergency response (?:procedures|equipment|measures)\b",
    r"\bno emergency response procedures for chemical\b",
    r"\bno list of incompatible chemicals\b",
    r"\bno msds\b",
    r"\bno air testing\b",
    r"\bno entry and exit\b",
    r"\bno operator document\b",
    r"\bno watchman, supervisor\b",
    r"\bworker eat and drink at the workplace, causing\b",  # behavioral but admin overlap
    r"\bno parking area sign board\b",
    r"\bno checklist or checklist not good\b",
    r"\bno declaration of equipment\b",
    r"\bno smoke regulation on site\b",
    r"\bno safety net for floor by floor\b",
    r"\bno information of the pic\b",
    r"\bno warning signs for transfomer area\b",
    r"\bno warning signs for transformer area\b",
]

# Behavioral — describes a person's action, not a site condition. Hard to
# verify from a single photo (often visible only with video / observation).
BEHAVIORAL_PATTERNS = [
    r"\bworker fighting\b",
    r"\bworker (?:engaging|threatening)\b",
    r"\bworker threatening\b",
    r"\bbringing or drinking alcohol\b",
    r"\bworker work(?:ing)? alone\b",
    r"\bworker working overtime\b",
    r"\busing a phone while walking\b",
    r"\bentering a prohibited area\b",
    r"\bbreaks not maintained\b",
    r"\bsmoking in the wrong place\b",
    r"\burinating, defecating\b",
    r"\beating and drinking at the wrong place\b",
    r"\bworker eat and drink at the workplace, causing\b",
    r"\bthrow material, equipment\b",
    r"\bthrowing materials\b",  # arguably could be photographed mid-throw, but rare
    r"\bonly one person is allowed\b",
    r"\bdo not stand on the top of the ladder\b",
    r"\bdo not hang heavy objects\b",
    r"\bonly charge batteries\b",
    r"\bno wear safety harness when working\b",  # this overlaps PPE
    r"\blet the other holding on vehicles\b",
    r"\bdo not arbitrarily connect\b",
    r"\bdo not use single-layer\b",
    r"\bdo not using bamboo and wooden ladder\b",  # design rule, not visible
    r"\bladders are not used according to the design\b",
    r"\bladder must not be connected together\b",
    r"\bonly one person is allowed to work\b",
    r"\bplace ladder on crate\b",
    r"\buse ladder on scaffolding\b",
    r"\bladders are placed too close to power lines\b",  # could photograph but ambiguous
    r"\bthe top of the ladder must not be at least\b",
    r"\bno people is holding the foot of the ladder\b",
    r"\bworker working alone \(must be 2 people\b",
    r"\bworkers? resting at the workplace\b",
    r"\bno warning area\b",
]


def classify_photographability(label: str) -> str:
    s = label.lower()
    for p in ADMIN_PATTERNS:
        if re.search(p, s):
            return "admin"
    for p in BEHAVIORAL_PATTERNS:
        if re.search(p, s):
            return "behavioral"
    return "site_condition"


# ---------- 2. Near-duplicate detection ----------

def _tokens(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    stop = {"the", "a", "an", "is", "are", "of", "to", "and", "or", "in",
            "for", "on", "at", "by", "with", "no", "not", "but"}
    return {t for t in s.split() if t and t not in stop and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicates(labels: list[str], threshold: float = 0.7) -> list[tuple[str, str, float]]:
    pairs = []
    token_cache = {l: _tokens(l) for l in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            sim = _jaccard(token_cache[a], token_cache[b])
            if sim >= threshold:
                pairs.append((a, b, sim))
    return sorted(pairs, key=lambda x: -x[2])


# ---------- main ----------

def main() -> int:
    tree_path = REPO_ROOT / "data" / "aecis_hse_tree.json"
    if not tree_path.exists():
        print("missing aecis_hse_tree.json — run parse_aecis_hse_tree.py first", file=sys.stderr)
        return 1
    tree = json.loads(tree_path.read_text(encoding="utf-8"))

    # Flatten
    all_entries: list[dict] = []
    for loc in tree["locations"]:
        for h in loc["hse_types"]:
            all_entries.append({
                "parent_location": loc["slug"],
                "parent_label_en": loc["label_en"],
                "label_en": h["label_en"],
                "label_vn": h["label_vn"],
                "slug": h["slug"],
            })

    # 1. Exact duplicates by EN label
    by_en: dict[str, list[dict]] = defaultdict(list)
    for e in all_entries:
        by_en[e["label_en"].strip()].append(e)

    exact_dupes = {k: v for k, v in by_en.items() if len(v) > 1}

    # 2. Photographability classification (per unique EN)
    photographability: dict[str, str] = {}
    for label in by_en:
        photographability[label] = classify_photographability(label)

    counts = {"admin": 0, "behavioral": 0, "site_condition": 0}
    for v in photographability.values():
        counts[v] += 1

    # 3. Near-duplicates among site_condition entries only
    site_labels = [k for k, v in photographability.items() if v == "site_condition"]
    near_dupes = find_near_duplicates(site_labels, threshold=0.7)

    # ----- write audit report -----
    audit = {
        "total_entries": len(all_entries),
        "unique_en_labels": len(by_en),
        "photographability_breakdown": counts,
        "exact_duplicate_count": len(exact_dupes),
        "near_duplicate_pair_count": len(near_dupes),
        "exact_duplicates": [
            {"label_en": k, "appears_in_locations": [e["parent_label_en"] for e in v]}
            for k, v in sorted(exact_dupes.items(), key=lambda x: -len(x[1]))
        ],
        "near_duplicates": [
            {"a": a, "b": b, "sim": round(s, 3)} for a, b, s in near_dupes
        ],
        "filtered_out_admin": sorted(k for k, v in photographability.items() if v == "admin"),
        "filtered_out_behavioral": sorted(k for k, v in photographability.items() if v == "behavioral"),
    }

    audit_path = REPO_ROOT / "data" / "aecis_hse_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----- write proposed cleaned vocabulary -----
    proposed: list[dict] = []
    seen = set()
    for e in all_entries:
        en = e["label_en"].strip()
        if en in seen:
            continue
        if photographability.get(en) != "site_condition":
            continue
        seen.add(en)
        proposed.append({
            "slug": e["slug"],
            "label_en": en,
            "label_vn": e["label_vn"],
            # Keep first-seen parent location as a hint for the prompt
            "primary_work_zone": e["parent_label_en"],
        })

    proposed_path = REPO_ROOT / "data" / "aecis_hse_proposed.json"
    proposed_path.write_text(json.dumps({
        "version": "aecis-cleaned-1.0",
        "source": "data/aecis_hse_tree.json filtered for photographability + dedup",
        "filter_summary": {
            "started_with": len(by_en),
            "dropped_admin": counts["admin"],
            "dropped_behavioral": counts["behavioral"],
            "kept_site_condition": counts["site_condition"],
        },
        "hse_types": proposed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----- print summary -----
    print(f"=== AECIS HSE TYPE AUDIT ===")
    print(f"Started with    : {len(all_entries)} entries ({len(by_en)} unique EN labels)")
    print(f"")
    print(f"Photographability breakdown:")
    print(f"  admin/paperwork     : {counts['admin']:3d}  -> DROPPED")
    print(f"  behavioral/conduct  : {counts['behavioral']:3d}  -> DROPPED")
    print(f"  site condition      : {counts['site_condition']:3d}  -> KEPT (proposed vocabulary)")
    print(f"")
    print(f"Exact duplicates (same EN, multiple work zones): {len(exact_dupes)}")
    if exact_dupes:
        top = sorted(exact_dupes.items(), key=lambda x: -len(x[1]))[:8]
        for k, v in top:
            print(f"  {len(v):2d}x  {k[:80]}")
    print(f"")
    print(f"Near-duplicate pairs (Jaccard >= 0.7): {len(near_dupes)}")
    if near_dupes:
        for a, b, s in near_dupes[:8]:
            print(f"  {s:.2f}  {a[:55]:<55}  <->  {b[:55]}")
    print(f"")
    print(f"Wrote:")
    print(f"  {audit_path}")
    print(f"  {proposed_path}  <= {len(proposed)} site-condition hse_types deduped")
    print(f"")
    print(f"Training-data check (1108 confirmed labels in pgvector):")
    print(f"  avg photos/class = 1108 / {len(proposed)} = {1108/len(proposed):.1f}")
    print(f"  (rule of thumb: <5 photos per class -> poor recall on that class)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
