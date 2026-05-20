"""Output-side taxonomy translation.

The model + CLIP + SupCon + kNN all run in the internal AECIS
taxonomy (29 hse_type slugs × 9 location slugs). When the inspector
chooses a different jurisdiction (Canada CSA, USA OSHA, etc.),
the API rewrites slugs + labels through a static lookup before
serializing the response.

Nothing else changes — the trained head, embeddings, photo pool,
and accuracy targets are all measured against AECIS slugs.

Mappings live under data/taxonomy_mappings/{id}.json. Each file
declares its own id, label_en, country, and the per-axis lookup:

  {
    "id": "csa_z1000_ca",
    "label_en": "CSA Z1000 (Canada)",
    "country": "CA",
    "hse_types": { "<aecis_slug>": {"slug": "<target>", "label_en": "..."} },
    "locations": { ... }
  }

Empty objects mean pass-through (aecis_default does this).
"""
from __future__ import annotations
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger("violation.taxonomy_translator")

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPINGS_DIR = REPO_ROOT / "data" / "taxonomy_mappings"


def list_available() -> list[dict[str, str]]:
    """Return a UI-friendly list of installed mappings sorted by country."""
    out: list[dict[str, str]] = []
    if not MAPPINGS_DIR.exists():
        return out
    for p in sorted(MAPPINGS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse mapping %s: %s", p.name, e)
            continue
        out.append({
            "id": d.get("id") or p.stem,
            "label_en": d.get("label_en") or d.get("id") or p.stem,
            "country": d.get("country") or "",
            "version": d.get("version") or "",
        })
    # aecis_default first, then alphabetical by country
    out.sort(key=lambda r: (0 if r["id"] == "aecis_default" else 1,
                            r["country"], r["id"]))
    return out


@lru_cache(maxsize=16)
def _load(mapping_id: str) -> dict[str, Any] | None:
    p = MAPPINGS_DIR / f"{mapping_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("invalid mapping %s: %s", mapping_id, e)
        return None


class TaxonomyTranslator:
    """Map AECIS slugs to a target country's taxonomy at response time."""

    def __init__(self, mapping_id: str = "aecis_default") -> None:
        self.mapping_id = mapping_id
        self._data = _load(mapping_id) or {}
        self._hse = self._data.get("hse_types") or {}
        self._loc = self._data.get("locations") or {}

    @property
    def is_passthrough(self) -> bool:
        return not (self._hse or self._loc)

    def translate_hse(self, aecis_slug: str | None) -> dict[str, str] | None:
        """Return {slug, label_en, [label_fr], aecis_slug} for the target,
        or None if input is falsy."""
        if not aecis_slug:
            return None
        entry = self._hse.get(aecis_slug)
        if not entry:
            # No mapping found — pass through the AECIS slug unchanged.
            # This is safe even for partial mappings.
            return {"slug": aecis_slug, "label_en": "", "aecis_slug": aecis_slug}
        out = {"slug": entry["slug"], "aecis_slug": aecis_slug}
        for k in ("label_en", "label_fr", "label_es", "label_vn"):
            if entry.get(k):
                out[k] = entry[k]
        return out

    def translate_loc(self, aecis_slug: str | None) -> dict[str, str] | None:
        if not aecis_slug:
            return None
        entry = self._loc.get(aecis_slug)
        if not entry:
            return {"slug": aecis_slug, "label_en": "", "aecis_slug": aecis_slug}
        out = {"slug": entry["slug"], "aecis_slug": aecis_slug}
        for k in ("label_en", "label_fr", "label_es", "label_vn"):
            if entry.get(k):
                out[k] = entry[k]
        return out

    def translate_classification_response(self, body: dict[str, Any]) -> dict[str, Any]:
        """In-place rewrite of an /api/classify-shaped response.

        Adds `localized` keys alongside the original fields rather than
        replacing them — keeps the AECIS slug accessible for storage /
        cross-tenant analysis while letting the UI render the chosen
        taxonomy.
        """
        if self.is_passthrough:
            return body
        # hse_type primary
        if (slug := body.get("hse_type_slug")):
            body["hse_type_local"] = self.translate_hse(slug)
        # location primary
        if (slug := body.get("location_slug")):
            body["location_local"] = self.translate_loc(slug)
        # alternatives — top-3
        if isinstance(alts := body.get("hse_type_alternatives"), list):
            body["hse_type_alternatives_local"] = [
                self.translate_hse(a.get("slug")) | {"confidence": a.get("confidence")}
                for a in alts if isinstance(a, dict)
            ]
        if isinstance(alts := body.get("location_alternatives"), list):
            body["location_alternatives_local"] = [
                self.translate_loc(a.get("slug")) | {"confidence": a.get("confidence")}
                for a in alts if isinstance(a, dict)
            ]
        body["taxonomy_id"] = self.mapping_id
        return body
