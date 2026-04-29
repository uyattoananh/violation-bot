"""Phase-1 zero-shot violation classifier.

Given a photo + the taxonomy.json prepared vocabularies, calls a Claude-class
VLM and returns a structured JSON label:

    {
        "location": {"slug": "Housekeeping", "label_en": "...",
                     "confidence": 0.92},
        "hse_type": {"slug": "Garbage_outside_designated_areas", "label_en": "...",
                     "confidence": 0.87},
        "rationale": "Short explanation for the inspector review UI.",
        "raw_response": {...}
    }

Provider auto-detection (first match wins):
  1. OPENROUTER_API_KEY set   -> OpenRouter via OpenAI-compatible SDK
  2. ANTHROPIC_API_KEY set    -> Anthropic SDK direct (supports Batch API)

Env vars:
    # OpenRouter path (pick one provider):
    OPENROUTER_API_KEY      e.g. sk-or-v1-...
    OPENROUTER_MODEL        optional, default "anthropic/claude-sonnet-4.5"
    OPENROUTER_REFERER      optional, app URL for OpenRouter analytics
    OPENROUTER_TITLE        optional, app name for OpenRouter analytics

    # Anthropic direct path:
    ANTHROPIC_API_KEY
    ANTHROPIC_MODEL         optional, default "claude-sonnet-4-5-20250929"

    # Common:
    VIOLATION_TAXONOMY      optional path, default "<repo>/taxonomy.json"

CLI usage (single image):
    python -m src.zero_shot --image path/to/photo.jpg
    python -m src.zero_shot --image photo.jpg --verbose
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Load .env from the repo root (if present) so API keys flow automatically.
try:
    from dotenv import load_dotenv
    _repo_env = Path(__file__).resolve().parents[1] / ".env"
    if _repo_env.exists():
        load_dotenv(_repo_env)
except ImportError:
    pass

# Provider SDKs (anthropic / openai) are imported lazily in the functions that
# use them, so this module remains importable without either installed.

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = REPO_ROOT / "taxonomy.json"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
# Default model: Gemini 2.5 Flash. Switched from Sonnet 4.5 after a 50-photo
# head-to-head eval showed Flash with +12pp HSE accuracy AND ~89% lower cost
# (see scripts/evaluate_models.py and evaluation_models_*.json). Per the
# decision rule "only switch if BOTH cost goes down AND accuracy goes up,"
# Flash is the only candidate that won. Set OPENROUTER_MODEL in env to
# override (e.g. OPENROUTER_MODEL=anthropic/claude-sonnet-4.5 to roll back).
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "classifier.md"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Photo-RAG: how many nearest neighbours to retrieve and how much to let
# them influence the prompt. Set RAG_NEIGHBOURS=0 to disable retrieval entirely.
# k=5 was tested at 60.0% post-dedup; k=12 regressed to 54.1% (the extra
# borderline neighbours dilute the strongest signal). Stay with 5 unless
# a future eval shows a different sweet spot.
RAG_NEIGHBOURS_DEFAULT = 5


# ---------- data shapes ----------

@dataclass
class AxisLabel:
    slug: str
    label_en: str
    label_vn: str = ""
    confidence: float = 0.0


@dataclass
class Classification:
    location: AxisLabel
    hse_type: AxisLabel
    rationale: str
    raw_response: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Top-3 alternatives (including the primary). Useful for the review UI
    # so the inspector can one-click switch to a runner-up.
    hse_type_alternatives: list["AxisLabel"] = None  # type: ignore[assignment]
    location_alternatives: list["AxisLabel"] = None  # type: ignore[assignment]
    # Fine-grained AECIS sub-type produced by Stage 2 of the two-stage
    # classifier. None when fine_grained=False or when Stage 2 confidence
    # was below threshold (the model said "none of these specific items
    # fit"). Both fields are optional so older callers keep working.
    fine_hse_type: AxisLabel | None = None
    fine_hse_type_alternatives: list["AxisLabel"] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.hse_type_alternatives is None:
            self.hse_type_alternatives = []
        if self.location_alternatives is None:
            self.location_alternatives = []
        if self.fine_hse_type_alternatives is None:
            self.fine_hse_type_alternatives = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": asdict(self.location),
            "hse_type": asdict(self.hse_type),
            "hse_type_alternatives": [asdict(a) for a in self.hse_type_alternatives],
            "location_alternatives": [asdict(a) for a in self.location_alternatives],
            "fine_hse_type": asdict(self.fine_hse_type) if self.fine_hse_type else None,
            "fine_hse_type_alternatives": [asdict(a) for a in self.fine_hse_type_alternatives],
            "rationale": self.rationale,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# ---------- taxonomy + prompt ----------

def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("VIOLATION_TAXONOMY") or DEFAULT_TAXONOMY)
    return json.loads(path.read_text(encoding="utf-8"))


def format_taxonomy_for_prompt(tax: dict[str, Any]) -> str:
    """Produce the bilingual vocabulary block that goes into the system prompt.
    Slugs are the stable machine identifiers the model MUST return verbatim.
    """
    loc_lines = []
    for loc in tax["locations"]:
        loc_lines.append(
            f"  - {loc['slug']}: {loc['label_en']}"
            + (f"  /  {loc['label_vn']}" if loc.get("label_vn") else "")
        )
    hse_lines = []
    for hse in tax["hse_types"]:
        hse_lines.append(
            f"  - {hse['slug']}: {hse['label_en']}"
            + (f"  /  {hse['label_vn']}" if hse.get("label_vn") else "")
        )
    return (
        "LOCATIONS:\n" + "\n".join(loc_lines)
        + "\n\nHSE TYPES:\n" + "\n".join(hse_lines)
    )


SYSTEM_PROMPT = """You are a construction safety violation classifier for \
Vietnamese construction sites.

You receive ONE site photograph. You must classify it on TWO independent axes:
  1. LOCATION — where/what kind of site area or activity context
  2. HSE TYPE — the specific safety violation present

Both labels MUST be chosen from the provided vocabularies below. Return the
`slug` field verbatim — do not invent new labels. If the image does not clearly
show any violation, pick the most plausible location and the `hse_type` whose
`label_en` best matches what you see (do not refuse).

==== HOW TO PICK AN HSE_TYPE ====
The HSE_TYPES vocabulary below contains 29 categories aligned with the
AECIS canonical safety taxonomy. Each category has an EN/VN label.

Decide in this order:

1. WHAT IS THE SUBJECT of the photo?
   - A specific worker visibly at risk        → Fall_protection_personal /
                                                 PPE_missing / Ladder_unsafe
   - A specific PIECE OF EQUIPMENT             → Scaffolding_unsafe /
     (scaffold, ladder, crane, panel, etc.)     Lifting_unsafe / Electrical_unsafe /
                                                 Pressure_equipment_unsafe / etc.
   - A SITE FEATURE (edge, opening, pit)       → Edge_protection_missing /
                                                 Excavation_unsafe / Floor_opening
   - A WORK ACTIVITY in progress               → Hot_work_hazard /
     (welding, hot work, concrete pump)         Concrete_work_unsafe
   - GENERAL MESS / DEBRIS                     → Housekeeping_general
   - SOMETHING ELSE that is unsafe but does
     not fit any specific category             → Site_general_unsafe (catch-all)

2. ESCAPE-FROM-CATCH-ALL THRESHOLD
   Housekeeping_general and Site_general_unsafe are the residual catch-alls.
   To pick anything else, you must be able to point at a SPECIFIC visible
   feature (a wire, a scaffold, a hook, a worker without a harness, a
   chemical drum, a pile head). If the photo is just messy without a
   specific hazard, pick Housekeeping_general.

3. SPECIFICITY TIE-BREAK
   When two classes both apply, pick the MORE SPECIFIC one. Examples:
   - "Worker without harness on scaffold" → Fall_protection_personal
     (NOT Scaffolding_unsafe — the worker is the subject)
   - "Damaged scaffold standing alone"     → Scaffolding_unsafe
     (NOT Site_general_unsafe — scaffold is the specific subject)
   - "Welder welding without sparks contained" → Hot_work_hazard
     (welding is hot work; the spark hazard is the safety concern)
   - "Crane hook with no safety latch"     → Lifting_unsafe
     (the hook is the specific feature)

4. RAG NEIGHBOURS as prior
   If RETRIEVED REFERENCE PHOTOS appear below, treat the dominant label
   among them as a strong prior. Override only when the photo shows a
   specific hazard the neighbours don't reflect.

==== OUTPUT ====
ONE JSON object, no prose before or after, no markdown fences. Include your
primary pick PLUS TOP-2 runner-up alternatives per axis (ordered by
decreasing confidence). The UI shows runners-up as one-click corrections,
so they matter — pick genuinely different alternatives, don't duplicate.

{
  "location": {
    "slug": "<best location slug>",
    "confidence": <float 0..1>
  },
  "location_alternatives": [
    {"slug": "<2nd-best slug>", "confidence": <float>},
    {"slug": "<3rd-best slug>", "confidence": <float>}
  ],
  "hse_type": {
    "slug": "<best hse_type slug>",
    "confidence": <float 0..1>
  },
  "hse_type_alternatives": [
    {"slug": "<2nd-best slug>", "confidence": <float>},
    {"slug": "<3rd-best slug>", "confidence": <float>}
  ],
  "rationale": "<one short sentence, max 40 words>"
}

Confidence guidance:
  > 0.85   : strong visual evidence (the violation is visible and unambiguous)
  0.60-0.85: plausible but not definitive
  < 0.60   : low — inspector should review

Taxonomies follow."""


def build_user_message_anthropic(image_b64: str, taxonomy_block: str, media_type: str) -> list[dict]:
    """Anthropic Messages API image-content block.

    The taxonomy text (~6.8k tokens, identical across all calls) is marked
    ephemeral-cacheable. This reduces the per-image input cost by ~90% after
    the first call and persists for 5 minutes of idle time.
    """
    return [
        {
            "type": "text",
            "text": taxonomy_block,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        },
        {"type": "text", "text": "Classify the photo above. Return JSON only."},
    ]


def build_user_message_openai(image_b64: str, taxonomy_block: str, media_type: str) -> list[dict]:
    """OpenAI-compatible chat-completions image content (works on OpenRouter).

    OpenRouter passes the `cache_control` extension through to Anthropic for
    provider-level prompt caching. Same ~90% input-cost reduction applies.
    """
    return [
        {
            "type": "text",
            "text": taxonomy_block,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        },
        {"type": "text", "text": "Classify the photo above. Return JSON only."},
    ]


def _system_with_cache() -> list[dict]:
    """Anthropic accepts `system` as either a string or a list of blocks.
    Return the list form so the system prompt (~400 tokens) is also cached."""
    return [{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]


# ---------- photo-RAG retrieval ----------

def _retrieve_similar_labels(image_path: Path, k: int) -> list[dict[str, Any]]:
    """Given a query photo, return the k visually nearest neighbour labels.

    Requires Supabase + pgvector + the photo_embeddings table populated (see
    scripts/embed_dataset.py). Returns [] if retrieval is unavailable — the
    classifier then falls back to pure zero-shot with no reference hints.
    """
    try:
        from src.embeddings import embed_image
    except Exception as e:  # noqa: BLE001
        log.debug("embeddings unavailable: %s", e)
        return []
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        return []
    try:
        from supabase import create_client
    except Exception:  # noqa: BLE001
        return []

    try:
        qvec = embed_image(image_path)
        db = create_client(url, key)
        resp = db.rpc("match_photo_embeddings", {
            "query_embedding": qvec.tolist(),
            "match_k": k,
        }).execute()
        return resp.data or []
    except Exception as e:  # noqa: BLE001
        log.warning("RAG retrieval failed (%s); continuing without hints", e)
        return []


def _format_corrections_block(
    corrections: list[dict[str, Any]],
    tax: dict[str, Any],
) -> str:
    """In-session learning. When the worker picks up a classify job, it
    fetches the most-recent inspector corrections in the SAME batch and
    passes them through here. We surface them as in-context examples so
    the model can adapt to systematic patterns within a single review
    session — e.g. the inspector keeps re-labeling "dirty walkway" as
    Site_access_unsafe rather than Housekeeping_general, and after seeing
    a few of those the model picks up the pattern.

    Empty list → empty string (no block injected). Each correction is
    rendered as: <hse-label> [- "<inspector note>"]. Note is truncated.
    """
    if not corrections:
        return ""
    hse_lookup = {h["slug"]: h for h in tax.get("hse_types", [])}
    lines = [
        "RECENT INSPECTOR CORRECTIONS in this batch (most recent first) — "
        "treat these as a strong prior. The inspector recently picked these "
        "labels for similar photos in the same upload session:",
    ]
    for c in corrections:
        slug = c.get("hse_type_slug") or ""
        h = hse_lookup.get(slug, {})
        label = h.get("label_en", slug) if h else slug
        if not label:
            continue
        note = (c.get("note") or "").strip()
        if note:
            note_short = note[:80] + ("..." if len(note) > 80 else "")
            lines.append(f'  - {label}  (note: "{note_short}")')
        else:
            lines.append(f"  - {label}")
    lines.append(
        "If the current photo is visually similar to recent corrections, "
        "lean toward the same label. Trust your own judgement when the "
        "evidence clearly differs."
    )
    return "\n".join(lines)


def _format_rag_block(neighbours: list[dict[str, Any]]) -> str:
    if not neighbours:
        return ""
    lines = [
        "RETRIEVED REFERENCE PHOTOS (most-to-least visually similar past photos from inspectors):",
    ]
    for i, n in enumerate(neighbours, 1):
        hse = n.get("hse_type_slug") or "?"
        loc = n.get("location_slug") or "?"
        dist = n.get("distance")
        dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "?"
        lines.append(
            f"  #{i} (cosine-dist {dist_str}): location={loc}  hse_type={hse}"
        )
    lines.append(
        "Use these neighbour labels as a strong prior — a photo visually similar to a "
        "neighbour often shares its labels. But trust your own visual judgement when the "
        "evidence in the current photo disagrees."
    )
    return "\n".join(lines)


# Back-compat alias; older call sites may import build_user_message.
build_user_message = build_user_message_anthropic


# ---------- io ----------

def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type). Detects real format from file bytes
    via Pillow so misnamed files (a PNG saved as foo.jpg, or a temp file with
    no extension) still get the correct Content-Type header. Anthropic strictly
    rejects mismatched headers with HTTP 400 "image was specified using
    image/jpeg media type but the image appears to be image/png".
    """
    data = path.read_bytes()
    media_type = "image/jpeg"  # safe default if everything below fails
    try:
        from PIL import Image
        import io as _io
        with Image.open(_io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
        media_type = {
            "JPEG": "image/jpeg",
            "PNG":  "image/png",
            "WEBP": "image/webp",
            "GIF":  "image/gif",
        }.get(fmt, "image/jpeg")
    except Exception:
        # Pillow couldn't open it; fall back to extension-based guess.
        suffix = path.suffix.lower().lstrip(".")
        media_type = {
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
            "png":  "image/png",
            "webp": "image/webp",
            "gif":  "image/gif",
        }.get(suffix, "image/jpeg")
    return base64.standard_b64encode(data).decode("ascii"), media_type


def _parse_response_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of the JSON object from a Sonnet response."""
    t = text.strip()
    # Strip markdown fences if the model ignored the 'no fences' instruction.
    if t.startswith("```"):
        t = t.strip("`")
        # drop language hint (e.g., "json\n")
        if "\n" in t:
            t = t.split("\n", 1)[1]
        if t.endswith("```"):
            t = t[:-3]
    # Find outermost braces
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    return json.loads(t[start : end + 1])


# ---------- core ----------

def _active_provider() -> str:
    """Return 'openrouter' or 'anthropic' based on which API key is present."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No provider API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
    )


def _classify_via_openrouter(
    image_b64: str, media_type: str, taxonomy_block: str, model_id: str,
    *, temperature: float | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Call OpenRouter's OpenAI-compatible chat-completions endpoint.
    Returns (parsed_json, input_tokens, output_tokens).
    """
    from openai import OpenAI
    headers = {}
    referer = os.environ.get("OPENROUTER_REFERER")
    title = os.environ.get("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
    )
    # max_tokens=3000 (was 500): Gemini 2.5 Pro and other "thinking"
    # models burn most of the output budget on internal reasoning before
    # emitting JSON, so 500 (and even 1500) truncates mid-string. 3000
    # is enough headroom even for verbose reasoning. Sonnet/Opus typically
    # use ~150-220 output tokens so the higher cap is free for them — we
    # only pay for what's actually emitted, not the cap.
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message_openai(image_b64, taxonomy_block, media_type)},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    parsed = _parse_response_json(text)
    usage = resp.usage
    return parsed, (usage.prompt_tokens or 0) if usage else 0, (usage.completion_tokens or 0) if usage else 0


def _classify_via_anthropic(
    image_b64: str, media_type: str, taxonomy_block: str, model_id: str,
    *, temperature: float | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Call Anthropic Messages API directly."""
    import anthropic
    c = anthropic.Anthropic()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 1500,   # see _classify_via_openrouter for rationale
        "system": _system_with_cache(),
        "messages": [{"role": "user",
                      "content": build_user_message_anthropic(image_b64, taxonomy_block, media_type)}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = c.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    parsed = _parse_response_json(text)
    in_tok = resp.usage.input_tokens if hasattr(resp, "usage") else 0
    out_tok = resp.usage.output_tokens if hasattr(resp, "usage") else 0
    return parsed, in_tok, out_tok


_FINE_TYPES_CACHE: dict[str, list[dict]] | None = None


def _load_fine_types() -> dict[str, list[dict]]:
    """Load and cache data/fine_hse_types_by_parent.json. Returns the
    `parents` map: parent_slug -> list of fine sub-type dicts. Each
    sub-type has slug, label_en, label_vn, primary_work_zone."""
    global _FINE_TYPES_CACHE
    if _FINE_TYPES_CACHE is not None:
        return _FINE_TYPES_CACHE
    p = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        _FINE_TYPES_CACHE = data.get("parents") or {}
    except Exception as e:  # noqa: BLE001
        log.warning("fine taxonomy load failed: %s", e)
        _FINE_TYPES_CACHE = {}
    return _FINE_TYPES_CACHE


_STAGE2_SYSTEM = """You are refining a construction safety classification.

A first-pass classifier already picked the broad category. Your job: pick the
SINGLE most-specific item from a closed list whose description matches the
photo. The list is short and was hand-authored by AECIS HSE inspectors —
every item is a specific real-world failure mode.

Rules:
1. Look at the photo. Look at the candidate list. Pick the ONE item whose
   description is the closest visible match.
2. If NO item clearly matches what you see, return null. Do not force a pick.
   The system will fall back to the broad category alone, which is fine.
3. The slug you return MUST be one of the ones listed below, verbatim.

Output ONE JSON object, no prose, no markdown fences:

{
  "fine_hse_type": {
    "slug": "<exact slug from the list, OR null>",
    "confidence": <float 0..1>
  },
  "fine_hse_type_alternatives": [
    {"slug": "<2nd-best slug>", "confidence": <float>},
    {"slug": "<3rd-best slug>", "confidence": <float>}
  ],
  "fine_rationale": "<one short sentence: what visible feature anchors your pick>"
}

If nothing fits, the JSON should still be valid:
{ "fine_hse_type": null, "fine_hse_type_alternatives": [], "fine_rationale": "..." }
"""


def _build_stage2_prompt(parent_slug: str, parent_label: str,
                         fine_items: list[dict]) -> str:
    """User-side prompt body for Stage 2. Lists the parent context + the
    candidate fine sub-types with EN/VN labels. Falls under 1k tokens for
    even the biggest parents (Lifting_unsafe with ~50 items)."""
    lines = [
        f"PARENT CATEGORY (already picked): {parent_slug} — {parent_label}",
        "",
        "CANDIDATE specific items under this category:",
    ]
    for it in fine_items:
        slug = it.get("slug") or ""
        lbl_en = it.get("label_en") or slug
        lbl_vn = it.get("label_vn") or ""
        if lbl_vn:
            lines.append(f"  - {slug}")
            lines.append(f"      EN: {lbl_en}")
            lines.append(f"      VN: {lbl_vn}")
        else:
            lines.append(f"  - {slug}: {lbl_en}")
    lines.append("")
    lines.append("Pick the single best match, or return null if none fits.")
    return "\n".join(lines)


def _classify_stage2(
    image_b64: str, media_type: str,
    parent_slug: str, parent_label: str, fine_items: list[dict],
    model_id: str, prov: str,
) -> tuple[dict[str, Any], int, int]:
    """Run a Stage 2 classify call. Reuses the same image bytes Stage 1
    used (no re-encode). Builds a narrow prompt with just the candidate
    fine sub-types. Returns (parsed_json, in_tokens, out_tokens) — same
    shape as Stage 1's transports.
    """
    user_text = _build_stage2_prompt(parent_slug, parent_label, fine_items)

    if prov == "openrouter":
        from openai import OpenAI
        headers = {}
        if os.environ.get("OPENROUTER_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
        if os.environ.get("OPENROUTER_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers or None,
        )
        # Reuse the OpenAI-shape image content from build_user_message_openai.
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            {"type": "text", "text": user_text},
        ]
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": _STAGE2_SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        text = resp.choices[0].message.content or ""
        parsed = _parse_response_json(text)
        usage = resp.usage
        return parsed, (usage.prompt_tokens or 0) if usage else 0, (usage.completion_tokens or 0) if usage else 0
    else:
        import anthropic
        c = anthropic.Anthropic()
        content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type, "data": image_b64,
            }},
            {"type": "text", "text": user_text},
        ]
        resp = c.messages.create(
            model=model_id,
            max_tokens=1000,
            system=_STAGE2_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        parsed = _parse_response_json(text)
        in_tok = resp.usage.input_tokens if hasattr(resp, "usage") else 0
        out_tok = resp.usage.output_tokens if hasattr(resp, "usage") else 0
        return parsed, in_tok, out_tok


# Stage 2 confidence below this -> emit fine_hse_type=None (i.e. "model
# isn't sure which specific item it is, just keep the parent class").
# Started at 0.4 but the n=50 audit showed Stage 2 was committing on 94%
# of photos and the committed picks were often valid-but-different from
# the dataset's specific GT slug. Bumped to 0.7 so Stage 2 only commits
# when it's genuinely confident — false fine labels are worse than no
# fine label (parent-only display is fine UX).
_STAGE2_CONFIDENCE_FLOOR = float(os.environ.get("STAGE2_CONFIDENCE_FLOOR", "0.7"))


def classify_image(
    image_path: Path,
    *,
    taxonomy: dict[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
    rag_neighbours: int | None = None,
    recent_corrections: list[dict[str, Any]] | None = None,
    samples: int | None = None,
    ensemble_models: list[str] | None = None,
    fine_grained: bool = False,
) -> Classification:
    """Send one image to the configured VLM and return a structured Classification.

    Provider is auto-detected from env vars (OpenRouter preferred if its key is
    set). Pass `provider="anthropic"` or `provider="openrouter"` to force.

    Photo-RAG: if the photo_embeddings table is populated, retrieves the k most
    visually similar reference photos via pgvector and includes their labels as
    a prior in the prompt. Set rag_neighbours=0 to disable retrieval.

    In-session learning: pass `recent_corrections` (a list of correction dicts
    from this batch, newest first) and they get appended to the prompt as
    in-context examples. The worker calls this with the last ~8 corrections
    in the same batch_id so the model can adapt to systematic patterns
    inside a review session.

    Self-consistency: pass `samples=N` (or set CLASSIFY_SAMPLES=N in env) to
    run the primary model N times at temperature 0.3 and majority-vote on
    hse_type. Final confidence is scaled by the ensemble agreement ratio.

    Cross-model ensemble: pass `ensemble_models=["anthropic/claude-sonnet-4.5"]`
    (or set CLASSIFY_ENSEMBLE_MODELS=...,...) to ALSO query those models once
    each, pooling their votes with the primary's. Use this when accuracy
    matters more than cost — e.g. final-pass review or retraining curation.
    """
    tax = taxonomy or load_taxonomy()
    prov = provider or _active_provider()

    if prov == "openrouter":
        model_id = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    else:
        model_id = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL

    k = rag_neighbours if rag_neighbours is not None else int(
        os.environ.get("RAG_NEIGHBOURS", RAG_NEIGHBOURS_DEFAULT)
    )

    # Build taxonomy block + optional RAG hints + optional in-session
    # corrections. Order matters: corrections come AFTER RAG so they're
    # the most-recent context in the model's view (humans tend to weight
    # later instructions more in long prompts).
    taxonomy_block = format_taxonomy_for_prompt(tax)
    neighbours = _retrieve_similar_labels(image_path, k) if k > 0 else []
    rag_block = _format_rag_block(neighbours)
    if rag_block:
        taxonomy_block = taxonomy_block + "\n\n" + rag_block
        log.info("RAG: retrieved %d neighbours for %s", len(neighbours), image_path.name)
    corrections_block = _format_corrections_block(recent_corrections or [], tax)
    if corrections_block:
        taxonomy_block = taxonomy_block + "\n\n" + corrections_block
        log.info("in-session: %d recent corrections injected for %s",
                 len(recent_corrections or []), image_path.name)

    image_b64, media_type = _encode_image(image_path)

    # ---- sampling / ensemble ----
    # samples: number of independent calls to make against `model_id`.
    # ensemble_models: additional models to also call once each (with their
    #                  own samples count of 1 — we vary across models, not
    #                  within each model). All results pool into a vote.
    samples_n = max(1, samples or int(os.environ.get("CLASSIFY_SAMPLES", "1")))
    extra_models_env = os.environ.get("CLASSIFY_ENSEMBLE_MODELS", "").strip()
    extra_models = ensemble_models or (
        [m.strip() for m in extra_models_env.split(",") if m.strip()]
    )

    def _maybe_run_stage2(cls: Classification, in_tok_total: int,
                          out_tok_total: int, used_model_label: str) -> Classification:
        """If fine_grained=True, run Stage 2 against the parent we just
        picked and attach fine_hse_type / fine_hse_type_alternatives to
        the Classification. No-op when fine_grained=False or when the
        parent has no fine sub-types defined. Stage 2 errors are non-fatal.
        """
        if not fine_grained:
            return cls
        parents_map = _load_fine_types()
        fine_items = parents_map.get(cls.hse_type.slug) or []
        if not fine_items:
            log.debug("Stage 2 skipped: no fine sub-types for %s", cls.hse_type.slug)
            return cls
        try:
            parsed_s2, in2, out2 = _classify_stage2(
                image_b64, media_type,
                cls.hse_type.slug, cls.hse_type.label_en,
                fine_items, model_id, prov,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Stage 2 failed (%s): %s", cls.hse_type.slug, str(e)[:120])
            return cls
        # Validate the slug against the closed list of fine items for
        # this parent. Hallucinations get nulled out.
        valid_slugs = {it["slug"] for it in fine_items}
        f = parsed_s2.get("fine_hse_type") if isinstance(parsed_s2, dict) else None
        if isinstance(f, dict):
            slug = (f.get("slug") or "").strip()
            conf = float(f.get("confidence") or 0.0)
            if slug in valid_slugs and conf >= _STAGE2_CONFIDENCE_FLOOR:
                lbl = next((it for it in fine_items if it["slug"] == slug), {})
                cls.fine_hse_type = AxisLabel(
                    slug=slug,
                    label_en=lbl.get("label_en", slug),
                    label_vn=lbl.get("label_vn", ""),
                    confidence=round(conf, 3),
                )
        # Alternatives — keep at most 2, validated.
        alts = parsed_s2.get("fine_hse_type_alternatives") if isinstance(parsed_s2, dict) else []
        if isinstance(alts, list):
            kept: list[AxisLabel] = []
            for a in alts[:5]:
                if not isinstance(a, dict):
                    continue
                s = (a.get("slug") or "").strip()
                if s and s in valid_slugs and (cls.fine_hse_type is None or s != cls.fine_hse_type.slug):
                    lbl = next((it for it in fine_items if it["slug"] == s), {})
                    kept.append(AxisLabel(
                        slug=s,
                        label_en=lbl.get("label_en", s),
                        label_vn=lbl.get("label_vn", ""),
                        confidence=round(float(a.get("confidence") or 0), 3),
                    ))
                if len(kept) >= 2:
                    break
            cls.fine_hse_type_alternatives = kept
        # Roll up token counts so the cost ticker / eval cost calc
        # captures the Stage 2 cost too.
        cls.input_tokens = in_tok_total + in2
        cls.output_tokens = out_tok_total + out2
        cls.model = f"{used_model_label}+stage2"
        # Bake fine results into raw_response too so /api/pending can
        # surface alternatives without needing a parallel column.
        if not isinstance(cls.raw_response, dict):
            cls.raw_response = {}
        if cls.fine_hse_type:
            cls.raw_response["fine_hse_type"] = asdict(cls.fine_hse_type)
        cls.raw_response["fine_hse_type_alternatives"] = [
            asdict(a) for a in cls.fine_hse_type_alternatives
        ]
        if isinstance(parsed_s2, dict) and parsed_s2.get("fine_rationale"):
            cls.raw_response["fine_rationale"] = parsed_s2["fine_rationale"]
        log.info("Stage 2: parent=%s -> fine=%s (conf=%.2f)",
                 cls.hse_type.slug,
                 cls.fine_hse_type.slug if cls.fine_hse_type else "<none>",
                 cls.fine_hse_type.confidence if cls.fine_hse_type else 0.0)
        return cls

    # Base case (cheapest path, == today's behaviour): one call, no extras.
    if samples_n == 1 and not extra_models:
        if prov == "openrouter":
            parsed, in_tok, out_tok = _classify_via_openrouter(
                image_b64, media_type, taxonomy_block, model_id)
        else:
            parsed, in_tok, out_tok = _classify_via_anthropic(
                image_b64, media_type, taxonomy_block, model_id)
        cls = _build_classification(parsed, tax, f"{prov}:{model_id}", in_tok, out_tok)
        return _maybe_run_stage2(cls, in_tok, out_tok, f"{prov}:{model_id}")

    # ---- multi-sample / ensemble path ----
    # Collect (parsed, model_id, in_tok, out_tok) tuples, then vote.
    runs: list[tuple[dict[str, Any], str, int, int]] = []

    def _call(model_id_: str, temp: float | None) -> None:
        prov_ = "openrouter" if model_id_.count("/") >= 1 else prov
        try:
            if prov_ == "openrouter":
                p, i, o = _classify_via_openrouter(
                    image_b64, media_type, taxonomy_block, model_id_,
                    temperature=temp,
                )
            else:
                p, i, o = _classify_via_anthropic(
                    image_b64, media_type, taxonomy_block, model_id_,
                    temperature=temp,
                )
            runs.append((p, f"{prov_}:{model_id_}", i, o))
        except Exception as e:  # noqa: BLE001
            log.warning("ensemble run failed (%s): %s", model_id_, str(e)[:120])

    # N samples on the primary model. Use temperature 0.3 to get a bit of
    # variation between samples (with temp=0 they'd be identical and there's
    # nothing to vote on). Only the primary model gets multi-sampled — extras
    # are run once each so we get cross-model diversity, not redundancy.
    primary_temp = 0.3 if samples_n > 1 else None
    for _ in range(samples_n):
        _call(model_id, primary_temp)

    for m in extra_models:
        _call(m, None)

    if not runs:
        raise RuntimeError("All ensemble runs failed")

    # ---- vote ----
    # Aggregate by hse_type slug. Weight = sum of confidences across votes
    # (so a 0.95-confident vote matters more than a 0.5-confident one).
    # Tiebreak by raw vote count, then by max confidence within that group.
    from collections import defaultdict
    weights: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    max_conf: dict[str, float] = defaultdict(float)
    parsed_by_slug: dict[str, dict[str, Any]] = {}
    for parsed_, _model, _i, _o in runs:
        slug = (parsed_.get("hse_type") or {}).get("slug") or ""
        conf = float((parsed_.get("hse_type") or {}).get("confidence") or 0.0)
        if not slug:
            continue
        weights[slug] += conf
        counts[slug] += 1
        if conf > max_conf[slug]:
            max_conf[slug] = conf
            # Keep the parsed JSON of the most-confident vote for THIS slug
            # so the rationale and alternatives in the final result come
            # from a coherent single response, not stitched together.
            parsed_by_slug[slug] = parsed_

    if not weights:
        # Every run failed to emit a usable hse_type — fall through to
        # the highest-confidence parsed object regardless of slug.
        winner_parsed = max(runs, key=lambda r: float((r[0].get("hse_type") or {}).get("confidence") or 0.0))[0]
    else:
        winner_slug = max(weights.items(), key=lambda kv: (kv[1], counts[kv[0]], max_conf[kv[0]]))[0]
        winner_parsed = parsed_by_slug[winner_slug]

    # Adjust the winner's confidence by the agreement ratio. If 5/5 voted
    # the same, leave confidence alone. If 3/5 voted the winner, scale
    # confidence down by 3/5. This is the most useful signal of a
    # downstream "needs review" — high model confidence + low ensemble
    # agreement means the inspector should look closely.
    n_total = sum(counts.values())
    if n_total > 1 and weights:
        winner_slug = max(weights.items(), key=lambda kv: (kv[1], counts[kv[0]], max_conf[kv[0]]))[0]
        agreement = counts[winner_slug] / n_total
        hse_block = winner_parsed.get("hse_type") or {}
        old_conf = float(hse_block.get("confidence") or 0.0)
        # Don't penalize hard — soft scaling so a mild disagreement still
        # registers but doesn't destroy useful confidence info.
        new_conf = old_conf * (0.5 + 0.5 * agreement)
        hse_block["confidence"] = round(new_conf, 3)
        winner_parsed["hse_type"] = hse_block

    total_in = sum(r[2] for r in runs)
    total_out = sum(r[3] for r in runs)
    model_label = f"ensemble[{n_total}]:{model_id}"
    log.info("ensemble: %d runs, winner=%s (agreement=%d/%d), tokens=%d/%d",
             n_total,
             (winner_parsed.get("hse_type") or {}).get("slug", "?"),
             counts.get(
                 (winner_parsed.get("hse_type") or {}).get("slug", ""),
                 0,
             ),
             n_total, total_in, total_out)
    cls = _build_classification(winner_parsed, tax, model_label, total_in, total_out)
    return _maybe_run_stage2(cls, total_in, total_out, model_label)


def _build_classification(
    parsed: dict[str, Any],
    tax: dict[str, Any],
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> Classification:
    """Validate + enrich the model's response against our taxonomy."""
    loc_slug = parsed.get("location", {}).get("slug", "")
    hse_slug = parsed.get("hse_type", {}).get("slug", "")
    loc_conf = float(parsed.get("location", {}).get("confidence", 0.0))
    hse_conf = float(parsed.get("hse_type", {}).get("confidence", 0.0))
    rationale = parsed.get("rationale", "")[:400]

    loc_lookup = {x["slug"]: x for x in tax["locations"]}
    hse_lookup = {x["slug"]: x for x in tax["hse_types"]}

    def _resolve(slug: str, conf: float, lookup: dict) -> AxisLabel:
        ref = lookup.get(slug) or {}
        return AxisLabel(
            slug=slug,
            label_en=ref.get("label_en", slug),
            label_vn=ref.get("label_vn", ""),
            confidence=conf,
        )

    def _alternatives(raw: list, primary_slug: str, lookup: dict) -> list[AxisLabel]:
        out: list[AxisLabel] = []
        seen = {primary_slug}
        for item in (raw or [])[:4]:   # cap defensively
            s = (item or {}).get("slug", "")
            if not s or s in seen:
                continue
            if s not in lookup:
                continue   # drop hallucinated slugs
            seen.add(s)
            c = float(item.get("confidence", 0.0))
            out.append(_resolve(s, c, lookup))
        return out[:2]   # keep at most 2 runners-up

    hse_alts = _alternatives(parsed.get("hse_type_alternatives", []), hse_slug, hse_lookup)
    loc_alts = _alternatives(parsed.get("location_alternatives", []), loc_slug, loc_lookup)

    return Classification(
        location=_resolve(loc_slug, loc_conf, loc_lookup),
        hse_type=_resolve(hse_slug, hse_conf, hse_lookup),
        hse_type_alternatives=hse_alts,
        location_alternatives=loc_alts,
        rationale=rationale,
        raw_response=parsed,
        model=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------- batch API ----------

def classify_batch_submit(
    image_paths: list[Path],
    *,
    custom_ids: list[str] | None = None,
    taxonomy: dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    """Submit a batch of images to Anthropic's Batch API.
    Returns the batch_id. Poll `classify_batch_results(batch_id)` later.
    """
    import anthropic
    from anthropic.types.messages import MessageCreateParamsNonStreaming, Request

    tax = taxonomy or load_taxonomy()
    model_id = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    taxonomy_block = format_taxonomy_for_prompt(tax)

    if custom_ids is None:
        custom_ids = [p.stem for p in image_paths]
    if len(custom_ids) != len(image_paths):
        raise ValueError("custom_ids length must match image_paths length")

    requests: list[Request] = []
    for cid, path in zip(custom_ids, image_paths):
        image_b64, media_type = _encode_image(path)
        requests.append(
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=model_id,
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": build_user_message(image_b64, taxonomy_block, media_type),
                    }],
                ),
            )
        )

    c = anthropic.Anthropic()
    batch = c.messages.batches.create(requests=requests)
    log.info("Submitted batch %s with %d requests", batch.id, len(requests))
    return batch.id


def classify_batch_results(
    batch_id: str,
    *,
    taxonomy: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Classification | str]:
    """Fetch results from a completed batch. Returns {custom_id: Classification|error_msg}."""
    import anthropic
    tax = taxonomy or load_taxonomy()
    model_id = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL

    c = anthropic.Anthropic()
    batch = c.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        return {"_status": batch.processing_status}

    out: dict[str, Classification | str] = {}
    for result in c.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            try:
                parsed = _parse_response_json(text)
                out[cid] = _build_classification(
                    parsed, tax, model_id,
                    msg.usage.input_tokens, msg.usage.output_tokens,
                )
            except Exception as e:  # noqa: BLE001
                out[cid] = f"parse_error: {e}"
        else:
            out[cid] = f"{result.result.type}: {result.result.error if hasattr(result.result, 'error') else 'unknown'}"
    return out


# ---------- CLI ----------

def main() -> int:
    # Force UTF-8 stdout on Windows so Vietnamese labels render cleanly.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--taxonomy", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format="%(levelname)-7s %(message)s")

    path = Path(args.image).expanduser().resolve()
    if not path.exists():
        print(f"Image not found: {path}", file=sys.stderr)
        return 1
    tax = load_taxonomy(Path(args.taxonomy)) if args.taxonomy else load_taxonomy()

    cls = classify_image(path, taxonomy=tax, model=args.model)
    print(json.dumps(cls.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
