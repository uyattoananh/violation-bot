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
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_PROMPT = REPO_ROOT / "prompts" / "classifier.md"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Photo-RAG: how many nearest neighbours to retrieve and how much to let them influence
# the prompt. Set RAG_NEIGHBOURS=0 to disable retrieval entirely.
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

    def __post_init__(self):
        if self.hse_type_alternatives is None:
            self.hse_type_alternatives = []
        if self.location_alternatives is None:
            self.location_alternatives = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": asdict(self.location),
            "hse_type": asdict(self.hse_type),
            "hse_type_alternatives": [asdict(a) for a in self.hse_type_alternatives],
            "location_alternatives": [asdict(a) for a in self.location_alternatives],
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
The HSE_TYPES vocabulary below contains 33 categories aligned with the
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
   - A WORK ACTIVITY in progress               → Welding_unsafe / Hot_work_hazard /
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
   - "Welder welding without sparks contained" → Welding_unsafe
     (NOT Hot_work_hazard — welding is more specific)
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
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message_openai(image_b64, taxonomy_block, media_type)},
        ],
    )
    text = resp.choices[0].message.content or ""
    parsed = _parse_response_json(text)
    usage = resp.usage
    return parsed, (usage.prompt_tokens or 0) if usage else 0, (usage.completion_tokens or 0) if usage else 0


def _classify_via_anthropic(
    image_b64: str, media_type: str, taxonomy_block: str, model_id: str,
) -> tuple[dict[str, Any], int, int]:
    """Call Anthropic Messages API directly."""
    import anthropic
    c = anthropic.Anthropic()
    resp = c.messages.create(
        model=model_id,
        max_tokens=500,
        system=_system_with_cache(),
        messages=[{"role": "user",
                   "content": build_user_message_anthropic(image_b64, taxonomy_block, media_type)}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    parsed = _parse_response_json(text)
    in_tok = resp.usage.input_tokens if hasattr(resp, "usage") else 0
    out_tok = resp.usage.output_tokens if hasattr(resp, "usage") else 0
    return parsed, in_tok, out_tok


def classify_image(
    image_path: Path,
    *,
    taxonomy: dict[str, Any] | None = None,
    model: str | None = None,
    provider: str | None = None,
    rag_neighbours: int | None = None,
) -> Classification:
    """Send one image to the configured VLM and return a structured Classification.

    Provider is auto-detected from env vars (OpenRouter preferred if its key is
    set). Pass `provider="anthropic"` or `provider="openrouter"` to force.

    Photo-RAG: if the photo_embeddings table is populated, retrieves the k most
    visually similar reference photos via pgvector and includes their labels as
    a prior in the prompt. Set rag_neighbours=0 to disable retrieval.
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

    # Build taxonomy block + optional RAG hints
    taxonomy_block = format_taxonomy_for_prompt(tax)
    neighbours = _retrieve_similar_labels(image_path, k) if k > 0 else []
    rag_block = _format_rag_block(neighbours)
    if rag_block:
        taxonomy_block = taxonomy_block + "\n\n" + rag_block
        log.info("RAG: retrieved %d neighbours for %s", len(neighbours), image_path.name)

    image_b64, media_type = _encode_image(image_path)

    if prov == "openrouter":
        parsed, in_tok, out_tok = _classify_via_openrouter(image_b64, media_type, taxonomy_block, model_id)
    else:
        parsed, in_tok, out_tok = _classify_via_anthropic(image_b64, media_type, taxonomy_block, model_id)

    return _build_classification(parsed, tax, f"{prov}:{model_id}", in_tok, out_tok)


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
