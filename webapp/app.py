"""Phase-1 construction-violation classifier webapp.

FastAPI + Jinja + HTMX. Single Python process. Deployable as a Docker image
to Fly.io, Railway, or any container host.

Endpoints:
  GET  /                  -> project picker (or redirect to /upload if only one)
  GET  /upload            -> batch photo uploader
  POST /api/upload        -> direct upload handler: stores to R2, creates photo
                             rows, enqueues classification jobs
  GET  /review            -> inspector review grid (suggestions + correct/confirm)
  POST /api/photos/{id}/confirm  -> inspector accepts AI suggestion
  POST /api/photos/{id}/correct  -> inspector corrects the AI suggestion
  GET  /healthz           -> liveness probe
  GET  /metrics           -> basic JSON counters (photos today, pending jobs)

Worker: a separate process (or `python -m webapp.worker`) polls
`classify_jobs` where status='pending', calls src.zero_shot.classify_image,
writes the result to `classifications`, marks the job done.

Env:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   -> DB access (server-side)
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
  ANTHROPIC_API_KEY                          -> forwarded to zero_shot module
  APP_ADMIN_PASSWORD                         -> HTTP Basic Auth gate (MVP)
  VIOLATION_TAXONOMY                         -> path override for taxonomy.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Load .env from the repo root (if present) before reading any env vars.
try:
    from dotenv import load_dotenv
    _repo_env = Path(__file__).resolve().parents[1] / ".env"
    if _repo_env.exists():
        load_dotenv(_repo_env)
except ImportError:
    pass

from fastapi import (
    FastAPI, File, Form, HTTPException, Request, UploadFile,
    status,
)
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse,
    StreamingResponse, PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Make `src.*` importable when running `uvicorn webapp.app:app`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.zero_shot import load_taxonomy, classify_image  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("webapp")


# ---------- config ----------

APP_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_ROOT / "templates"))
STATIC_DIR = APP_ROOT / "static"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY")
R2_BUCKET = _env("R2_BUCKET")
R2_ACCOUNT_ID = _env("R2_ACCOUNT_ID")

# Default tenant/project every upload is attributed to. No login required —
# the app operates in open-capture mode so photos can be contributed without
# friction and added to the RAG training set on confirm.
DEFAULT_TENANT_NAME = _env("DEFAULT_TENANT_NAME", "Public Demo")
DEFAULT_PROJECT_CODE = _env("DEFAULT_PROJECT_CODE", "PUBLIC")
DEFAULT_PROJECT_NAME = _env("DEFAULT_PROJECT_NAME", "Public uploads")

# Resolved once on startup to UUIDs; see the lifespan handler below.
DEFAULT_TENANT_ID: str | None = None
DEFAULT_PROJECT_ID: str | None = None


# ---------- data layer (thin wrappers around supabase-py) ----------

_sb_client: Any = None


def get_db() -> Any:
    """Return a supabase-py client. Lazy so the module imports without creds."""
    global _sb_client
    if _sb_client is not None:
        return _sb_client
    if not (SUPABASE_URL and SUPABASE_KEY):
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    from supabase import create_client
    _sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb_client


_r2_client: Any = None


def get_r2() -> Any:
    """Return a boto3 S3 client pointed at Cloudflare R2."""
    global _r2_client
    if _r2_client is not None:
        return _r2_client
    if not all([R2_ACCOUNT_ID, _env("R2_ACCESS_KEY_ID"), _env("R2_SECRET_ACCESS_KEY")]):
        raise RuntimeError("R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY must be set")
    import boto3
    _r2_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )
    return _r2_client


# ---------- lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global DEFAULT_TENANT_ID, DEFAULT_PROJECT_ID
    # Warm-load taxonomy so the first request is fast.
    try:
        tax = load_taxonomy()
        log.info("taxonomy loaded: %d locations, %d hse_types",
                 len(tax["locations"]), len(tax["hse_types"]))
        app.state.taxonomy = tax
    except Exception as e:  # noqa: BLE001
        log.error("failed to load taxonomy: %s", e)
        app.state.taxonomy = None

    # Ensure a default tenant + project exists. This is the everyone-uploads-here
    # target — no login, no per-user selection needed. Idempotent.
    try:
        db = get_db()
        tenants = (
            db.table("tenants").select("id, name")
              .eq("name", DEFAULT_TENANT_NAME).execute().data or []
        )
        if tenants:
            DEFAULT_TENANT_ID = tenants[0]["id"]
        else:
            DEFAULT_TENANT_ID = db.table("tenants").insert(
                {"name": DEFAULT_TENANT_NAME}
            ).execute().data[0]["id"]
            log.info("Created default tenant %s", DEFAULT_TENANT_ID)

        projects = (
            db.table("projects").select("id, code")
              .eq("tenant_id", DEFAULT_TENANT_ID).eq("code", DEFAULT_PROJECT_CODE)
              .execute().data or []
        )
        if projects:
            DEFAULT_PROJECT_ID = projects[0]["id"]
        else:
            DEFAULT_PROJECT_ID = db.table("projects").insert({
                "tenant_id": DEFAULT_TENANT_ID,
                "code": DEFAULT_PROJECT_CODE,
                "name": DEFAULT_PROJECT_NAME,
            }).execute().data[0]["id"]
            log.info("Created default project %s", DEFAULT_PROJECT_ID)
        log.info("Default routing: tenant=%s project=%s",
                 DEFAULT_TENANT_ID, DEFAULT_PROJECT_ID)
    except Exception as e:  # noqa: BLE001
        log.error("Failed to resolve default tenant/project: %s", e)
    yield


app = FastAPI(title="Construction Violation Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- helpers ----------

# Register HEIC opener so iPhone photos load. Optional — falls through if
# pillow-heif isn't installed (older deployments).
try:
    import pillow_heif as _pillow_heif  # noqa: E402
    _pillow_heif.register_heif_opener()
except Exception:  # noqa: BLE001
    pass


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _storage_key(tenant_id: str, project_id: str, sha: str, ext: str) -> str:
    # partitioned for listing sanity: <tenant>/<project>/<sha[:2]>/<sha>.<ext>
    return f"{tenant_id}/{project_id}/{sha[:2]}/{sha}{ext}"


# Formats Anthropic's image content block accepts directly. Anything outside
# this set (HEIC, BMP, TIFF, AVIF, etc.) we convert to JPEG before storing,
# so both the API call AND the in-browser thumbnail "just work".
_API_NATIVE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}


def _looks_like_supported_image(body: bytes) -> bool:
    """Cheap pre-flight: does this look like a Pillow-decodable image?
    Used by /api/upload to reject non-images at the boundary so they
    don't hang in 'Analyzing...' forever after the worker fails to
    classify them. Only does the cheap header/magic-bytes check —
    actual decode happens later in _normalize_image."""
    try:
        from PIL import Image
        import io as _io
        with Image.open(_io.BytesIO(body)) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


# Quality thresholds tuned for site photos. Lower = more permissive.
# Photos that don't pass are uploaded anyway but flagged with a quality
# warning so the user knows the AI may struggle.
_QUALITY_BLUR_THRESHOLD = 30.0      # variance-of-laplacian
_QUALITY_DARK_THRESHOLD = 30.0      # mean luminance 0..255 (very dark = hard to see)
_QUALITY_SMALL_PIXELS = 200 * 200   # photos smaller than this can't show detail


def _image_quality_check(body: bytes) -> dict | None:
    """Return a dict with quality issues, or None if the photo passes.
    Issue keys: blur (low variance-of-laplacian), dark (low mean luminance),
    tiny (image too small to show detail). Cheap to compute on a downscaled
    grayscale copy; runs in <50ms typical."""
    try:
        from PIL import Image
        import io as _io
        import numpy as np

        with Image.open(_io.BytesIO(body)) as img:
            w, h = img.size
            if w * h < _QUALITY_SMALL_PIXELS:
                return {"reason": "tiny", "width": w, "height": h}
            # Downscale to 256-wide grayscale for fast quality stats
            ratio = 256 / max(1, w)
            small = img.convert("L").resize((256, max(1, int(h * ratio))))
        arr = np.asarray(small, dtype=np.float32)
        mean_lum = float(arr.mean())
        if mean_lum < _QUALITY_DARK_THRESHOLD:
            return {"reason": "dark", "mean_luminance": round(mean_lum, 1)}
        # Variance of Laplacian as blur proxy: higher = sharper
        # 3x3 Laplacian kernel via numpy diff (cheap, no scipy/cv2)
        gy = np.diff(arr, axis=0, prepend=arr[:1])
        gx = np.diff(arr, axis=1, prepend=arr[:, :1])
        lap = gx[:-1, :-1] + gy[:-1, :-1] - 2 * arr[:-1, :-1]
        var_lap = float(lap.var())
        if var_lap < _QUALITY_BLUR_THRESHOLD:
            return {"reason": "blurry", "variance_of_laplacian": round(var_lap, 1)}
    except Exception as e:  # noqa: BLE001
        log.debug("image quality check failed: %s", e)
        return None
    return None


def _normalize_image(body: bytes, original_filename: str | None) -> tuple[bytes, str, str]:
    """Return (bytes, ext, content_type) — converted to JPEG if the input
    isn't already in an API-/browser-friendly format. Falls through with
    the original bytes if Pillow can't decode it (let the provider 4xx)."""
    import io as _io
    from PIL import Image

    try:
        with Image.open(_io.BytesIO(body)) as img:
            fmt = (img.format or "").upper()
            # Already supported — pass through unchanged.
            if fmt in _API_NATIVE_FORMATS:
                ext_map = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "WEBP": ".webp"}
                ct_map  = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
                return body, ext_map[fmt], ct_map[fmt]

            # Convert anything else (HEIC, BMP, TIFF, AVIF, ...) to JPEG.
            # Strip alpha (RGBA → RGB) so JPEG encoder doesn't error.
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            log.info("Converted %s → JPEG (%d → %d bytes)",
                     fmt or "?", len(body), len(buf.getvalue()))
            return buf.getvalue(), ".jpg", "image/jpeg"
    except Exception as e:  # noqa: BLE001
        log.warning("image normalize failed (%s) — passing original through: %s",
                    original_filename or "?", e)
        ext = (Path(original_filename or "").suffix.lower() or ".jpg")
        ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")
        return body, ext, ct


def _safe_corrections_insert(payload: dict) -> None:
    """Insert into the corrections table, stripping any column that the
    deployed schema doesn't yet have. Lets the webapp survive the window
    between code-deploy and SQL-migration without 500ing on every confirm."""
    db = get_db()
    p = dict(payload)   # don't mutate caller's dict
    for _ in range(4):  # at most one retry per optional column
        try:
            db.table("corrections").insert(p).execute()
            return
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            stripped = False
            for col in ("fine_hse_type_slug", "batch_id"):
                if col in msg and col in p:
                    p.pop(col, None)
                    log.warning("corrections.%s column missing — degraded insert", col)
                    stripped = True
                    break
            if not stripped:
                raise


def _upsert_embedding_for_correction(photo_id: str, hse_slug: str | None,
                                     loc_slug: str | None,
                                     fine_hse_type_slug: str | None = None) -> None:
    """Feedback loop: when an inspector confirms or corrects a photo,
    compute its CLIP embedding (if not already in pgvector) and upsert it
    with the inspector-verified labels. This makes future retrievals
    benefit from human-verified ground truth.

    fine_hse_type_slug, if provided, gets stored alongside so the AECIS
    canonical sub-type is part of the retrieval prior too.

    Called best-effort — failure is logged but does not break the user's
    correction flow.
    """
    import tempfile
    try:
        from src.embeddings import embed_image
    except Exception as e:  # noqa: BLE001
        log.warning("feedback-loop: embeddings unavailable: %s", e)
        return
    db = get_db()
    photo = db.table("photos").select("*").eq("id", photo_id).execute().data
    if not photo:
        return
    p = photo[0]
    sha = p.get("sha256")
    if not sha:
        return
    # Fetch photo bytes from R2 -> tempfile -> CLIP
    try:
        r2 = get_r2()
        resp = r2.get_object(Bucket=R2_BUCKET, Key=p["storage_key"])
        body = resp["Body"].read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(body)
            tmp_path = Path(tf.name)
        try:
            vec = embed_image(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.warning("feedback-loop: embed failed for photo=%s: %s", photo_id, e)
        return
    # Upsert into photo_embeddings. Correction labels take precedence over
    # any existing scraped labels, so re-confirming or re-correcting the same
    # photo always updates the index.
    try:
        upsert_payload = {
            "sha256": sha,
            "hse_type_slug": hse_slug,
            "location_slug": loc_slug,
            "fine_hse_type_slug": fine_hse_type_slug,
            "label_source": "manual",
            "project_code": "INSPECTOR",
            "issue_id": photo_id,
            "source_path": f"inspector_corrected/{photo_id}",
            "embedding": vec.tolist(),
            "tenant_id": p.get("tenant_id"),
        }
        try:
            db.table("photo_embeddings").upsert(upsert_payload, on_conflict="sha256").execute()
        except Exception as e2:  # noqa: BLE001
            if "fine_hse_type_slug" in str(e2):
                upsert_payload.pop("fine_hse_type_slug", None)
                db.table("photo_embeddings").upsert(upsert_payload, on_conflict="sha256").execute()
                log.warning("fine_hse_type_slug column missing on photo_embeddings — degraded upsert")
            else:
                raise
        log.info("feedback-loop: upserted embedding for photo=%s hse=%s loc=%s fine=%s",
                 photo_id, hse_slug, loc_slug, fine_hse_type_slug)
    except Exception as e:  # noqa: BLE001
        log.warning("feedback-loop: upsert failed for photo=%s: %s", photo_id, e)


# ---------- routes ----------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    tax = app.state.taxonomy or load_taxonomy()
    # Load fine-types lookup once and cache on app.state
    if not hasattr(app.state, "fine_types"):
        fine_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
        try:
            app.state.fine_types = json.loads(fine_path.read_text(encoding="utf-8")).get("parents", {})
        except Exception:  # noqa: BLE001
            app.state.fine_types = {}
    return TEMPLATES.TemplateResponse(
        request, "index.html",
        {
            "taxonomy": tax,
            "fine_types": app.state.fine_types,
            "model": os.environ.get("OPENROUTER_MODEL", "?"),
        },
    )


# Legacy paths — redirect everything to /
@app.get("/upload", include_in_schema=False)
def _upload_redirect():
    return RedirectResponse("/", status_code=308)


@app.get("/review", include_in_schema=False)
def _review_redirect():
    return RedirectResponse("/", status_code=308)


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    tenant_id: str | None = Form(None),
    project_id: str | None = Form(None),
    batch_id: str | None = Form(None),
    batch_label: str | None = Form(None),
):
    """Upload one or more photos. The frontend generates a UUID per
    upload session and passes it as `batch_id`; all photos in the same
    session share that ID. Downstream queries (pending / export) filter
    by batch so users only see / download the current job's photos."""
    tenant_id = tenant_id or DEFAULT_TENANT_ID
    project_id = project_id or DEFAULT_PROJECT_ID
    if not (tenant_id and project_id):
        raise HTTPException(500, "Server has no default tenant/project configured")
    # Server-side fallback: if the client didn't send a batch id, OR sent
    # something that's not a valid UUID, mint a fresh one. Keeps a malformed
    # client / stale URL / direct API-poke from blowing up the upload.
    def _coerce_uuid(s: str | None) -> str:
        try:
            return str(uuid.UUID(s)) if s else str(uuid.uuid4())
        except Exception:  # noqa: BLE001
            log.warning("upload received non-UUID batch_id=%r — minting new one", s)
            return str(uuid.uuid4())
    batch_id = _coerce_uuid(batch_id)

    db = get_db()
    r2 = get_r2()
    created = []
    rejected = []
    warnings: list[dict] = []  # per-photo quality flags (uploaded but suspect)
    for f in files:
        raw = await f.read()
        # Reject non-image / 0-byte payloads at the upload boundary. If we
        # accept them, they sit in "Analyzing..." forever because the
        # classifier can't process them — bad UX.
        if not raw or len(raw) < 100:   # any real photo > 100 bytes
            rejected.append({"filename": f.filename, "reason": "empty_or_too_small"})
            continue
        if not _looks_like_supported_image(raw):
            rejected.append({"filename": f.filename, "reason": "not_an_image"})
            continue
        # Normalize first — convert HEIC/BMP/TIFF/etc → JPEG so the rest of
        # the pipeline (R2 thumbnail in browser + Anthropic API call) doesn't
        # need to care about the source format. sha is hashed on the FINAL
        # bytes so dedup matches what's actually stored.
        body, ext, content_type = _normalize_image(raw, f.filename)
        sha = _hash_bytes(body)
        key = _storage_key(tenant_id, project_id, sha, ext)

        # Cheap quality pre-flight on the normalized bytes. We DO NOT reject
        # here — a blurry / dark / tiny photo can still be a real violation
        # and the AI sometimes nails it. We just surface a warning so the
        # user knows the result might be unreliable.
        quality = _image_quality_check(body)

        existing = (
            db.table("photos").select("id").eq("tenant_id", tenant_id)
              .eq("sha256", sha).execute()
        )
        if existing.data:
            # Dedup hit: re-attach the existing photo to the CURRENT batch
            # so the user sees it in their working view. Previous batches
            # lose this photo, which matches the "current job" mental model.
            try:
                db.table("photos").update({
                    "batch_id": batch_id,
                    "batch_label": batch_label,
                }).eq("id", existing.data[0]["id"]).execute()
            except Exception as e:  # noqa: BLE001
                if "batch_id" not in str(e):
                    log.warning("dedup batch update failed: %s", e)
            created.append({"id": existing.data[0]["id"], "dedup": True})
            if quality:
                warnings.append({"filename": f.filename, "id": existing.data[0]["id"], **quality})
            continue

        r2.put_object(
            Bucket=R2_BUCKET, Key=key, Body=body,
            ContentType=content_type,
        )
        photo_payload = {
            "tenant_id": tenant_id,
            "project_id": project_id,
            "storage_key": key,
            "storage_bucket": R2_BUCKET,
            "sha256": sha,
            "original_filename": f.filename,
            "bytes": len(body),
            "batch_id": batch_id,
            "batch_label": batch_label,
        }
        try:
            photo_row = db.table("photos").insert(photo_payload).execute().data[0]
        except Exception as e:  # noqa: BLE001
            # Fallback if the batch columns haven't been ALTER-TABLE'd yet.
            if "batch_id" in str(e) or "batch_label" in str(e):
                photo_payload.pop("batch_id", None)
                photo_payload.pop("batch_label", None)
                photo_row = db.table("photos").insert(photo_payload).execute().data[0]
                log.warning("photos.batch_id column missing — degraded upload path used")
            else:
                raise
        db.table("classify_jobs").insert({"photo_id": photo_row["id"]}).execute()
        created.append({"id": photo_row["id"], "dedup": False})
        if quality:
            warnings.append({"filename": f.filename, "id": photo_row["id"], **quality})

    return {
        "uploaded": created,
        "rejected": rejected,
        "warnings": warnings,
        "count": len(created),
        "batch_id": batch_id,
    }


@app.get("/api/batches")
def api_batches(limit: int = 100):
    """List the batches in the default tenant, with stats per batch.
    Used by the landing-page batches list. NULL-batch photos (e.g. legacy
    auto-seeded data) are excluded — they aren't user-created batches.

    Returns batches sorted newest-first by latest_uploaded_at.
    """
    if not DEFAULT_TENANT_ID:
        return {"batches": []}
    db = get_db()
    # Pull every photo's batch_id + uploaded_at; aggregate in Python (Supabase's
    # PostgREST doesn't expose GROUP BY directly without RPCs).
    try:
        rows = (
            db.table("photos")
              .select("id, batch_id, batch_label, uploaded_at")
              .eq("tenant_id", DEFAULT_TENANT_ID)
              .not_.is_("batch_id", "null")
              .order("uploaded_at", desc=True)
              .limit(5000)
              .execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        if "batch_id" in str(e):
            log.warning("photos.batch_id missing — /api/batches returns empty")
            return {"batches": []}
        raise
    if not rows:
        return {"batches": []}

    # Group by batch_id, capturing label (latest non-empty wins) + photo count + first/latest uploaded_at
    by_batch: dict[str, dict] = {}
    photo_ids_by_batch: dict[str, list[str]] = {}
    for r in rows:
        bid = r["batch_id"]
        if bid not in by_batch:
            by_batch[bid] = {
                "batch_id": bid,
                "label": r.get("batch_label") or "",
                "photo_count": 0,
                "latest_uploaded_at": r["uploaded_at"],
                "earliest_uploaded_at": r["uploaded_at"],
            }
            photo_ids_by_batch[bid] = []
        b = by_batch[bid]
        b["photo_count"] += 1
        if r.get("batch_label"):
            b["label"] = r["batch_label"]
        if r["uploaded_at"] > b["latest_uploaded_at"]:
            b["latest_uploaded_at"] = r["uploaded_at"]
        if r["uploaded_at"] < b["earliest_uploaded_at"]:
            b["earliest_uploaded_at"] = r["uploaded_at"]
        photo_ids_by_batch[bid].append(r["id"])

    # Count reviewed photos per batch — corrections scoped to that batch_id
    for bid, photo_ids in photo_ids_by_batch.items():
        reviewed_count = 0
        for i in range(0, len(photo_ids), 100):
            chunk = photo_ids[i:i + 100]
            try:
                cs = (
                    db.table("corrections").select("photo_id", count="exact")
                      .in_("photo_id", chunk).eq("batch_id", bid)
                      .limit(1).execute()
                )
                reviewed_count += cs.count or 0
            except Exception:  # noqa: BLE001
                pass
        by_batch[bid]["reviewed_count"] = reviewed_count

    batches = sorted(by_batch.values(), key=lambda b: b["latest_uploaded_at"], reverse=True)
    return {"batches": batches[:limit]}


@app.post("/api/batches/{batch_id}/clear-reviews")
def clear_batch_reviews(batch_id: str):
    """Delete every correction (confirm + correct) the user made in this
    batch, so they can re-review from scratch. Photos remain, AI predictions
    remain, only the inspector's choices are wiped — for the current batch
    only."""
    db = get_db()
    try:
        db.table("corrections").delete().eq("batch_id", batch_id).execute()
    except Exception as e:  # noqa: BLE001
        if "batch_id" in str(e):
            raise HTTPException(500, "batch_id column not yet migrated")
        raise
    return {"ok": True, "batch_id": batch_id}


@app.post("/api/photos/{photo_id}/retry")
def retry_classify(photo_id: str):
    """Reset a failed (or stuck) classify_jobs row back to 'pending' so the
    worker picks it up again. Surfaced from the UI when an /api/pending
    response shows classify_status='error'."""
    db = get_db()
    try:
        db.table("classify_jobs").update({
            "status": "pending",
            "error": None,
        }).eq("photo_id", photo_id).execute()
    except Exception as e:  # noqa: BLE001
        log.warning("retry_classify: %s", e)
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.get("/api/pending")
def api_pending(limit: int = 40, batch_id: str | None = None):
    """Return the most recent photos + their current classification.

    If `batch_id` is provided, only photos in that batch are returned —
    this is what the frontend always passes so users see only their
    current upload session, not the full tenant history.

    Poll this from the frontend every few seconds to update cards as the
    worker classifies them.
    """
    db = get_db()
    q = db.table("photos").select("*")
    if batch_id:
        q = q.eq("batch_id", batch_id)
    try:
        photos = q.order("uploaded_at", desc=True).limit(limit).execute().data or []
    except Exception as e:  # noqa: BLE001
        # If the batch_id column isn't migrated yet, retry without the filter.
        if batch_id and "batch_id" in str(e):
            log.warning("photos.batch_id column missing — falling back to unfiltered pending")
            photos = (db.table("photos").select("*")
                        .order("uploaded_at", desc=True).limit(limit)
                        .execute().data or [])
        else:
            raise
    if not photos:
        return {"photos": [], "training_set_size": _training_set_size()}

    photo_ids = [p["id"] for p in photos]
    cls_rows = (
        db.table("classifications").select("*")
          .in_("photo_id", photo_ids).eq("is_current", True)
          .execute().data or []
    )
    cls_by_photo = {c["photo_id"]: c for c in cls_rows}

    # Pull classify_jobs status per photo so the UI can show 'failed — retry'
    # when the worker hit an error (otherwise the card sits in 'Analyzing...'
    # forever with no feedback).
    job_status_by_photo: dict[str, dict] = {}
    try:
        for i in range(0, len(photo_ids), 100):
            chunk = photo_ids[i:i + 100]
            jobs = (
                db.table("classify_jobs").select("photo_id, status, error, attempt")
                  .in_("photo_id", chunk).execute().data or []
            )
            for j in jobs:
                # Latest job per photo (only one should exist normally)
                job_status_by_photo[j["photo_id"]] = j
    except Exception as e:  # noqa: BLE001
        log.warning("classify_jobs lookup failed: %s", e)
    # Also fetch the latest correction action per photo so the UI can mark
    # photos as already-reviewed AND show the inspector's final label
    # (which differs from the AI prediction when action='correct').
    # Per-batch review state: only count corrections that belong to the
    # current batch. Without this filter, a photo whose sha256 matched an
    # auto-seeded one would inherit the auto-seed's "confirm" correction
    # and appear as already-reviewed in the user's fresh batch — skipping
    # the inspector workflow entirely.
    q = db.table("corrections").select("*").in_("photo_id", photo_ids)
    if batch_id:
        q = q.eq("batch_id", batch_id)
    try:
        corr_rows = q.order("created_at", desc=True).execute().data or []
    except Exception as e:  # noqa: BLE001
        # Pre-migration: fall back to unfiltered (legacy behaviour). After
        # the SQL is applied this branch never fires.
        if batch_id and "batch_id" in str(e):
            corr_rows = (
                db.table("corrections").select("*")
                  .in_("photo_id", photo_ids)
                  .order("created_at", desc=True)
                  .execute().data or []
            )
        else:
            raise
    latest_correction: dict[str, dict] = {}
    for c in corr_rows:
        latest_correction.setdefault(c["photo_id"], c)

    r2 = get_r2()
    out: list[dict] = []
    for p in photos:
        thumb = r2.generate_presigned_url(
            "get_object",
            Params={"Bucket": R2_BUCKET, "Key": p["storage_key"]},
            ExpiresIn=60 * 30,
        )
        cls = cls_by_photo.get(p["id"])
        alts_hse: list[dict] = []
        alts_loc: list[dict] = []
        if cls:
            raw = cls.get("raw_response") or {}
            alts_hse = raw.get("hse_type_alternatives") or []
            alts_loc = raw.get("location_alternatives") or []
        corr = latest_correction.get(p["id"]) or {}
        job = job_status_by_photo.get(p["id"]) or {}
        out.append({
            "id": p["id"],
            "thumb_url": thumb,
            "original_filename": p.get("original_filename"),
            "uploaded_at": p.get("uploaded_at"),
            # Include batch_id (may be None on pre-migration photos) so the
            # frontend can also filter client-side, defense-in-depth.
            "batch_id": p.get("batch_id"),
            # Classify-job state: 'pending' / 'in_progress' / 'done' / 'error'.
            # UI surfaces 'error' as a retry-able card.
            "classify_status": job.get("status"),
            "classify_error": (job.get("error") or "")[:500],
            "classification": (
                {
                    "location_slug": cls["location_slug"],
                    "hse_type_slug": cls["hse_type_slug"],
                    "location_confidence": cls.get("location_confidence") or 0,
                    "hse_type_confidence": cls.get("hse_type_confidence") or 0,
                    "rationale": cls.get("rationale", ""),
                    "hse_type_alternatives": alts_hse,
                    "location_alternatives": alts_loc,
                    "model": cls.get("model"),
                } if cls else None
            ),
            "reviewed": bool(corr),
            "review_action": corr.get("action"),
            # Final labels — what the inspector actually saved (could differ
            # from the AI prediction on a correction). UI shows these on
            # reviewed cards so the user sees their own choice, not the AI's.
            "final_hse_type_slug": corr.get("hse_type_slug") or (cls.get("hse_type_slug") if cls else None),
            "final_location_slug": corr.get("location_slug") or (cls.get("location_slug") if cls else None),
            # Fine-grained AECIS sub-type the inspector picked (optional;
            # null when the inspector skipped the refinement step).
            "final_fine_hse_type_slug": corr.get("fine_hse_type_slug"),
        })
    return {
        "photos": out,
        "training_set_size": _training_set_size(),
    }


def _training_set_size() -> int:
    """Count of confirmed + scraped embeddings in the retrieval index."""
    try:
        r = (
            get_db().table("photo_embeddings")
              .select("sha256", count="exact").limit(1).execute()
        )
        return r.count or 0
    except Exception:  # noqa: BLE001
        return 0


@app.post("/api/photos/{photo_id}/confirm")
def confirm(photo_id: str, fine_hse_type_slug: str | None = Form(None)):
    """Inspector confirms the AI's prediction. Optionally also picks a
    fine-grained AECIS sub-type to refine within the predicted parent."""
    db = get_db()
    cls = (
        db.table("classifications")
          .select("*").eq("photo_id", photo_id).eq("is_current", True)
          .execute().data
    )
    if not cls:
        raise HTTPException(400, "no current classification")
    c = cls[0]
    if fine_hse_type_slug == "":
        fine_hse_type_slug = None
    # Pull the photo's current batch_id so the correction is tied to it.
    # Stale corrections from previous batches (e.g. auto_seed) won't count
    # toward "this photo is reviewed in the current batch".
    batch_id_for_corr = None
    try:
        photo_rows = db.table("photos").select("batch_id").eq("id", photo_id).execute().data
        if photo_rows:
            batch_id_for_corr = photo_rows[0].get("batch_id")
    except Exception:  # noqa: BLE001
        pass
    payload = {
        "photo_id": photo_id,
        "classification_id": c["id"],
        "action": "confirm",
        "location_slug": c["location_slug"],
        "hse_type_slug": c["hse_type_slug"],
        "fine_hse_type_slug": fine_hse_type_slug,
        "batch_id": batch_id_for_corr,
    }
    _safe_corrections_insert(payload)
    # Feedback loop: confirmations are high-quality labels — add to retrieval index
    _upsert_embedding_for_correction(photo_id, c["hse_type_slug"], c["location_slug"],
                                     fine_hse_type_slug=fine_hse_type_slug)
    return {"ok": True}


@app.post("/api/photos/{photo_id}/correct")
def correct(
    photo_id: str,
    location_slug: str = Form(...),
    hse_type_slug: str = Form(...),
    fine_hse_type_slug: str | None = Form(None),
    note: str | None = Form(None),
):
    """Inspector correction. fine_hse_type_slug is the optional AECIS-canonical
    sub-type they picked from the dropdown filtered to children of the chosen
    coarse hse_type_slug. Empty string from the form is normalised to None."""
    db = get_db()
    cls = (
        db.table("classifications")
          .select("*").eq("photo_id", photo_id).eq("is_current", True)
          .execute().data
    )
    cid = cls[0]["id"] if cls else None
    if fine_hse_type_slug == "":
        fine_hse_type_slug = None
    # Tie the correction to the photo's current batch (see confirm() above
    # for why — keeps stale per-photo reviews from leaking across batches).
    batch_id_for_corr = None
    try:
        photo_rows = db.table("photos").select("batch_id").eq("id", photo_id).execute().data
        if photo_rows:
            batch_id_for_corr = photo_rows[0].get("batch_id")
    except Exception:  # noqa: BLE001
        pass
    payload = {
        "photo_id": photo_id,
        "classification_id": cid,
        "action": "correct",
        "location_slug": location_slug,
        "hse_type_slug": hse_type_slug,
        "fine_hse_type_slug": fine_hse_type_slug,
        "batch_id": batch_id_for_corr,
        "note": note,
    }
    _safe_corrections_insert(payload)
    # Feedback loop: inspector-corrected label is high-quality — add to retrieval index
    _upsert_embedding_for_correction(photo_id, hse_type_slug, location_slug,
                                     fine_hse_type_slug=fine_hse_type_slug)
    return {"ok": True}


@app.delete("/api/photos/{photo_id}")
def delete_photo(photo_id: str):
    """Hard-delete a photo plus all its child rows and the R2 object.
    Allowed in any state — pending classification, predicted, or reviewed.

    Cascade order matters because of FK constraints in some Supabase setups:
       classify_jobs → corrections → classifications → photo_embeddings
       → photos → R2 object

    photo_embeddings is keyed on sha256, not photo_id. If a different photo
    in the same tenant happens to share the sha (literal duplicate bytes),
    the embedding stays — it's the AI's training signal, still useful.
    """
    db = get_db()
    rows = db.table("photos").select("*").eq("id", photo_id).execute().data or []
    if not rows:
        raise HTTPException(404, "photo not found")
    p = rows[0]

    # Cascade child rows first.
    for tbl in ("classify_jobs", "corrections", "classifications"):
        try:
            db.table(tbl).delete().eq("photo_id", photo_id).execute()
        except Exception as e:  # noqa: BLE001
            log.warning("delete from %s failed for photo=%s: %s", tbl, photo_id, e)

    # Drop the embedding only if no other photo refers to this sha.
    try:
        sha = p.get("sha256")
        if sha:
            others = (
                db.table("photos").select("id").eq("sha256", sha)
                  .neq("id", photo_id).limit(1).execute().data or []
            )
            if not others:
                db.table("photo_embeddings").delete().eq("sha256", sha).execute()
    except Exception as e:  # noqa: BLE001
        log.warning("photo_embeddings delete failed: %s", e)

    # Drop the photos row.
    try:
        db.table("photos").delete().eq("id", photo_id).execute()
    except Exception as e:  # noqa: BLE001
        log.error("photos delete failed for %s: %s", photo_id, e)
        raise HTTPException(500, f"photos row delete failed: {e}")

    # Drop the R2 object (best-effort — photo row is already gone).
    try:
        get_r2().delete_object(Bucket=R2_BUCKET, Key=p["storage_key"])
    except Exception as e:  # noqa: BLE001
        log.warning("R2 object delete failed for %s: %s", photo_id, e)

    return {"ok": True, "deleted_photo_id": photo_id}


# ---------- export ----------

def _slug_safe(s: str | None) -> str:
    """Sanitize for use in a filename: keep alnum/_/-, replace others with _."""
    if not s:
        return "unlabeled"
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)[:60]


def _collect_export_rows(tenant_id: str, limit: int = 5000,
                         batch_id: str | None = None) -> list[dict]:
    """Pull every photo for a tenant (optionally scoped to one batch)
    joined with its current classification, latest correction, and the
    original filename. Used by all 3 export formats. Filtering by
    batch_id is the default in the UI so users only download the photos
    in their current upload session, not the full tenant history."""
    db = get_db()
    q = db.table("photos").select("*").eq("tenant_id", tenant_id)
    if batch_id:
        q = q.eq("batch_id", batch_id)
    try:
        photos = q.order("uploaded_at", desc=True).limit(limit).execute().data or []
    except Exception as e:  # noqa: BLE001
        if batch_id and "batch_id" in str(e):
            log.warning("photos.batch_id missing — degraded export collected unfiltered")
            photos = (db.table("photos").select("*")
                        .eq("tenant_id", tenant_id)
                        .order("uploaded_at", desc=True).limit(limit)
                        .execute().data or [])
        else:
            raise
    if not photos:
        return []

    photo_ids = [p["id"] for p in photos]

    # Chunk IN-queries to stay under PostgREST URL limits (~100 IDs/chunk safe)
    def _fetch_chunked(table: str, select_cols: str, extra_filter=None):
        out = []
        for i in range(0, len(photo_ids), 100):
            chunk = photo_ids[i:i + 100]
            q = db.table(table).select(select_cols).in_("photo_id", chunk)
            if extra_filter:
                q = extra_filter(q)
            out.extend(q.execute().data or [])
        return out

    cls_rows = _fetch_chunked("classifications", "*",
                              lambda q: q.eq("is_current", True))
    cls_by_photo = {c["photo_id"]: c for c in cls_rows}

    # SELECT * so missing fine_hse_type_slug column doesn't 400 the export.
    # When filtering by batch, ALSO scope the corrections lookup so a stale
    # auto-seeded "confirm" doesn't make a fresh-batch upload appear as
    # already-reviewed. (Same fix as in api_pending.)
    def _corrections_filter(q):
        q = q.order("created_at", desc=True)
        if batch_id:
            q = q.eq("batch_id", batch_id)
        return q
    try:
        corr_rows = _fetch_chunked("corrections", "*", _corrections_filter)
    except Exception as e:  # noqa: BLE001
        if batch_id and "batch_id" in str(e):
            corr_rows = _fetch_chunked("corrections", "*",
                                       lambda q: q.order("created_at", desc=True))
        else:
            raise
    latest_correction: dict[str, dict] = {}
    for c in corr_rows:
        latest_correction.setdefault(c["photo_id"], c)

    rows: list[dict] = []
    for p in photos:
        cls = cls_by_photo.get(p["id"]) or {}
        corr = latest_correction.get(p["id"])
        ai_hse = cls.get("hse_type_slug")
        ai_loc = cls.get("location_slug")
        # Final = corrected if present, else AI's pick
        final_hse = (corr or {}).get("hse_type_slug") or ai_hse
        final_loc = (corr or {}).get("location_slug") or ai_loc
        rows.append({
            "id": p["id"],
            "sha256": p.get("sha256"),
            "original_filename": p.get("original_filename"),
            "uploaded_at": p.get("uploaded_at"),
            "storage_key": p["storage_key"],
            "ai_hse_type_slug": ai_hse,
            "ai_location_slug": ai_loc,
            "ai_hse_confidence": cls.get("hse_type_confidence") or 0,
            "ai_location_confidence": cls.get("location_confidence") or 0,
            "ai_rationale": cls.get("rationale", ""),
            "ai_model": cls.get("model", ""),
            "reviewed": bool(corr),
            "review_action": (corr or {}).get("action"),
            "review_note": (corr or {}).get("note"),
            "reviewed_at": (corr or {}).get("created_at"),
            "final_hse_type_slug": final_hse,
            "final_location_slug": final_loc,
            # Fine-grained AECIS-canonical sub-type the inspector picked
            # (optional). Empty string means "no refinement chosen".
            "final_fine_hse_type_slug": (corr or {}).get("fine_hse_type_slug") or "",
        })
    return rows


def _enrich_with_labels(rows: list[dict], tax: dict) -> None:
    """Add label_en / label_vn columns from the taxonomy in place.
    Also resolves the fine_hse_type_slug against data/fine_hse_types_by_parent.json
    so the export carries human-readable AECIS sub-type names."""
    import json as _json
    hse_lookup = {h["slug"]: h for h in tax["hse_types"]}
    loc_lookup = {l["slug"]: l for l in tax["locations"]}

    # Build a flat fine-slug lookup once
    fine_lookup: dict[str, dict] = {}
    fine_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    if fine_path.exists():
        try:
            fine_data = _json.loads(fine_path.read_text(encoding="utf-8"))
            for parent, items in (fine_data.get("parents") or {}).items():
                for it in items:
                    fine_lookup[it["slug"]] = it
        except Exception:  # noqa: BLE001
            pass

    for r in rows:
        for axis, lookup in (("hse_type", hse_lookup), ("location", loc_lookup)):
            for prefix in ("ai_", "final_"):
                slug = r.get(f"{prefix}{axis}_slug")
                lbl = lookup.get(slug or "") or {}
                r[f"{prefix}{axis}_label_en"] = lbl.get("label_en", slug or "")
                r[f"{prefix}{axis}_label_vn"] = lbl.get("label_vn", "")
        # Fine-grained AECIS sub-type labels
        fine_slug = r.get("final_fine_hse_type_slug") or ""
        fine_lbl = fine_lookup.get(fine_slug) or {}
        r["final_fine_hse_type_label_en"] = fine_lbl.get("label_en", fine_slug)
        r["final_fine_hse_type_label_vn"] = fine_lbl.get("label_vn", "")


@app.get("/api/export/csv")
def export_csv(limit: int = 5000, batch_id: str | None = None):
    """Stream a CSV of every photo + AI prediction + final label.
    Filter to one batch via ?batch_id=... — frontend always passes the
    current batch so users only download photos from this upload session."""
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=limit, batch_id=batch_id)
    tax = app.state.taxonomy or load_taxonomy()
    _enrich_with_labels(rows, tax)

    import csv
    import io
    buf = io.StringIO()
    cols = [
        "original_filename", "uploaded_at", "sha256",
        "ai_hse_type_slug", "ai_hse_type_label_en", "ai_hse_type_label_vn",
        "ai_hse_confidence",
        "ai_location_slug", "ai_location_label_en", "ai_location_label_vn",
        "ai_location_confidence",
        "ai_rationale", "ai_model",
        "reviewed", "review_action", "review_note", "reviewed_at",
        "final_hse_type_slug", "final_hse_type_label_en", "final_hse_type_label_vn",
        # Fine-grained AECIS canonical sub-type, optional. Empty when the
        # inspector skipped refinement.
        "final_fine_hse_type_slug",
        "final_fine_hse_type_label_en",
        "final_fine_hse_type_label_vn",
        "final_location_slug", "final_location_label_en", "final_location_label_vn",
    ]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    csv_text = buf.getvalue()

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="violations_{stamp}.csv"',
        },
    )


@app.get("/api/export/json")
def export_json(limit: int = 5000, batch_id: str | None = None):
    """JSON dump of every photo + AI prediction + final label."""
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=limit, batch_id=batch_id)
    tax = app.state.taxonomy or load_taxonomy()
    _enrich_with_labels(rows, tax)

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return JSONResponse(
        {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(rows),
            "photos": rows,
        },
        headers={"Content-Disposition": f'attachment; filename="violations_{stamp}.json"'},
    )


@app.get("/api/export/zip")
def export_zip(limit: int = 5000, batch_id: str | None = None):
    """Stream a ZIP with each photo renamed to <hse>__<location>__<seq>.<ext>
    and a manifest.csv listing the mapping. Killer feature for non-technical
    users — drag back into Windows folders organized by violation class."""
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=limit, batch_id=batch_id)
    if not rows:
        raise HTTPException(404, "no photos to export")

    tax = app.state.taxonomy or load_taxonomy()
    _enrich_with_labels(rows, tax)

    import csv
    import io
    import tempfile
    import zipfile
    r2 = get_r2()

    # SpooledTemporaryFile keeps the ZIP in RAM until it exceeds 50 MiB,
    # then transparently spills to disk. Lets us serve big batches without
    # blowing memory or requiring a streaming-zip dependency.
    spool = tempfile.SpooledTemporaryFile(max_size=50 * 1024 * 1024)
    seq_per_class: dict[str, int] = {}
    manifest_buf = io.StringIO()
    mw = csv.writer(manifest_buf)
    mw.writerow([
        "exported_filename", "original_filename",
        "final_hse", "final_location",
        "ai_hse", "ai_location", "reviewed",
    ])

    with zipfile.ZipFile(spool, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for r in rows:
            try:
                obj = r2.get_object(Bucket=R2_BUCKET, Key=r["storage_key"])
                body = obj["Body"].read()
            except Exception as e:  # noqa: BLE001
                log.warning("export_zip: skip %s (R2 fetch failed: %s)", r["sha256"][:10], e)
                continue
            ext = (Path(r.get("original_filename") or "").suffix.lower() or ".jpg")
            hse = _slug_safe(r["final_hse_type_slug"])
            loc = _slug_safe(r["final_location_slug"])
            key = f"{hse}__{loc}"
            seq_per_class[key] = seq_per_class.get(key, 0) + 1
            new_name = f"{hse}/{hse}__{loc}__{seq_per_class[key]:03d}{ext}"
            zf.writestr(new_name, body)
            mw.writerow([
                new_name,
                r.get("original_filename") or "",
                r.get("final_hse_type_label_en") or r["final_hse_type_slug"] or "",
                r.get("final_location_label_en") or r["final_location_slug"] or "",
                r.get("ai_hse_type_label_en") or r["ai_hse_type_slug"] or "",
                r.get("ai_location_label_en") or r["ai_location_slug"] or "",
                "yes" if r["reviewed"] else "no",
            ])
        zf.writestr("manifest.csv", manifest_buf.getvalue())

    spool.seek(0)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    # StreamingResponse iterates the spool in chunks so big ZIPs don't
    # block the event loop loading into memory all at once.
    def _iter():
        while True:
            chunk = spool.read(64 * 1024)
            if not chunk:
                spool.close()
                break
            yield chunk
    return StreamingResponse(
        _iter(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="violations_{stamp}.zip"'},
    )


@app.get("/api/export/summary")
def export_summary(batch_id: str | None = None):
    """Per-batch digest used by the UI's summary card. Cheap aggregate query."""
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=5000, batch_id=batch_id)
    from collections import Counter
    hse_counts = Counter(r["final_hse_type_slug"] for r in rows if r.get("final_hse_type_slug"))
    loc_counts = Counter(r["final_location_slug"] for r in rows if r.get("final_location_slug"))
    reviewed = sum(1 for r in rows if r["reviewed"])
    confs = [r["ai_hse_confidence"] for r in rows if r.get("ai_hse_confidence")]
    return {
        "total_photos": len(rows),
        "reviewed": reviewed,
        "pending": len(rows) - reviewed,
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
        "top_hse_types": hse_counts.most_common(5),
        "top_locations": loc_counts.most_common(5),
    }


@app.get("/metrics")
def metrics():
    db = get_db()
    photos = db.table("photos").select("id", count="exact").execute()
    pending = db.table("classify_jobs").select("id", count="exact").eq("status", "pending").execute()
    done = db.table("classify_jobs").select("id", count="exact").eq("status", "done").execute()
    errs = db.table("classify_jobs").select("id", count="exact").eq("status", "error").execute()
    return {
        "photos_total": photos.count,
        "jobs_pending": pending.count,
        "jobs_done": done.count,
        "jobs_error": errs.count,
        "training_set_size": _training_set_size(),
    }
