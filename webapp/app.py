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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _storage_key(tenant_id: str, project_id: str, sha: str, ext: str) -> str:
    # partitioned for listing sanity: <tenant>/<project>/<sha[:2]>/<sha>.<ext>
    return f"{tenant_id}/{project_id}/{sha[:2]}/{sha}{ext}"


def _upsert_embedding_for_correction(photo_id: str, hse_slug: str | None,
                                     loc_slug: str | None) -> None:
    """Feedback loop: when an inspector confirms or corrects a photo,
    compute its CLIP embedding (if not already in pgvector) and upsert it
    with the inspector-verified labels. This makes future retrievals
    benefit from human-verified ground truth.

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
        db.table("photo_embeddings").upsert({
            "sha256": sha,
            "hse_type_slug": hse_slug,
            "location_slug": loc_slug,
            "label_source": "manual",
            "project_code": "INSPECTOR",
            "issue_id": photo_id,
            "source_path": f"inspector_corrected/{photo_id}",
            "embedding": vec.tolist(),
            "tenant_id": p.get("tenant_id"),
        }, on_conflict="sha256").execute()
        log.info("feedback-loop: upserted embedding for photo=%s hse=%s loc=%s",
                 photo_id, hse_slug, loc_slug)
    except Exception as e:  # noqa: BLE001
        log.warning("feedback-loop: upsert failed for photo=%s: %s", photo_id, e)


# ---------- routes ----------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    tax = app.state.taxonomy or load_taxonomy()
    return TEMPLATES.TemplateResponse(
        request, "index.html",
        {"taxonomy": tax, "model": os.environ.get("OPENROUTER_MODEL", "?")},
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
):
    tenant_id = tenant_id or DEFAULT_TENANT_ID
    project_id = project_id or DEFAULT_PROJECT_ID
    if not (tenant_id and project_id):
        raise HTTPException(500, "Server has no default tenant/project configured")

    db = get_db()
    r2 = get_r2()
    created = []
    for f in files:
        body = await f.read()
        sha = _hash_bytes(body)
        ext = Path(f.filename or "").suffix.lower() or ".jpg"
        key = _storage_key(tenant_id, project_id, sha, ext)

        existing = (
            db.table("photos").select("id").eq("tenant_id", tenant_id)
              .eq("sha256", sha).execute()
        )
        if existing.data:
            created.append({"id": existing.data[0]["id"], "dedup": True})
            continue

        r2.put_object(
            Bucket=R2_BUCKET, Key=key, Body=body,
            ContentType=f.content_type or "image/jpeg",
        )
        photo_row = db.table("photos").insert({
            "tenant_id": tenant_id,
            "project_id": project_id,
            "storage_key": key,
            "storage_bucket": R2_BUCKET,
            "sha256": sha,
            "original_filename": f.filename,
            "bytes": len(body),
        }).execute().data[0]
        db.table("classify_jobs").insert({"photo_id": photo_row["id"]}).execute()
        created.append({"id": photo_row["id"], "dedup": False})

    return {"uploaded": created, "count": len(created)}


@app.get("/api/pending")
def api_pending(limit: int = 40):
    """Return the most recent photos + their current classification.

    Poll this from the frontend every few seconds to update cards as the
    worker classifies them.
    """
    db = get_db()
    photos = (
        db.table("photos").select("*")
          .order("uploaded_at", desc=True).limit(limit)
          .execute().data or []
    )
    if not photos:
        return {"photos": [], "training_set_size": _training_set_size()}

    photo_ids = [p["id"] for p in photos]
    cls_rows = (
        db.table("classifications").select("*")
          .in_("photo_id", photo_ids).eq("is_current", True)
          .execute().data or []
    )
    cls_by_photo = {c["photo_id"]: c for c in cls_rows}
    # Also fetch the latest correction action per photo so the UI can mark
    # photos as already-reviewed.
    corr_rows = (
        db.table("corrections").select("photo_id, action, created_at")
          .in_("photo_id", photo_ids)
          .order("created_at", desc=True)
          .execute().data or []
    )
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
        out.append({
            "id": p["id"],
            "thumb_url": thumb,
            "original_filename": p.get("original_filename"),
            "uploaded_at": p.get("uploaded_at"),
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
            "reviewed": p["id"] in latest_correction,
            "review_action": (latest_correction.get(p["id"]) or {}).get("action"),
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
def confirm(photo_id: str):
    db = get_db()
    cls = (
        db.table("classifications")
          .select("*").eq("photo_id", photo_id).eq("is_current", True)
          .execute().data
    )
    if not cls:
        raise HTTPException(400, "no current classification")
    c = cls[0]
    db.table("corrections").insert({
        "photo_id": photo_id,
        "classification_id": c["id"],
        "action": "confirm",
        "location_slug": c["location_slug"],
        "hse_type_slug": c["hse_type_slug"],
    }).execute()
    # Feedback loop: confirmations are high-quality labels — add to retrieval index
    _upsert_embedding_for_correction(photo_id, c["hse_type_slug"], c["location_slug"])
    return {"ok": True}


@app.post("/api/photos/{photo_id}/correct")
def correct(
    photo_id: str,
    location_slug: str = Form(...),
    hse_type_slug: str = Form(...),
    note: str | None = Form(None),
):
    db = get_db()
    cls = (
        db.table("classifications")
          .select("*").eq("photo_id", photo_id).eq("is_current", True)
          .execute().data
    )
    cid = cls[0]["id"] if cls else None
    db.table("corrections").insert({
        "photo_id": photo_id,
        "classification_id": cid,
        "action": "correct",
        "location_slug": location_slug,
        "hse_type_slug": hse_type_slug,
        "note": note,
    }).execute()
    # Feedback loop: inspector-corrected label is high-quality — add to retrieval index
    _upsert_embedding_for_correction(photo_id, hse_type_slug, location_slug)
    return {"ok": True}


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
