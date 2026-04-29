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
import re
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
    StreamingResponse, PlainTextResponse, Response, FileResponse,
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

# ---------- rate limiting ----------
# slowapi: per-IP token bucket on the most-abusable endpoints. We DON'T
# rate-limit reads (api_pending, api_batches) — those are cheap and
# repeated polling is normal user behavior. We DO limit writes that
# either cost money (uploads -> classifications) or amplify load
# (retry button hammered).
#
# Defaults are generous enough that no real human bumps them; tight
# enough that a script gone rogue is contained. Override via env:
#   RATE_LIMIT_UPLOAD = "30/minute"   # default
#   RATE_LIMIT_RETRY  = "20/minute"
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT_UPLOAD = os.environ.get("RATE_LIMIT_UPLOAD", "30/minute")
    _RATE_LIMIT_RETRY  = os.environ.get("RATE_LIMIT_RETRY",  "20/minute")
    _RATE_LIMITING_ENABLED = True
    log.info("rate limiting enabled: upload=%s retry=%s",
             _RATE_LIMIT_UPLOAD, _RATE_LIMIT_RETRY)
except ImportError:
    # slowapi missing in this environment — degrade gracefully so the
    # webapp still boots. Production should always have it.
    _RATE_LIMITING_ENABLED = False
    log.warning("slowapi not installed — rate limiting disabled")


# ---------- cookie-keyed user identification + quota ----------
# Phase 2 of the v2 roadmap (FUTURE_FEATURES.txt). Until real auth
# (Azure AD / phase 5) ships, we identify users by a stable cookie.
# Every cookie maps to a row in daily_quota_usage for the rate cap and
# daily_user_stats for the admin dashboard. When auth ships, a one-shot
# migration maps cookie-keys to real user.id values and the same tables
# keep working without code changes.

_USER_COOKIE_NAME = "vai_uid"
_USER_COOKIE_MAX_AGE = 60 * 60 * 24 * 365   # 1 year
_DAILY_QUOTA = int(os.environ.get("QUOTA_FREE_PER_DAY", "30"))
_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")    # None = admin disabled
_ADMIN_COOKIE_NAME = "vai_admin"
_ADMIN_COOKIE_MAX_AGE = 60 * 60 * 12   # 12 hours
# Phase 4 — per-cookie daily cap on new HSE-class proposals. Three is
# small enough to deter spam, large enough that an inspector finding
# multiple gaps in a single shift can submit them all.
_PROPOSAL_DAILY_PER_USER = int(os.environ.get("PROPOSAL_DAILY_PER_USER", "3"))

# Phase 5 — Google OAuth. None = OAuth disabled (auth routes return 503).
_GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
_GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
# Default to the production redirect; override via env for dev.
_GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI", "https://hse.aecis.ca/auth/callback/google"
)
# Phase 7 — Azure Entra ID OAuth. Multitenant by default ('common'
# tenant in the auth URL); override AZURE_TENANT_ID for single-tenant.
_AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
_AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET")
_AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "common")
_AZURE_REDIRECT_URI = os.environ.get(
    "AZURE_REDIRECT_URI", "https://hse.aecis.ca/auth/callback/azure"
)
_SESSION_COOKIE_NAME = "vai_session"
_OAUTH_STATE_COOKIE_NAME = "vai_oauth_state"
_SESSION_MAX_AGE_SECONDS = int(os.environ.get("SESSION_MAX_AGE_DAYS", "30")) * 86400
# Phase 7 — sign-in is REQUIRED to use the webapp. Set AUTH_REQUIRED=0
# in dev to disable the gate. Whitelist below covers the routes that
# must work pre-auth (the sign-in flow itself, static assets, the SW).
_AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "1") not in ("0", "false", "no")


def _get_or_set_user_key(request: Request, response_headers: dict | None = None) -> str:
    """Read the user-key cookie; mint a fresh UUID if missing.

    Returns the key. The caller is responsible for setting the cookie
    on the outbound response when this function minted a new one —
    we expose that via response.set_cookie in the routes that care.
    """
    val = request.cookies.get(_USER_COOKIE_NAME)
    if val and len(val) >= 16:
        return val
    return f"vk_{uuid.uuid4().hex}"


def _attach_user_cookie(response, key: str) -> None:
    """Set the user-key cookie on an outbound response if it's not
    already set on this request."""
    response.set_cookie(
        _USER_COOKIE_NAME,
        value=key,
        max_age=_USER_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )


def _quota_today_used(user_key: str) -> int:
    """Return how many classifications this user has consumed today
    (UTC). 0 if no row yet."""
    if not user_key:
        return 0
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        rows = (
            get_db().table("daily_quota_usage")
              .select("photos_classified")
              .eq("user_key", user_key).eq("day", today)
              .limit(1).execute().data or []
        )
        return int(rows[0]["photos_classified"]) if rows else 0
    except Exception as e:  # noqa: BLE001
        # If the table doesn't exist yet (pre-migration), treat as
        # unbounded — quota is best-effort, not a security gate.
        if "daily_quota_usage" in str(e):
            return 0
        log.warning("quota lookup failed: %s", e)
        return 0


def _quota_increment(user_key: str, n: int) -> None:
    """Add n classifications to the user's today bucket. Upsert: row
    is created on first photo; subsequent photos atomic-add. Best-effort
    — failures are logged, not raised, so a quota table outage doesn't
    block uploads."""
    if not user_key or n <= 0:
        return
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    db = get_db()
    try:
        # Read-modify-write. PostgREST doesn't expose UPSERT-with-add
        # so we do select-then-upsert. Race window is small at our
        # write rate; even if two photos race we lose at most 1 count
        # (the limit is approximate by design).
        existing = (
            db.table("daily_quota_usage").select("photos_classified")
              .eq("user_key", user_key).eq("day", today)
              .limit(1).execute().data or []
        )
        current = int(existing[0]["photos_classified"]) if existing else 0
        db.table("daily_quota_usage").upsert({
            "user_key": user_key,
            "day": today,
            "photos_classified": current + n,
        }, on_conflict="user_key,day").execute()
    except Exception as e:  # noqa: BLE001
        if "daily_quota_usage" not in str(e):
            log.warning("quota increment failed: %s", e)


def _admin_authed(request: Request) -> bool:
    """Cheap admin gate. Cookie set after correct password POST.
    Returns False when ADMIN_PASSWORD env var isn't set (admin off).

    Phase 5 also recognises a signed-in user with users.is_admin = TRUE,
    so once the OAuth flow is in production the env-var fallback can
    be removed without breaking the dashboard."""
    if _ADMIN_PASSWORD and request.cookies.get(_ADMIN_COOKIE_NAME) == _ADMIN_PASSWORD:
        return True
    user = _get_session_user(request)
    return bool(user and user.get("is_admin"))


# ---------- phase 5: google oauth helpers ----------

def _oauth_enabled() -> bool:
    """OAuth is opt-in via env vars. When the credentials aren't set,
    every /auth/* route returns 503 with a clear message — webapp keeps
    working as anonymous-cookie-only."""
    return bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET)


def _new_session_token() -> str:
    """64-char hex string. URL-safe, unguessable, fits any cookie."""
    return secrets.token_hex(32)


def _attach_session_cookie(response, token: str) -> None:
    """Set the session cookie on an outbound response. HttpOnly so JS
    can't read it; SameSite=Lax so it survives top-level navigation
    from Google's redirect; Secure unless explicitly disabled for dev."""
    response.set_cookie(
        _SESSION_COOKIE_NAME,
        value=token,
        max_age=_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )


def _clear_session_cookie(response) -> None:
    """Logout: clear the cookie by setting an empty value with max_age=0
    AND the same Secure/HttpOnly/SameSite/Path attributes used when it
    was issued. Some browsers (Chrome, Safari) treat a Secure cookie and
    a non-Secure clearing header as different cookies and refuse to
    delete the original — Starlette's response.delete_cookie() omits
    those flags. Always re-set them explicitly here.
    """
    response.set_cookie(
        _SESSION_COOKIE_NAME,
        value="",
        max_age=0,
        path="/",
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )


def _get_session_user(request: Request) -> dict | None:
    """Look up the current session and return its user row + the
    boolean is_admin flag. Returns None when:
      - cookie absent
      - session row missing or expired
      - sessions table not yet migrated (treat as 'not signed in')
    Failures are silent — auth must never break anonymous browsing.
    """
    token = request.cookies.get(_SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        sess = (
            get_db().table("sessions").select("user_id, expires_at")
              .eq("token", token).limit(1).execute().data or []
        )
        if not sess:
            return None
        if sess[0]["expires_at"] <= now:
            return None
        user_id = sess[0]["user_id"]
        users = (
            get_db().table("users")
              .select("id, provider, email, name, picture_url, is_admin")
              .eq("id", user_id).limit(1).execute().data or []
        )
        return users[0] if users else None
    except Exception as e:  # noqa: BLE001
        # Pre-migration or transient DB hiccup — treat as not signed in.
        if "sessions" in str(e) or "users" in str(e):
            return None
        log.warning("session lookup failed: %s", e)
        return None


def _google_authorize_url(state: str) -> str:
    """Build the Google OAuth 2.0 authorize URL. We ask for openid +
    profile + email (the same set declared in API permissions). Adding
    `prompt=select_account` lets the user pick which Google identity
    to use even when they're already logged into Google in this browser
    — important for shared inspector workstations.
    """
    from urllib.parse import urlencode
    qs = urlencode({
        "client_id": _GOOGLE_CLIENT_ID,
        "redirect_uri": _GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
        "access_type": "online",
    })
    return f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"


async def _google_exchange_code(code: str) -> dict:
    """POST the auth code to Google's token endpoint and return the
    parsed JSON. Raises HTTPException on any non-200 — caller surfaces
    a 'sign-in failed, please try again' page.
    """
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": _GOOGLE_CLIENT_ID,
                "client_secret": _GOOGLE_CLIENT_SECRET,
                "redirect_uri": _GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
    if r.status_code != 200:
        log.warning("google token exchange failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(401, "google token exchange failed")
    return r.json()


async def _google_fetch_userinfo(access_token: str) -> dict:
    """Pull the user's profile claims (sub, email, name, picture) from
    the Google userinfo endpoint. Avoids parsing the id_token JWT
    ourselves — Google's userinfo is the canonical surface for the same
    claims and saves us a JWT-verification dependency."""
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if r.status_code != 200:
        log.warning("google userinfo failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(401, "google userinfo failed")
    return r.json()


def _upsert_user(provider: str, profile: dict) -> dict:
    """Find-or-create the users row for this provider/sub. Returns the
    full row. Updates email/name/picture on every login since Google
    profile fields can change between sessions."""
    from datetime import datetime, timezone
    db = get_db()
    sub = profile.get("sub") or profile.get("oid")
    if not sub:
        raise HTTPException(401, "missing user identifier from provider")
    existing = (
        db.table("users").select("*")
          .eq("provider", provider)
          .eq("provider_user_id", sub)
          .limit(1).execute().data or []
    )
    now = datetime.now(timezone.utc).isoformat()
    if existing:
        user = existing[0]
        db.table("users").update({
            "email": profile.get("email") or user.get("email"),
            "name": profile.get("name") or user.get("name"),
            "picture_url": profile.get("picture") or user.get("picture_url"),
            "last_seen_at": now,
        }).eq("id", user["id"]).execute()
        return {**user,
                "email": profile.get("email") or user.get("email"),
                "name": profile.get("name") or user.get("name"),
                "picture_url": profile.get("picture") or user.get("picture_url"),
                "last_seen_at": now}
    inserted = (
        db.table("users").insert({
            "provider": provider,
            "provider_user_id": sub,
            "email": profile.get("email"),
            "name": profile.get("name"),
            "picture_url": profile.get("picture"),
            "last_seen_at": now,
        }).execute().data or []
    )
    if not inserted:
        raise HTTPException(500, "failed to create user record")
    return inserted[0]


def _create_session(user_id: str, request: Request) -> str:
    """Insert a sessions row and return the token to put in the cookie.
    The token is the only secret — user_id never leaves the server."""
    from datetime import datetime, timezone, timedelta
    token = _new_session_token()
    expires = datetime.now(timezone.utc) + timedelta(seconds=_SESSION_MAX_AGE_SECONDS)
    ua = request.headers.get("user-agent", "")[:500]
    fwd = request.headers.get("x-forwarded-for") or ""
    ip = (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else None))
    try:
        get_db().table("sessions").insert({
            "user_id": user_id,
            "token": token,
            "expires_at": expires.isoformat(),
            "user_agent": ua,
            "ip": ip,
        }).execute()
    except Exception as e:  # noqa: BLE001
        log.error("session insert failed: %s", e)
        raise HTTPException(500, "session create failed")
    return token


def _retro_attribute_user(user_id: str, user_key: str | None) -> dict[str, int]:
    """Stamp user_id onto every existing row currently keyed by the
    user's pre-auth cookie value. Idempotent — re-running on subsequent
    logins is a no-op once everything is already attributed.

    Best-effort: per-table failures are logged but don't break sign-in.
    Returns a {table: rows_touched} dict for the audit log + admin
    visibility.
    """
    if not user_key:
        return {}
    db = get_db()
    counts: dict[str, int] = {}
    # Tables to retro-patch + the column that holds the cookie key.
    tables = [
        ("photos", "user_key"),
        ("corrections", "user_key"),
        ("hse_class_proposals", "proposed_by_user_key"),
        ("daily_quota_usage", "user_key"),
        ("email_jobs", "user_key"),
    ]
    for tbl, key_col in tables:
        try:
            (db.table(tbl)
               .update({"user_id": user_id})
               .eq(key_col, user_key)
               .is_("user_id", "null")
               .execute())
            # PostgREST doesn't return updated row count cheaply for
            # bulk update without a select; we report 'best-effort' as
            # 1 (touched) and rely on the retry-idempotency.
            counts[tbl] = 1
        except Exception as e:  # noqa: BLE001
            # Pre-migration tables (no user_id column yet) raise;
            # log + continue so partially-migrated environments
            # still let the user sign in.
            log.warning("retro-attribute %s failed: %s", tbl, e)
            counts[tbl] = 0
    return counts


# ---------- phase 7: azure entra id oauth helpers ----------

def _azure_oauth_enabled() -> bool:
    """Azure OAuth is opt-in alongside Google. Both can be enabled at
    once — the sign-in chooser shows whichever providers are configured."""
    return bool(_AZURE_CLIENT_ID and _AZURE_CLIENT_SECRET)


def _azure_authorize_url(state: str) -> str:
    """Build the Azure Entra ID authorize URL (v2.0 endpoint).

    For multitenant apps (AZURE_TENANT_ID=common) any work/school
    Microsoft account in any tenant can sign in. Single-tenant deployments
    set AZURE_TENANT_ID to the tenant GUID and only members of that
    tenant can authenticate.

    Scopes: openid + profile + email + User.Read (Microsoft Graph).
    User.Read lets us call /me to fetch displayName, mail, oid; the
    other three are the standard OIDC claims subset.
    """
    from urllib.parse import urlencode
    qs = urlencode({
        "client_id": _AZURE_CLIENT_ID,
        "redirect_uri": _AZURE_REDIRECT_URI,
        "response_type": "code",
        "response_mode": "query",
        "scope": "openid email profile User.Read",
        "state": state,
        "prompt": "select_account",
    })
    return (
        f"https://login.microsoftonline.com/{_AZURE_TENANT_ID}/oauth2/v2.0/authorize?{qs}"
    )


async def _azure_exchange_code(code: str) -> dict:
    """POST the auth code to Microsoft's v2.0 token endpoint and return
    the parsed JSON. Raises HTTPException on any non-200."""
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"https://login.microsoftonline.com/{_AZURE_TENANT_ID}/oauth2/v2.0/token",
            data={
                "code": code,
                "client_id": _AZURE_CLIENT_ID,
                "client_secret": _AZURE_CLIENT_SECRET,
                "redirect_uri": _AZURE_REDIRECT_URI,
                "grant_type": "authorization_code",
                "scope": "openid email profile User.Read",
            },
        )
    if r.status_code != 200:
        log.warning("azure token exchange failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(401, "azure token exchange failed")
    return r.json()


async def _azure_fetch_userinfo(access_token: str) -> dict:
    """Fetch the signed-in user's profile from Microsoft Graph /me.

    Maps the Graph response into the same shape used by Google so
    _upsert_user can stay provider-agnostic:
      sub      -> 'id' (Graph object id; identical to the JWT 'oid' claim)
      email    -> mail | userPrincipalName  (mail can be null for
                  personal MSAs; userPrincipalName is always present)
      name     -> displayName
      picture  -> not exposed; left None. Graph /me/photo/$value returns
                  binary, which would need a separate proxy route.
    """
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if r.status_code != 200:
        log.warning("azure /me failed: %s %s", r.status_code, r.text[:300])
        raise HTTPException(401, "azure userinfo failed")
    me = r.json()
    return {
        "sub": me.get("id"),
        "email": me.get("mail") or me.get("userPrincipalName"),
        "name": me.get("displayName"),
        "picture": None,
    }


# ---------- phase 7: auth gate (sign-in required) ----------

# Path prefixes that bypass the gate. Anything matching one of these
# can be reached without a session — keep this list narrow.
_AUTH_GATE_WHITELIST_PREFIXES = (
    "/auth/",            # OAuth flow itself + signin chooser + /auth/me
    "/static/",          # Tailwind CDN, app CSS, manifest, icons
    "/admin/login",      # Env-password admin fallback (still works pre-auth)
    "/service-worker.js",  # SW MUST load pre-auth or PWA breaks
    "/manifest.json",    # Some browsers fetch this without cookies
    "/favicon",          # /favicon.ico, /favicon.png
    "/healthz",          # Future health endpoint, not currently used
    "/metrics",          # Existing metrics endpoint — keep open for now
)

# Exact paths the gate lets through. The root handler is split: it
# renders the landing page for anonymous visitors and the app shell for
# authenticated ones, so the gate must NOT short-circuit it to /signin.
_AUTH_GATE_WHITELIST_EXACT = (
    "/",
)


def _is_authed(request: Request) -> bool:
    """A request is 'authed' if it has either a valid OAuth session
    OR the env-password admin cookie. The latter keeps the env-var
    bootstrap path working pre-OAuth-rollout."""
    if _admin_authed(request):
        return True
    return _get_session_user(request) is not None


@app.middleware("http")
async def auth_gate_middleware(request: Request, call_next):
    """Phase 7 — sign-in is required to use the webapp.

    Whitelisted prefixes (the OAuth flow itself, static assets, the
    service worker, the env-password admin login form) bypass the gate.
    Everything else needs a valid session OR the env-password admin
    cookie. Unauthed HTML requests redirect to /auth/signin with the
    original path passed as `next`; unauthed API requests get a 401
    JSON so frontend fetch() handlers can react.

    Disable the gate via `AUTH_REQUIRED=0` in the env (dev / testing).
    """
    if not _AUTH_REQUIRED:
        return await call_next(request)

    path = request.url.path

    # Whitelist short-circuits the gate. Keep this list minimal — every
    # entry is a route that runs WITHOUT a session check.
    if path in _AUTH_GATE_WHITELIST_EXACT:
        return await call_next(request)
    if any(path.startswith(p) for p in _AUTH_GATE_WHITELIST_PREFIXES):
        return await call_next(request)

    # Authed: continue. We deliberately don't pre-fetch the user onto
    # request.state — most handlers don't need it and the few that do
    # call _get_session_user() themselves (one extra query is cheap).
    if _is_authed(request):
        return await call_next(request)

    # Unauthed:
    #   - /api/* and any request that obviously expects JSON -> 401 JSON
    #   - everything else -> 303 redirect to /auth/signin?next=<path>
    accept = request.headers.get("accept", "")
    wants_json = (
        path.startswith("/api/")
        or "application/json" in accept
        or "application/json" in request.headers.get("content-type", "")
    )
    if wants_json:
        return JSONResponse(
            content={"error": "auth_required", "login_url": "/auth/signin"},
            status_code=401,
        )

    from urllib.parse import quote
    return RedirectResponse(
        f"/auth/signin?next={quote(path, safe='')}", status_code=303
    )


# Registered AFTER the auth gate so it's the OUTER wrapper — adds
# headers to every response, including the gate's 303 / 401 outputs.
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Apply baseline security + cache hygiene headers to every response.

    Hardening pass added after a QA sweep noticed the production / and
    /admin responses had ZERO security headers. Without these:
      - Browsers may iframe the app into another origin (clickjacking)
      - Browsers may sniff content type from response body
      - Referer leaks to outbound links include the full path + query
      - HSTS not announced -> first visit could downgrade to HTTP if
        a captive portal hijacks DNS

    Cache-Control on / is critical: the same URL serves the landing
    page (anonymous) and the app shell (authenticated). Without
    no-store, an authed user's app HTML could be cached by a shared
    proxy or browser and served back to a logged-out visitor.
    """
    response = await call_next(request)

    # Universal hygiene headers — cheap, safe defaults.
    response.headers.setdefault(
        "Strict-Transport-Security",
        "max-age=63072000; includeSubDomains; preload",
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "geolocation=(), microphone=(), camera=(self), payment=()",
    )

    # `/` dispatches between landing + app based on auth state — never
    # cache it shared. Same for /admin and /auth/me which leak
    # session-tied state. Static + SW handle their own cache headers.
    path = request.url.path
    if path == "/" or path.startswith("/admin") or path == "/auth/me":
        response.headers["Cache-Control"] = "private, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"

    return response


# ---------- phase 4: hse-class proposal helpers ----------

_SLUG_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def _normalize_proposed_slug(label_en: str) -> str:
    """Convert an English label into a slug that matches the existing
    fine_hse_types_by_parent.json convention: leading/trailing junk
    stripped, runs of non-alnum collapsed to '_', truncated to ~80 chars.

    Example:
      "Worker without harness on scaffold" -> "Worker_without_harness_on_scaffold"
      "PPE — missing!! (gloves)"          -> "PPE_missing_gloves"
    """
    s = _SLUG_NON_ALNUM.sub("_", (label_en or "").strip()).strip("_")
    return s[:80] or "proposal"


def _unique_proposed_slug(base: str) -> str:
    """Return `base`, or `base_2`, `base_3`, ... if the DB already has
    that proposed_slug. Bounded loop — gives up after 50 attempts and
    falls back to a uuid suffix so the unique constraint never blocks a
    submission. The loop bound is generous; real collisions are rare."""
    db = get_db()
    candidate = base
    for n in range(2, 52):
        try:
            rows = (
                db.table("hse_class_proposals").select("id")
                  .eq("proposed_slug", candidate)
                  .limit(1).execute().data or []
            )
        except Exception as e:  # noqa: BLE001
            # Table missing (pre-migration) — return base; the insert
            # will surface the real error if any.
            if "hse_class_proposals" in str(e):
                return base
            raise
        if not rows:
            return candidate
        candidate = f"{base}_{n}"
    # 50-deep collision is implausible — punt with a uuid suffix.
    return f"{base}_{uuid.uuid4().hex[:6]}"


def _proposal_count_today(user_key: str) -> int:
    """Count proposals submitted by user_key in the current UTC day.
    Used by the throttle check before insert. Returns 0 on error so a
    flaky DB doesn't accidentally block legit submissions."""
    if not user_key:
        return 0
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        rows = (
            get_db().table("hse_class_proposals")
              .select("id", count="exact")
              .eq("proposed_by_user_key", user_key)
              .gte("created_at", f"{today}T00:00:00Z")
              .limit(1).execute()
        )
        return int(rows.count or 0)
    except Exception as e:  # noqa: BLE001
        if "hse_class_proposals" in str(e):
            return 0
        log.warning("proposal count failed: %s", e)
        return 0


def _reload_fine_types_cache() -> None:
    """Force the next request to re-read fine_hse_types_by_parent.json.
    Called after the admin appends a new approved slug, so workers see
    the new vocabulary on the next /api/upload without a process restart.
    """
    if hasattr(app.state, "fine_types"):
        try:
            delattr(app.state, "fine_types")
        except Exception:  # noqa: BLE001
            pass


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


def _extract_gps(body: bytes) -> tuple[float, float] | None:
    """Pull (lat, lon) decimal-degrees from the photo's EXIF if present.
    Used so map views and geo-filters can land on real on-site coordinates
    without the inspector typing them in. Cell phones with location
    services on tag every photo with this; DSLRs and screenshots don't.
    Returns None on any error (missing tags, broken EXIF, etc.) — never
    raises, since this is a best-effort enrichment, not a gate."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
        import io as _io
        with Image.open(_io.BytesIO(body)) as img:
            exif = img._getexif() or {}
        # Find the GPSInfo sub-IFD via its EXIF tag id
        gps_ifd_id = next((k for k, v in TAGS.items() if v == "GPSInfo"), None)
        if gps_ifd_id is None or gps_ifd_id not in exif:
            return None
        gps_raw = exif[gps_ifd_id]
        # Translate sub-tag ids to names, e.g. {1: "N", 2: (deg, min, sec), ...}
        gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

        def _to_deg(dms) -> float:
            d, m, s = (float(x) for x in dms)
            return d + m / 60 + s / 3600

        if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
            return None
        lat = _to_deg(gps["GPSLatitude"])
        lon = _to_deg(gps["GPSLongitude"])
        if gps.get("GPSLatitudeRef") in ("S", b"S"):
            lat = -lat
        if gps.get("GPSLongitudeRef") in ("W", b"W"):
            lon = -lon
        # Sanity bounds — catch corrupted EXIF tagging photos as (0, 0)
        # which Apple sometimes emits when GPS is enabled but no fix was
        # acquired. (0, 0) is a real point in the Atlantic, so skipping
        # exact zeros loses 0.0001% of legitimate uses for high signal.
        if lat == 0 and lon == 0:
            return None
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None
        return (round(lat, 6), round(lon, 6))
    except Exception as e:  # noqa: BLE001
        log.debug("EXIF GPS extract failed: %s", e)
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
    for _ in range(5):  # at most one retry per optional column
        try:
            db.table("corrections").insert(p).execute()
            return
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            stripped = False
            for col in ("fine_hse_type_slug", "batch_id", "user_key"):
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


@app.get("/service-worker.js", include_in_schema=False)
def service_worker_js():
    """Serve the PWA service worker from the ROOT path with the
    Service-Worker-Allowed header set to '/'. Required so the SW
    registered with scope '/' can intercept the whole app even though
    its source file lives in /static/. Without this, browsers reject
    the registration with a "scope outside script's path" error.

    Cache-Control: no-cache so users always get the latest SW after
    a deploy (the SW itself is what manages other caches; if it
    cached itself, deploying a new version would be a chicken-egg).
    """
    p = STATIC_DIR / "service-worker.js"
    if not p.exists():
        raise HTTPException(404, "service worker not built")
    return FileResponse(
        p,
        media_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


@app.head("/", include_in_schema=False)
def root_head():
    """HEAD / — uptime monitors and link previewers send HEAD before
    GET. Without this, FastAPI returns 405 and the monitor thinks the
    site is down. Return 200 with no body; security headers from the
    middleware still get applied.
    """
    return Response(status_code=200)


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Root handler — softlocks the app behind sign-in.

    Anonymous visitors get the marketing landing page (introduces the
    product, sign-in CTA, public metadata only). Authenticated visitors
    get the existing app shell. This means / is a useful URL to share
    externally (no immediate redirect to /auth/signin), and existing
    bookmarks of / continue to work for signed-in inspectors without
    a round-trip.
    """
    if not _is_authed(request):
        return _render_landing_page(request)

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


def _render_landing_page(request: Request) -> HTMLResponse:
    """Public marketing page rendered to anonymous visitors at /.

    Black & white minimalist theme. Hero section sits inside a 16:9
    'wave strip' framed by hairline white borders, top and bottom.
    The wave is a vanilla-canvas reconstruction of the 21st.dev
    `xubohuah/wave-background` component:
      - vertical lines + horizontal sub-points form a flexible mesh
      - per-frame distortion via inline 2D simplex noise (no
        npm `simplex-noise` dep — same algorithm, hand-rolled in
        ~30 lines so the page stays self-contained for SW caching)
      - cursor proximity adds local velocity, decays back via
        spring damping
    Bilingual EN/VN, mobile responsive, fully inline.
    """
    google_on = _oauth_enabled()
    azure_on = _azure_oauth_enabled()
    providers_blurb_en = "Sign in with " + (
        "Google or Microsoft" if (google_on and azure_on) else
        ("Google" if google_on else ("Microsoft" if azure_on else "your provider"))
    )
    providers_blurb_vn = "Đăng nhập bằng " + (
        "Google hoặc Microsoft" if (google_on and azure_on) else
        ("Google" if google_on else ("Microsoft" if azure_on else "nhà cung cấp của bạn"))
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Violation AI — AECIS HSE inspection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#000000">
<meta name="description" content="AI-powered HSE violation detection for construction sites — built for AECIS inspectors. Sign in with Google or Microsoft to access the inspection workflow.">
<link rel="manifest" href="/static/manifest.json">
<link rel="apple-touch-icon" href="/static/icons/icon-180.png">
<link rel="icon" type="image/png" sizes="192x192" href="/static/icons/icon-192.png">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, "Segoe UI", sans-serif;
       color: #ffffff; background: #000000; min-height: 100vh;
       -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
       overflow-x: hidden; }}
.shell {{ display: flex; flex-direction: column; min-height: 100vh; }}

header {{ display: flex; justify-content: space-between; align-items: center;
         padding: 18px 24px; max-width: 1280px; margin: 0 auto; width: 100%;
         position: relative; z-index: 5; }}
.brand {{ display: inline-flex; align-items: center; gap: 10px;
         font-weight: 600; font-size: 16px; color: #ffffff; text-decoration: none;
         letter-spacing: -0.01em; }}
.brand .check {{ width: 26px; height: 26px; border: 1px solid rgba(255,255,255,0.4);
                border-radius: 999px; display: grid; place-items: center; }}
.brand .check svg {{ width: 14px; height: 14px; }}
.brand .accent {{ color: rgba(255,255,255,0.45); font-weight: 400; }}

.locale-toggle {{ display: flex; gap: 0; border: 1px solid rgba(255,255,255,0.18);
                 border-radius: 999px; overflow: hidden; }}
.locale-toggle button {{ font: inherit; background: transparent; border: 0; cursor: pointer;
                        padding: 5px 12px; color: rgba(255,255,255,0.55); font-weight: 500; font-size: 11px;
                        letter-spacing: 0.04em; transition: background .15s, color .15s; }}
.locale-toggle button:hover {{ color: #ffffff; }}
.locale-toggle button.active {{ background: #ffffff; color: #000000; }}

/* ============= Hero with wave strip ============= */
.hero {{ position: relative; width: 100%;
        min-height: 72vh; display: flex; align-items: center; justify-content: center;
        padding: 64px 16px; overflow: hidden; }}
.hero-canvas {{ position: absolute; inset: 0; width: 100%; height: 100%;
               z-index: 0; pointer-events: auto; }}
.hero-hairline {{ position: absolute; left: 0; right: 0; height: 1px;
                 background: rgba(255,255,255,0.7); z-index: 2; pointer-events: none; }}
.hero-hairline.top {{ top: 0; }}
.hero-hairline.bottom {{ bottom: 0; }}
.hero-fade {{ position: absolute; left: 0; right: 0; height: 80px; z-index: 1;
             pointer-events: none; }}
.hero-fade.top {{ top: 0; background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 100%); }}
.hero-fade.bottom {{ bottom: 0; background: linear-gradient(0deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 100%); }}

.hero-content {{ position: relative; z-index: 3; max-width: 880px; width: 100%;
                text-align: center; pointer-events: none; }}
.hero-content > * {{ pointer-events: auto; }}

.eyebrow {{ display: inline-block; font-size: 11px; text-transform: uppercase;
           letter-spacing: 0.18em; font-weight: 600; color: rgba(255,255,255,0.7);
           border: 1px solid rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 999px;
           margin-bottom: 24px; backdrop-filter: blur(8px); background: rgba(0,0,0,0.3); }}
h1 {{ font-size: clamp(36px, 6.5vw, 76px); line-height: 1.02; font-weight: 600;
     margin: 0 0 22px; letter-spacing: -0.035em; color: #ffffff; }}
h1 .em {{ font-style: italic; font-weight: 400; color: rgba(255,255,255,0.85); }}
.lead {{ font-size: clamp(15px, 1.6vw, 18px); color: rgba(255,255,255,0.65);
        margin: 0 auto 36px; max-width: 620px; line-height: 1.6; }}
.cta {{ display: inline-flex; align-items: center; gap: 10px; background: #ffffff;
       color: #000000; text-decoration: none; padding: 14px 28px; border-radius: 999px;
       font-weight: 600; font-size: 14px; letter-spacing: 0.01em;
       transition: transform .04s, background .15s, color .15s, box-shadow .15s;
       box-shadow: 0 0 0 1px rgba(255,255,255,0.05), 0 8px 32px rgba(255,255,255,0.08); }}
.cta:hover {{ background: rgba(255,255,255,0.92); box-shadow: 0 0 0 1px rgba(255,255,255,0.1), 0 12px 40px rgba(255,255,255,0.12); }}
.cta:active {{ transform: scale(0.98); }}
.cta svg {{ width: 16px; height: 16px; }}
.providers-note {{ display: block; margin-top: 18px; color: rgba(255,255,255,0.4);
                  font-size: 12px; letter-spacing: 0.02em; }}

/* ============= Features ============= */
.features-section {{ padding: 80px 16px 60px; max-width: 1080px; margin: 0 auto;
                    width: 100%; }}
.features-eyebrow {{ display: block; text-align: center; font-size: 11px;
                    text-transform: uppercase; letter-spacing: 0.18em;
                    color: rgba(255,255,255,0.4); margin-bottom: 32px; font-weight: 600; }}
.features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 0; }}
.feature {{ background: #000000; padding: 28px 24px; }}
.feature .ic {{ width: 32px; height: 32px; border: 1px solid rgba(255,255,255,0.25);
                color: #ffffff; display: grid; place-items: center; margin-bottom: 16px;
                border-radius: 6px; }}
.feature .ic svg {{ width: 16px; height: 16px; }}
.feature h3 {{ font-size: 15px; margin: 0 0 8px; font-weight: 600; color: #ffffff;
              letter-spacing: -0.005em; }}
.feature p {{ font-size: 13px; color: rgba(255,255,255,0.55); margin: 0; line-height: 1.6; }}

/* ============= Footer ============= */
footer {{ text-align: center; color: rgba(255,255,255,0.35); font-size: 11px;
         padding: 40px 16px 32px; letter-spacing: 0.02em;
         border-top: 1px solid rgba(255,255,255,0.08); margin-top: auto; }}
footer a {{ color: rgba(255,255,255,0.7); text-decoration: none; }}
footer a:hover {{ color: #ffffff; text-decoration: underline; }}

[data-locale]:not(.show) {{ display: none; }}

@media (max-width: 640px) {{
  header {{ padding: 14px 16px; }}
  .hero {{ min-height: 60vh; padding: 48px 12px; }}
  .features-section {{ padding: 56px 12px 48px; }}
}}
</style></head><body>
<div class="shell">
<header>
  <a class="brand" href="/">
    <span class="check"><svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg></span>
    <span>Violation <span class="accent">/ AI</span></span>
  </a>
  <div class="locale-toggle">
    <button id="loc-en" type="button" class="active">EN</button>
    <button id="loc-vn" type="button">VN</button>
  </div>
</header>

<section class="hero">
  <canvas id="waves" class="hero-canvas"></canvas>
  <div class="hero-fade top"></div>
  <div class="hero-fade bottom"></div>
  <div class="hero-hairline top"></div>
  <div class="hero-hairline bottom"></div>
  <div class="hero-content">
    <span class="eyebrow">
      <span data-locale="en">AECIS HSE inspection · powered by AI</span>
      <span data-locale="vn">Kiểm tra HSE AECIS · do AI hỗ trợ</span>
    </span>
    <h1>
      <span data-locale="en">Spot construction-site safety violations <span class="em">in seconds.</span></span>
      <span data-locale="vn">Phát hiện vi phạm an toàn công trường <span class="em">trong vài giây.</span></span>
    </h1>
    <p class="lead">
      <span data-locale="en">Upload site photos. The AI classifies them against the AECIS HSE taxonomy in English &amp; Vietnamese, ranks confidence, and lets you correct on the spot. Built for inspectors, exported as PDF / ZIP / CSV.</span>
      <span data-locale="vn">Tải ảnh hiện trường lên. AI phân loại theo bảng phân loại HSE AECIS bằng tiếng Anh &amp; tiếng Việt, xếp hạng độ tin cậy và cho phép bạn chỉnh tại chỗ. Xây dựng cho cán bộ kiểm tra, xuất PDF / ZIP / CSV.</span>
    </p>
    <a class="cta" href="/auth/signin">
      <span data-locale="en">Sign in to continue</span>
      <span data-locale="vn">Đăng nhập để tiếp tục</span>
      <svg fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
    </a>
    <span class="providers-note">
      <span data-locale="en">{providers_blurb_en} · AECIS-internal &amp; partner accounts only</span>
      <span data-locale="vn">{providers_blurb_vn} · chỉ dành cho tài khoản nội bộ &amp; đối tác AECIS</span>
    </span>
  </div>
</section>

<section class="features-section">
  <span class="features-eyebrow">
    <span data-locale="en">What it does</span>
    <span data-locale="vn">Tính năng</span>
  </span>
  <div class="features">
    <div class="feature">
      <div class="ic"><svg fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5l5-5 4 4 6-7 3 3v6.5a2 2 0 01-2 2H5a2 2 0 01-2-2V16.5z"/></svg></div>
      <h3>
        <span data-locale="en">Upload &amp; classify</span>
        <span data-locale="vn">Tải lên &amp; phân loại</span>
      </h3>
      <p>
        <span data-locale="en">Drop site photos straight from your phone. The AI tags HSE violation type and fine-grained AECIS sub-type within seconds.</span>
        <span data-locale="vn">Thả ảnh từ điện thoại. AI gắn nhãn loại vi phạm HSE và loại con AECIS chi tiết trong vài giây.</span>
      </p>
    </div>
    <div class="feature">
      <div class="ic"><svg fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.6 2A9 9 0 11 3.4 14a9 9 0 0117.2 0z"/></svg></div>
      <h3>
        <span data-locale="en">Confirm or correct</span>
        <span data-locale="vn">Xác nhận hoặc chỉnh sửa</span>
      </h3>
      <p>
        <span data-locale="en">Every confirmation trains the model. Wrong call? Re-mark the region or propose a new sub-type for admin review.</span>
        <span data-locale="vn">Mỗi xác nhận đều giúp AI học. Sai? Vẽ lại vùng hoặc đề xuất loại con mới để quản trị viên duyệt.</span>
      </p>
    </div>
    <div class="feature">
      <div class="ic"><svg fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg></div>
      <h3>
        <span data-locale="en">Export &amp; share</span>
        <span data-locale="vn">Xuất &amp; chia sẻ</span>
      </h3>
      <p>
        <span data-locale="en">PDF cover-sheet, renamed-photos ZIP, CSV, or JSON — emailed straight to the safety officer or downloaded.</span>
        <span data-locale="vn">PDF có trang bìa, ZIP ảnh đã đổi tên, CSV hoặc JSON — gửi email cho cán bộ an toàn hoặc tải về.</span>
      </p>
    </div>
  </div>
</section>

<footer>
  <span data-locale="en">© AECIS · HSE Inspection Tool · <a href="/auth/signin">Sign in</a></span>
  <span data-locale="vn">© AECIS · Công cụ kiểm tra HSE · <a href="/auth/signin">Đăng nhập</a></span>
</footer>
</div>

<script>
  // ============= Locale toggle (same pattern as before) =============
  (function () {{
    const saved = localStorage.getItem("vai-lang") || "en";
    const enBtn = document.getElementById("loc-en");
    const vnBtn = document.getElementById("loc-vn");
    function apply(loc) {{
      document.querySelectorAll("[data-locale]").forEach(el => {{
        el.classList.toggle("show", el.dataset.locale === loc);
      }});
      enBtn.classList.toggle("active", loc === "en");
      vnBtn.classList.toggle("active", loc === "vn");
      localStorage.setItem("vai-lang", loc);
      document.documentElement.lang = loc === "vn" ? "vi" : "en";
    }}
    enBtn.addEventListener("click", () => apply("en"));
    vnBtn.addEventListener("click", () => apply("vn"));
    apply(saved);
  }})();

  // ============= Mesh-gradient shader =============
  // Vanilla WebGL port of the 21st.dev reuno-ui/background-paper-shaders
  // component, which uses @paper-design/shaders-react's <MeshGradient>
  // with grayscale colors ["#000000", "#1a1a1a", "#333333", "#ffffff"].
  // We don't have React or the npm package available; this fragment
  // shader produces the same visual effect — animated grayscale forms
  // smoothly interpolated between four drifting control points using
  // inverse-distance-squared (Shepard) weighting.
  (function () {{
    const canvas = document.getElementById("waves");
    if (!canvas) return;
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) {{
      // No WebGL — fall back to a solid black canvas. Hairlines + text
      // still render correctly above it.
      const ctx2d = canvas.getContext("2d");
      function paint() {{
        const r = canvas.getBoundingClientRect();
        canvas.width = r.width; canvas.height = r.height;
        ctx2d.fillStyle = "#000"; ctx2d.fillRect(0, 0, r.width, r.height);
      }}
      paint();
      window.addEventListener("resize", paint);
      return;
    }}

    // ---- Shaders ----
    // Vertex shader: trivial fullscreen quad.
    const VS = `attribute vec2 a_pos;
void main() {{
  gl_Position = vec4(a_pos, 0.0, 1.0);
}}`;

    // Fragment shader: Shepard interpolation across four moving
    // control points in grayscale (#000, #1a1a1a, #333, #fff). The
    // power on the inverse-distance weight controls how soft the
    // transitions are; 2.0 gives a smooth, paper-shader-like feel.
    const FS = `precision highp float;
uniform vec2  u_res;
uniform float u_time;

// 4 grayscale control colors, matching the reuno-ui demo.
const vec3 c0 = vec3(0.0);                  // #000000
const vec3 c1 = vec3(0.10196);              // #1a1a1a
const vec3 c2 = vec3(0.20);                 // #333333
const vec3 c3 = vec3(1.0);                  // #ffffff

// Smooth wrap of a position around the unit square so control
// points orbit gently rather than colliding with the edges.
vec2 driftAt(float t, float a, float b, float c, float d) {{
  return vec2(0.5 + 0.42 * sin(t * a + c),
              0.5 + 0.42 * cos(t * b + d));
}}

void main() {{
  // Aspect-correct UV so distance metric isn't squashed by viewport.
  vec2 uv = gl_FragCoord.xy / u_res.xy;
  float aspect = u_res.x / max(u_res.y, 1.0);
  vec2 p = vec2(uv.x * aspect, uv.y);

  float t = u_time * 0.18;

  // Four drifting attractors, each with its own phase + speed so
  // they never line up periodically.
  vec2 q0 = driftAt(t, 0.31, 0.27, 0.0, 0.0);
  vec2 q1 = driftAt(t, 0.23, 0.41, 1.7, 2.3);
  vec2 q2 = driftAt(t, 0.37, 0.19, 3.2, 4.1);
  vec2 q3 = driftAt(t, 0.17, 0.29, 5.0, 1.1);

  // Aspect-correct attractor positions.
  q0.x *= aspect; q1.x *= aspect; q2.x *= aspect; q3.x *= aspect;

  // Higher-power inverse distance so the nearest attractor strongly
  // dominates — equal-power Shepard always pulls toward the average,
  // which would blunt contrast and leave distant pixels stuck at a
  // boring mid-gray. Power 6 gives the dramatic dark/bright contrast
  // of the reuno-ui reference.
  float d0 = distance(p, q0);
  float d1 = distance(p, q1);
  float d2 = distance(p, q2);
  float d3 = distance(p, q3);
  float w0 = 1.0 / (pow(d0, 6.0) + 1e-4);
  float w1 = 1.0 / (pow(d1, 6.0) + 1e-4);
  float w2 = 1.0 / (pow(d2, 6.0) + 1e-4);
  float w3 = 1.0 / (pow(d3, 6.0) + 1e-4);
  float total = w0 + w1 + w2 + w3;

  vec3 col = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / total;

  // Push the result a touch darker overall so the bright attractor
  // reads as a moving highlight rather than ambient lift. Black layout
  // around the hero strip stays clean.
  col = mix(vec3(0.0), col, 0.92);

  gl_FragColor = vec4(col, 1.0);
}}`;

    function compile(type, src) {{
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {{
        console.warn("[shader]", gl.getShaderInfoLog(sh));
        gl.deleteShader(sh);
        return null;
      }}
      return sh;
    }}

    const vs = compile(gl.VERTEX_SHADER, VS);
    const fs = compile(gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) return;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {{
      console.warn("[shader] link failed:", gl.getProgramInfoLog(prog));
      return;
    }}
    gl.useProgram(prog);

    // Fullscreen quad — two triangles spanning [-1, 1].
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1,  1, -1,  -1,  1,
                        -1,  1,  1, -1,   1,  1]),
      gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const uRes = gl.getUniformLocation(prog, "u_res");
    const uTime = gl.getUniformLocation(prog, "u_time");

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w;
        canvas.height = h;
      }}
      gl.viewport(0, 0, w, h);
      gl.uniform2f(uRes, w, h);
    }}

    let t0 = performance.now();
    let raf = 0;
    function frame(now) {{
      gl.uniform1f(uTime, (now - t0) / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      raf = requestAnimationFrame(frame);
    }}

    document.addEventListener("visibilitychange", () => {{
      if (document.hidden) {{
        cancelAnimationFrame(raf);
      }} else {{
        // Reset the time origin so the shader doesn't lurch forward.
        t0 = performance.now() - (now => now)(0);
        t0 = performance.now();
        raf = requestAnimationFrame(frame);
      }}
    }});

    window.addEventListener("resize", resize);
    resize();
    raf = requestAnimationFrame(frame);
  }})();
</script>
</body></html>"""
    return HTMLResponse(html)


# Legacy paths — redirect everything to /
@app.get("/upload", include_in_schema=False)
def _upload_redirect():
    return RedirectResponse("/", status_code=308)


@app.get("/review", include_in_schema=False)
def _review_redirect():
    return RedirectResponse("/", status_code=308)


def _maybe_limit(rate_attr: str):
    """Wrap an endpoint with the slowapi rate limiter if it's enabled.
    Returns a no-op decorator when slowapi isn't installed (so the rest
    of the app still works in dev/test environments without it)."""
    def decorator(fn):
        if not _RATE_LIMITING_ENABLED:
            return fn
        rate = globals()[rate_attr]
        return limiter.limit(rate)(fn)
    return decorator


@app.post("/api/upload")
@_maybe_limit("_RATE_LIMIT_UPLOAD")
async def upload(
    request: Request,
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

    # Quota gate. Read user-key cookie, look up today's usage, refuse
    # the request if already at cap. Per-photo enforcement (a 50-photo
    # batch from a free user with 28/30 used uploads the next 2 and
    # rejects the remaining 48 with a clear error).
    user_key = _get_or_set_user_key(request)
    used_today = _quota_today_used(user_key)
    remaining = max(0, _DAILY_QUOTA - used_today)
    if remaining <= 0:
        from datetime import datetime, timezone, timedelta
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        resets_at = f"{tomorrow.isoformat()}T00:00:00Z"
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_reached",
                "limit": _DAILY_QUOTA,
                "used": used_today,
                "resets_at": resets_at,
            },
        )

    db = get_db()
    r2 = get_r2()
    created = []
    rejected = []
    warnings: list[dict] = []  # per-photo quality flags (uploaded but suspect)
    accepted_count = 0
    quota_used = used_today
    for f in files:
        # Per-photo quota enforcement: stop when we hit cap mid-batch.
        # The remaining files in this request go to "rejected" with
        # reason=daily_limit so the frontend can group them in the
        # error banner.
        if quota_used >= _DAILY_QUOTA:
            rejected.append({"filename": f.filename, "reason": "daily_limit_reached"})
            continue
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
        # Read EXIF GPS off the ORIGINAL bytes (before normalization, since
        # converting HEIC->JPEG strips most metadata). Best-effort; phones
        # with location services on tag every photo, DSLRs and screenshots
        # usually don't.
        gps = _extract_gps(raw)

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
            "user_key": user_key,
        }
        if gps:
            photo_payload["exif_lat"] = gps[0]
            photo_payload["exif_lon"] = gps[1]

        def _insert_with_fallback(payload):
            """Try the full payload; on schema-error, strip the missing
            optional columns and retry once. Lets the webapp keep working
            in the window between code-deploy and SQL-migration."""
            try:
                return db.table("photos").insert(payload).execute().data[0]
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                stripped = False
                for col in ("batch_id", "batch_label", "exif_lat", "exif_lon", "user_key"):
                    if col in msg and col in payload:
                        payload.pop(col, None)
                        stripped = True
                if stripped:
                    log.warning("photos schema column missing — degraded insert: %s",
                                msg.split('\n', 1)[0][:120])
                    return db.table("photos").insert(payload).execute().data[0]
                raise

        photo_row = _insert_with_fallback(photo_payload)
        db.table("classify_jobs").insert({"photo_id": photo_row["id"]}).execute()
        created.append({"id": photo_row["id"], "dedup": False})
        accepted_count += 1
        quota_used += 1
        if quality:
            warnings.append({"filename": f.filename, "id": photo_row["id"], **quality})

    # Update the daily-quota counter once with the final accepted count
    # — single DB round-trip per upload regardless of batch size.
    if accepted_count:
        _quota_increment(user_key, accepted_count)

    response_body = {
        "uploaded": created,
        "rejected": rejected,
        "warnings": warnings,
        "count": len(created),
        "batch_id": batch_id,
        "quota_used": quota_used,
        "quota_limit": _DAILY_QUOTA,
    }
    # Set the user-key cookie on the response so subsequent /api/usage
    # calls can identify the same client. JSONResponse lets us mutate
    # cookies; the @app.post handler returning a dict gets a default
    # JSONResponse from FastAPI, so we wrap explicitly here.
    resp = JSONResponse(content=response_body)
    _attach_user_cookie(resp, user_key)
    return resp


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


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: str):
    """Hard-delete every photo in the batch — DB rows + R2 objects + the
    classify_jobs / classifications / corrections cascade. Surfaced from
    the batch-list view so the inspector can wipe a finished or abandoned
    job in one click instead of deleting photos one at a time.

    Cascade order matches /api/photos/{id} delete: classify_jobs ->
    corrections -> classifications -> photo_embeddings (only if no other
    photo shares the sha) -> photos -> R2 objects.

    Returns count of photos deleted, plus any individual failures so the
    caller can surface partial-success.
    """
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "Server has no default tenant configured")
    db = get_db()
    r2 = get_r2()

    # Fetch every photo in the batch up-front — we need the storage_keys
    # for R2 cleanup and the sha256s to know which embeddings are safe to drop.
    try:
        photos = (
            db.table("photos").select("id, storage_key, sha256")
              .eq("tenant_id", DEFAULT_TENANT_ID)
              .eq("batch_id", batch_id)
              .limit(10000).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        if "batch_id" in str(e):
            raise HTTPException(500, "batch_id column not yet migrated")
        raise
    if not photos:
        # Idempotent: empty batch = nothing to do, not an error.
        return {"ok": True, "batch_id": batch_id, "deleted": 0}

    photo_ids = [p["id"] for p in photos]
    shas = list({p["sha256"] for p in photos if p.get("sha256")})

    failures: list[dict] = []

    # Cascade child rows in chunks to avoid PostgREST URL-length limits.
    # 100 ids per IN-clause keeps us well under any practical limit.
    for tbl in ("classify_jobs", "corrections", "classifications"):
        for i in range(0, len(photo_ids), 100):
            chunk = photo_ids[i:i + 100]
            try:
                db.table(tbl).delete().in_("photo_id", chunk).execute()
            except Exception as e:  # noqa: BLE001
                log.warning("delete from %s failed for batch=%s: %s", tbl, batch_id, e)
                failures.append({"step": f"delete_{tbl}", "error": str(e)[:200]})

    # photo_embeddings: drop only the shas that no surviving photo references.
    # We're about to delete this batch's photos, so a sha is safe to drop iff
    # no OTHER photo (outside this batch) shares it.
    safe_to_drop_shas: list[str] = []
    for sha in shas:
        try:
            others = (
                db.table("photos").select("id").eq("sha256", sha)
                  .not_.in_("id", photo_ids).limit(1).execute().data or []
            )
            if not others:
                safe_to_drop_shas.append(sha)
        except Exception as e:  # noqa: BLE001
            log.warning("embedding-survival check failed for sha=%s: %s", sha[:8], e)
    for i in range(0, len(safe_to_drop_shas), 100):
        chunk = safe_to_drop_shas[i:i + 100]
        try:
            db.table("photo_embeddings").delete().in_("sha256", chunk).execute()
        except Exception as e:  # noqa: BLE001
            log.warning("photo_embeddings delete failed: %s", e)
            failures.append({"step": "delete_embeddings", "error": str(e)[:200]})

    # photos rows.
    deleted = 0
    for i in range(0, len(photo_ids), 100):
        chunk = photo_ids[i:i + 100]
        try:
            db.table("photos").delete().in_("id", chunk).execute()
            deleted += len(chunk)
        except Exception as e:  # noqa: BLE001
            log.error("photos delete failed (chunk %d): %s", i, e)
            failures.append({"step": "delete_photos", "error": str(e)[:200]})

    # R2 objects (best-effort — DB is authoritative).
    for p in photos:
        key = p.get("storage_key")
        if not key:
            continue
        try:
            r2.delete_object(Bucket=R2_BUCKET, Key=key)
        except Exception as e:  # noqa: BLE001
            log.warning("R2 delete failed for %s: %s", key, e)

    return {
        "ok": not failures,
        "batch_id": batch_id,
        "deleted": deleted,
        "embeddings_dropped": len(safe_to_drop_shas),
        "failures": failures,
    }


@app.get("/api/photos/{photo_id}/history")
def photo_history(photo_id: str):
    """Audit trail for one photo: every correction the inspector ever
    submitted, oldest first. Used by the per-card "history" panel so an
    AECIS QA reviewer can see exactly who changed what when.

    Includes both confirm and correct actions. Returns slug + label
    (resolved against the current taxonomy), the inspector note, the
    timestamp, and the batch_id (so reviewers can navigate back to the
    batch context the correction was made in)."""
    db = get_db()
    try:
        rows = (
            db.table("corrections")
              .select("id, action, location_slug, hse_type_slug, "
                      "fine_hse_type_slug, note, created_at, batch_id")
              .eq("photo_id", photo_id)
              .order("created_at", desc=False)
              .limit(200).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        log.warning("photo_history failed: %s", e)
        return {"photo_id": photo_id, "history": [], "error": str(e)[:200]}

    # Resolve labels via the current taxonomy. We don't store labels on
    # corrections rows, just slugs — keeps the table small and lets us
    # rename labels without rewriting history.
    tax = app.state.taxonomy or load_taxonomy()
    hse_lookup = {h["slug"]: h for h in tax.get("hse_types", [])}
    loc_lookup = {l["slug"]: l for l in tax.get("locations", [])}
    out = []
    for r in rows:
        hse = hse_lookup.get(r.get("hse_type_slug") or "", {})
        loc = loc_lookup.get(r.get("location_slug") or "", {})
        out.append({
            "id": r["id"],
            "action": r.get("action") or "",
            "hse_type_slug": r.get("hse_type_slug") or "",
            "hse_type_label_en": hse.get("label_en", r.get("hse_type_slug") or ""),
            "hse_type_label_vn": hse.get("label_vn", ""),
            "location_slug": r.get("location_slug") or "",
            "location_label_en": loc.get("label_en", r.get("location_slug") or ""),
            "fine_hse_type_slug": r.get("fine_hse_type_slug") or "",
            "note": r.get("note") or "",
            "created_at": r.get("created_at"),
            "batch_id": r.get("batch_id"),
        })
    return {"photo_id": photo_id, "history": out}


@app.post("/api/photos/{photo_id}/reclassify-region")
@_maybe_limit("_RATE_LIMIT_RETRY")
async def reclassify_region(request: Request, photo_id: str,
                            region: UploadFile = File(...)):
    """Re-classify a photo using a region the inspector cropped/highlighted.

    Use case: the AI got confused (e.g. picked Electrical for a photo
    where the actual hazard is the missing handrail at the floor edge).
    Inspector circles/crops the actual hazard, posts the cropped pixels
    here, and we re-run the classifier on JUST that region. Result
    overwrites the photo's current classifications row.

    The cropped image is NOT stored separately — we don't replace the
    original photo in R2. Only the classification gets updated. So the
    inspector's pixel-level annotation is ephemeral; what persists is
    the AI's better answer plus rationale.
    """
    db = get_db()
    photo_rows = db.table("photos").select("*").eq("id", photo_id).execute().data or []
    if not photo_rows:
        raise HTTPException(404, "photo not found")
    photo = photo_rows[0]

    raw = await region.read()
    if not raw or len(raw) < 100:
        raise HTTPException(400, "region payload empty or truncated")
    if not _looks_like_supported_image(raw):
        raise HTTPException(400, "region not a supported image")

    # Run the classifier on the cropped bytes via a temp file (the
    # classify pipeline expects a file path so it can run CLIP for RAG
    # retrieval on the same bytes).
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(raw)
        tmp_path = Path(tf.name)
    try:
        # Same in-session learning pull-in as the worker, so corrections
        # made earlier in the batch carry over to the re-classify call.
        recent_corrections: list[dict] = []
        try:
            cs = (
                db.table("corrections").select(
                    "hse_type_slug, fine_hse_type_slug, note, created_at")
                .eq("batch_id", photo.get("batch_id") or "")
                .order("created_at", desc=True).limit(8).execute().data or []
            )
            recent_corrections = [r for r in cs if r.get("hse_type_slug")]
        except Exception:  # noqa: BLE001
            pass

        cls = classify_image(
            tmp_path,
            recent_corrections=recent_corrections,
            fine_grained=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Replace the current classifications row.
    try:
        db.table("classifications").update({"is_current": False}).eq(
            "photo_id", photo_id).execute()
    except Exception as e:  # noqa: BLE001
        log.warning("clear-current failed for %s: %s", photo_id, e)

    insert_payload = {
        "photo_id": photo_id,
        "location_slug": cls.location.slug,
        "hse_type_slug": cls.hse_type.slug,
        "location_confidence": cls.location.confidence,
        "hse_type_confidence": cls.hse_type.confidence,
        "rationale": cls.rationale,
        "model": cls.model + "+region-marked",
        "source": "user_marked_region",
        "input_tokens": cls.input_tokens,
        "output_tokens": cls.output_tokens,
        "raw_response": cls.raw_response,
        "is_current": True,
    }
    if cls.fine_hse_type:
        insert_payload["fine_hse_type_slug"] = cls.fine_hse_type.slug
        insert_payload["fine_hse_type_confidence"] = cls.fine_hse_type.confidence
    try:
        db.table("classifications").insert(insert_payload).execute()
    except Exception as e:  # noqa: BLE001
        if "fine_hse_type" in str(e):
            insert_payload.pop("fine_hse_type_slug", None)
            insert_payload.pop("fine_hse_type_confidence", None)
            db.table("classifications").insert(insert_payload).execute()
        else:
            raise

    log.info("reclassify-region photo=%s -> hse=%s fine=%s",
             photo_id, cls.hse_type.slug,
             cls.fine_hse_type.slug if cls.fine_hse_type else "<none>")

    return {
        "ok": True,
        "photo_id": photo_id,
        "hse_type_slug": cls.hse_type.slug,
        "fine_hse_type_slug": cls.fine_hse_type.slug if cls.fine_hse_type else None,
        "fine_hse_type_label_en": cls.fine_hse_type.label_en if cls.fine_hse_type else None,
        "rationale": cls.rationale,
        "confidence": cls.hse_type.confidence,
    }


@app.post("/api/photos/{photo_id}/retry")
@_maybe_limit("_RATE_LIMIT_RETRY")
def retry_classify(request: Request, photo_id: str):
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
        alts_fine: list[dict] = []
        if cls:
            raw = cls.get("raw_response") or {}
            alts_hse = raw.get("hse_type_alternatives") or []
            alts_loc = raw.get("location_alternatives") or []
            # Fine-grained alternatives from Stage 2 of the two-stage classifier.
            # raw stores them under 'fine_hse_type_alternatives' as the canonical
            # list; UI uses them so the inspector can one-click swap to a
            # close-runner-up specific item.
            alts_fine = raw.get("fine_hse_type_alternatives") or []
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
                    # Stage-2 fine sub-type fields. May be missing on
                    # pre-migration classifications rows or when Stage 2
                    # confidence was below threshold and emitted null.
                    "fine_hse_type_slug": cls.get("fine_hse_type_slug"),
                    "fine_hse_type_confidence": cls.get("fine_hse_type_confidence") or 0,
                    "fine_hse_type_alternatives": alts_fine,
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
            # Fine-grained AECIS sub-type. Inspector's pick wins when
            # they made a correction; otherwise the AI's Stage 2 pick.
            "final_fine_hse_type_slug": (
                corr.get("fine_hse_type_slug")
                or (cls.get("fine_hse_type_slug") if cls else None)
            ),
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
def confirm(request: Request, photo_id: str, fine_hse_type_slug: str | None = Form(None)):
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
        "user_key": _get_or_set_user_key(request),
    }
    _safe_corrections_insert(payload)
    # Feedback loop: confirmations are high-quality labels — add to retrieval index
    _upsert_embedding_for_correction(photo_id, c["hse_type_slug"], c["location_slug"],
                                     fine_hse_type_slug=fine_hse_type_slug)
    return {"ok": True}


@app.post("/api/photos/{photo_id}/correct")
def correct(
    request: Request,
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
        "user_key": _get_or_set_user_key(request),
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


# ---------- export-blob builder for email ----------

def _build_export_blob(
    fmt: str, batch_id: str | None, limit: int = 5000,
) -> tuple[bytes, str, str]:
    """Return (blob_bytes, mime_type, filename) for the requested format.

    Loads the full export into memory rather than streaming — used by
    the email endpoint which needs the bytes to attach. The existing
    download endpoints keep their streaming code paths since they're
    tuned for large batches.

    Supported formats: 'pdf' | 'zip' | 'csv' | 'json'. Caller has
    already validated the format string.
    """
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=limit, batch_id=batch_id)
    if not rows:
        raise HTTPException(404, "no photos to export")
    tax = app.state.taxonomy or load_taxonomy()
    _enrich_with_labels(rows, tax)

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    if fmt == "csv":
        import csv as _csv, io as _io
        buf = _io.StringIO()
        cols = [
            "original_filename", "uploaded_at",
            "final_fine_hse_type_slug",
            "final_fine_hse_type_label_en", "final_fine_hse_type_label_vn",
            "final_hse_type_slug", "final_hse_type_label_en", "final_hse_type_label_vn",
            "final_location_slug", "final_location_label_en", "final_location_label_vn",
            "reviewed", "review_action", "review_note", "reviewed_at",
            "ai_hse_type_slug", "ai_hse_type_label_en", "ai_hse_type_label_vn",
            "ai_hse_confidence",
            "ai_location_slug", "ai_location_label_en", "ai_location_label_vn",
            "ai_location_confidence",
            "ai_rationale", "ai_model", "sha256",
        ]
        w = _csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return (
            buf.getvalue().encode("utf-8"),
            "text/csv; charset=utf-8",
            f"violations_{stamp}.csv",
        )

    if fmt == "json":
        import json as _json
        from datetime import datetime as _dt
        payload = {
            "exported_at": _dt.now(timezone.utc).isoformat(),
            "count": len(rows),
            "photos": rows,
        }
        return (
            _json.dumps(payload, indent=2, default=str).encode("utf-8"),
            "application/json",
            f"violations_{stamp}.json",
        )

    if fmt == "pdf":
        from webapp.pdf_export import build_violation_pdf
        batch_label = ""
        for r in rows:
            if r.get("batch_label"):
                batch_label = r["batch_label"]
                break
        pdf_bytes = build_violation_pdf(
            rows,
            batch_label=batch_label,
            batch_id=batch_id,
            r2_client=get_r2(),
            r2_bucket=R2_BUCKET,
            project_label="AECIS HSE",
        )
        return (pdf_bytes, "application/pdf", f"violations_{stamp}.pdf")

    if fmt == "zip":
        import csv as _csv, io as _io, zipfile
        r2 = get_r2()
        zip_buf = _io.BytesIO()
        seq_per_class: dict[str, int] = {}
        manifest_buf = _io.StringIO()
        mw = _csv.writer(manifest_buf)
        mw.writerow([
            "exported_filename", "original_filename",
            "final_hse", "final_location",
            "ai_hse", "ai_location", "reviewed",
        ])
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
            for r in rows:
                try:
                    obj = r2.get_object(Bucket=R2_BUCKET, Key=r["storage_key"])
                    body = obj["Body"].read()
                except Exception as e:  # noqa: BLE001
                    log.warning("email-export-zip skip %s: %s",
                                (r.get("sha256") or "?")[:10], e)
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
        return (zip_buf.getvalue(), "application/zip", f"violations_{stamp}.zip")

    raise HTTPException(400, f"unsupported format: {fmt}")


# ---------- email export ----------

# SMTP defaults (Gmail). Override per-deploy via env vars.
_SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
_SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
_SMTP_USER = os.environ.get("SMTP_USER")
_SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
_SMTP_FROM = os.environ.get("SMTP_FROM") or _SMTP_USER  # display addr
_SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1") not in ("0", "false", "no")
# Inline-attachment cap. Gmail rejects >25MB; Outlook 20MB. Stay under
# the lower for safety. Larger exports go via R2 presigned URL.
_EMAIL_INLINE_LIMIT_BYTES = int(os.environ.get("EMAIL_INLINE_LIMIT_BYTES", str(20 * 1024 * 1024)))
_EMAIL_DAILY_PER_USER = int(os.environ.get("EMAIL_DAILY_PER_USER", "5"))


def _email_configured() -> bool:
    """Whether SMTP creds are present. False = endpoint returns
    503 with a clear "not configured" message instead of 500."""
    return bool(_SMTP_USER and _SMTP_PASSWORD)


def _email_count_today(user_key: str) -> int:
    """How many email-export jobs has this user created today (UTC)?
    Used to enforce _EMAIL_DAILY_PER_USER. Returns 0 if the table
    isn't migrated yet (degrades silently)."""
    if not user_key:
        return 0
    from datetime import datetime, timezone
    start_iso = f"{datetime.now(timezone.utc).date().isoformat()}T00:00:00Z"
    try:
        r = (
            get_db().table("email_jobs")
              .select("id", count="exact")
              .eq("user_key", user_key)
              .gte("created_at", start_iso)
              .limit(1).execute()
        )
        return r.count or 0
    except Exception as e:  # noqa: BLE001
        if "email_jobs" in str(e):
            return 0
        log.warning("email rate lookup failed: %s", e)
        return 0


def _send_email_with_attachment(
    to_email: str, subject: str, body: str,
    attachment_bytes: bytes | None,
    attachment_name: str | None,
    attachment_mime: str | None,
) -> None:
    """Single-shot SMTP send. Raises on transport / auth failures so
    the caller can record the error in email_jobs.error and return a
    useful message to the inspector. Supports skipping the attachment
    (None) for the "too large — here's a link" path."""
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = _SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_bytes is not None and attachment_name:
        # Split mime "type/subtype" so EmailMessage can categorize.
        mime = (attachment_mime or "application/octet-stream").split(";", 1)[0]
        if "/" in mime:
            maintype, subtype = mime.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(
            attachment_bytes,
            maintype=maintype, subtype=subtype,
            filename=attachment_name,
        )

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=30) as s:
        s.ehlo()
        if _SMTP_USE_TLS:
            s.starttls()
            s.ehlo()
        s.login(_SMTP_USER, _SMTP_PASSWORD)
        s.send_message(msg)


def _upload_export_to_r2_for_link(
    user_key: str, job_id: str, blob: bytes, filename: str, content_type: str,
) -> str:
    """For exports too big to attach inline. Stash under
    exports/<user>/<job>/<filename> in R2 and return a presigned URL
    valid for 7 days."""
    r2 = get_r2()
    key = f"exports/{user_key}/{job_id}/{filename}"
    r2.put_object(
        Bucket=R2_BUCKET, Key=key, Body=blob, ContentType=content_type,
    )
    url = r2.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=7 * 24 * 60 * 60,
    )
    return url


@app.post("/api/export/email")
@_maybe_limit("_RATE_LIMIT_RETRY")
async def export_email(
    request: Request,
    batch_id: str | None = Form(None),
    fmt: str = Form(..., alias="format"),
    to_email: str = Form(...),
    subject: str | None = Form(None),
    body: str | None = Form(None),
):
    """Generate an export and email it to `to_email`.

    Synchronous: blocks the request for the duration of generation +
    SMTP send (typical: 3-10s for small batches; longer for large PDF/
    ZIP). The frontend shows a spinner during this time.

    Rate limit: per-cookie-id, _EMAIL_DAILY_PER_USER (default 5) per
    UTC day. Counted against the email_jobs table directly.

    Auto-fallback: if the generated blob exceeds
    _EMAIL_INLINE_LIMIT_BYTES (default 20MB), upload to R2 with a
    7-day presigned URL and email the link instead of the attachment.
    """
    if not _email_configured():
        raise HTTPException(503, {
            "error": "email_not_configured",
            "message": "Email export is not enabled on this server. "
                       "Set SMTP_USER + SMTP_PASSWORD env vars to turn it on.",
        })
    fmt = fmt.lower().strip()
    if fmt not in ("pdf", "zip", "csv", "json"):
        raise HTTPException(400, "format must be one of: pdf, zip, csv, json")

    # Cheap email-syntax check. Real validation is the SMTP server's job.
    to_email = (to_email or "").strip()
    if "@" not in to_email or len(to_email) < 5:
        raise HTTPException(400, "to_email looks invalid")

    user_key = _get_or_set_user_key(request)
    used_today = _email_count_today(user_key)
    if used_today >= _EMAIL_DAILY_PER_USER:
        from datetime import datetime, timezone, timedelta
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        raise HTTPException(429, {
            "error": "email_daily_limit_reached",
            "limit": _EMAIL_DAILY_PER_USER,
            "used": used_today,
            "resets_at": f"{tomorrow.isoformat()}T00:00:00Z",
        })

    db = get_db()

    # Insert a pending row so we have an audit trail even if the
    # generation/send blows up halfway through.
    job_id_local: str | None = None
    try:
        ins = db.table("email_jobs").insert({
            "user_key": user_key,
            "batch_id": batch_id,
            "format": fmt,
            "to_email": to_email,
            "subject": subject or None,
            "body": body or None,
            "status": "sending",
            "attempts": 1,
        }).execute()
        if ins.data:
            job_id_local = ins.data[0]["id"]
    except Exception as e:  # noqa: BLE001
        # Pre-migration window: table doesn't exist yet. Continue
        # without the audit row — better to send the email than to
        # 500 the inspector.
        if "email_jobs" not in str(e):
            log.warning("email_jobs insert failed: %s", e)

    # Generate the blob (this is the slow part).
    try:
        blob, mime, filename = _build_export_blob(fmt, batch_id)
    except HTTPException:
        if job_id_local:
            db.table("email_jobs").update(
                {"status": "failed", "error": "no photos to export"}
            ).eq("id", job_id_local).execute()
        raise
    except Exception as e:  # noqa: BLE001
        if job_id_local:
            db.table("email_jobs").update(
                {"status": "failed", "error": f"build: {str(e)[:300]}"}
            ).eq("id", job_id_local).execute()
        raise HTTPException(500, f"export build failed: {e}")

    # Compose subject + body
    final_subject = subject or f"AECIS HSE — Violation Report ({len(blob) // 1024} KB, {fmt.upper()})"
    base_body = body or (
        "Attached: AECIS HSE violation report.\n\n"
        "Generated by hse.aecis.ca."
    )

    # Branch: inline attachment vs link
    use_link = len(blob) > _EMAIL_INLINE_LIMIT_BYTES
    final_status = "sent"
    try:
        if use_link:
            url = _upload_export_to_r2_for_link(
                user_key, job_id_local or "anon", blob, filename, mime,
            )
            link_body = (
                f"{base_body}\n\n"
                f"This export is too large to attach ("
                f"{len(blob) / 1e6:.1f} MB). Download here, link valid 7 days:\n\n"
                f"{url}\n"
            )
            _send_email_with_attachment(
                to_email, final_subject, link_body,
                attachment_bytes=None, attachment_name=None, attachment_mime=None,
            )
            final_status = "too_large_emailed_link"
        else:
            _send_email_with_attachment(
                to_email, final_subject, base_body,
                attachment_bytes=blob,
                attachment_name=filename,
                attachment_mime=mime,
            )
    except Exception as e:  # noqa: BLE001
        if job_id_local:
            db.table("email_jobs").update(
                {"status": "failed", "error": f"send: {str(e)[:300]}",
                 "blob_bytes": len(blob)}
            ).eq("id", job_id_local).execute()
        log.exception("email send failed for job %s", job_id_local)
        raise HTTPException(502, {
            "error": "send_failed",
            "message": str(e)[:200],
        })

    if job_id_local:
        from datetime import datetime, timezone
        db.table("email_jobs").update({
            "status": final_status,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "blob_bytes": len(blob),
        }).eq("id", job_id_local).execute()

    log.info("emailed %s (%d bytes) to %s [job=%s]",
             fmt, len(blob), to_email, job_id_local or "n/a")
    return {
        "ok": True,
        "job_id": job_id_local,
        "to_email": to_email,
        "format": fmt,
        "bytes": len(blob),
        "delivery": "link" if use_link else "attachment",
        "remaining_today": max(0, _EMAIL_DAILY_PER_USER - used_today - 1),
    }


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
    # Column order: the user-facing "violation" comes first as the FINE
    # sub-type (specific AECIS item), then the parent category, then the
    # AI's predictions and inspector metadata. Spreadsheet jocks read
    # left-to-right; the most actionable column belongs leftmost.
    cols = [
        "original_filename", "uploaded_at",
        # Fine sub-type — the headline "Violation" column. Empty when
        # Stage 2 didn't commit (inspector skipped refinement OR AI's
        # Stage 2 confidence was below threshold).
        "final_fine_hse_type_slug",
        "final_fine_hse_type_label_en",
        "final_fine_hse_type_label_vn",
        # Parent / category.
        "final_hse_type_slug", "final_hse_type_label_en", "final_hse_type_label_vn",
        "final_location_slug", "final_location_label_en", "final_location_label_vn",
        "reviewed", "review_action", "review_note", "reviewed_at",
        # AI prediction columns (for traceability / training pipelines).
        "ai_hse_type_slug", "ai_hse_type_label_en", "ai_hse_type_label_vn",
        "ai_hse_confidence",
        "ai_location_slug", "ai_location_label_en", "ai_location_label_vn",
        "ai_location_confidence",
        "ai_rationale", "ai_model",
        "sha256",
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


@app.get("/api/export/pdf")
def export_pdf(limit: int = 5000, batch_id: str | None = None):
    """Render a printable PDF report — cover page + one photo per page
    with the violation type, location, AI vs final label, and inspector
    note. Format AECIS HSE clients expect to hand to safety officers /
    project managers / regulators.

    Photos are downscaled to 1200px wide before embedding so the PDF
    stays a reasonable size even on a 200-photo batch (~30-40MB).
    """
    if not DEFAULT_TENANT_ID:
        raise HTTPException(500, "tenant not configured")
    rows = _collect_export_rows(DEFAULT_TENANT_ID, limit=limit, batch_id=batch_id)
    if not rows:
        raise HTTPException(404, "no photos to export")
    tax = app.state.taxonomy or load_taxonomy()
    _enrich_with_labels(rows, tax)

    # Resolve a friendly batch label for the cover page (latest non-empty
    # wins, mirroring /api/batches behavior). If unset, the builder shows
    # "(unlabeled batch)".
    batch_label = ""
    for r in rows:
        if r.get("batch_label"):
            batch_label = r["batch_label"]
            break

    from webapp.pdf_export import build_violation_pdf
    pdf_bytes = build_violation_pdf(
        rows,
        batch_label=batch_label,
        batch_id=batch_id,
        r2_client=get_r2(),
        r2_bucket=R2_BUCKET,
        project_label="AECIS HSE",
    )

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    filename = f"violations_{stamp}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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


# Public OpenRouter list pricing (USD per 1M tokens). Mirrors
# scripts/evaluate_models.py — keep in sync if rates change. Used by
# the cost ticker to estimate spend; the OpenRouter dashboard is always
# authoritative for billing.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.5":   (3.0, 15.0),
    "anthropic/claude-opus-4.5":     (15.0, 75.0),
    "anthropic/claude-haiku-4.5":    (1.0, 5.0),
    "google/gemini-2.5-flash":       (0.30, 2.50),
    "google/gemini-2.5-pro":         (1.25, 5.0),
    "openai/gpt-4o":                 (2.5, 10.0),
    "openai/gpt-4o-mini":            (0.15, 0.60),
}


def _price_for_model(model_id: str | None) -> tuple[float, float]:
    """Return (in_rate, out_rate) per 1M tokens. Defaults to Sonnet
    pricing if the model isn't in the table — that's the conservative
    over-estimate, so the ticker errs on the side of "we're spending
    more than we think" rather than the inverse."""
    if not model_id:
        return (3.0, 15.0)
    # classifications.model is stored as "openrouter:google/gemini-2.5-flash"
    # — strip the provider prefix to match _MODEL_PRICING keys.
    bare = model_id.split(":", 1)[-1] if ":" in model_id else model_id
    return _MODEL_PRICING.get(bare, (3.0, 15.0))


@app.get("/api/usage/me")
def api_usage_me(request: Request):
    """Per-user view of today's quota: how many photos used, what's
    the limit, when does the bucket reset. Read by the footer to
    render the N/30 indicator without exposing global cost numbers
    (those are admin-only and live at /api/usage/today)."""
    from datetime import datetime, timezone, timedelta
    user_key = _get_or_set_user_key(request)
    used = _quota_today_used(user_key)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    body = {
        "used": used,
        "limit": _DAILY_QUOTA,
        "remaining": max(0, _DAILY_QUOTA - used),
        "resets_at": f"{tomorrow.isoformat()}T00:00:00Z",
    }
    resp = JSONResponse(content=body)
    _attach_user_cookie(resp, user_key)
    return resp


# ---------- phase 4: hse-class proposals (user side) ----------

@app.post("/api/proposals")
async def api_proposals_create(
    request: Request,
    parent_slug: str = Form(...),
    label_en: str = Form(...),
    label_vn: str = Form(...),
    description: str | None = Form(None),
    example_photo_id: str | None = Form(None),
):
    """Inspector proposes a new fine HSE sub-type.

    Validates: parent_slug exists in fine_hse_types_by_parent.json,
    label_en + label_vn are non-empty + reasonable length, daily cap
    not exceeded, generated slug isn't already in the live taxonomy
    or pending queue.

    Returns the proposal row (with the placeholder 'pending:<slug>'
    that the UI can stash on the correction so the user keeps working
    while the admin reviews).
    """
    user_key = _get_or_set_user_key(request)

    # Hot-load fine-types if app.state hasn't seen one yet (e.g. fresh
    # boot, no /). Same lazy load as the / route.
    if not hasattr(app.state, "fine_types"):
        fine_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
        try:
            app.state.fine_types = json.loads(
                fine_path.read_text(encoding="utf-8")
            ).get("parents", {})
        except Exception:  # noqa: BLE001
            app.state.fine_types = {}

    parents: dict = app.state.fine_types or {}
    if parent_slug not in parents:
        raise HTTPException(400, f"unknown parent_slug: {parent_slug}")

    label_en = (label_en or "").strip()
    label_vn = (label_vn or "").strip()
    description = (description or "").strip() or None
    if not label_en or not label_vn:
        raise HTTPException(400, "label_en and label_vn required")
    if len(label_en) > 200 or len(label_vn) > 200:
        raise HTTPException(400, "label too long (max 200 chars)")
    if description and len(description) > 1000:
        raise HTTPException(400, "description too long (max 1000 chars)")

    # Per-cookie throttle. Keep this check after cheap validation so the
    # error surface for a typo isn't "you hit the daily cap".
    used_today = _proposal_count_today(user_key)
    if used_today >= _PROPOSAL_DAILY_PER_USER:
        raise HTTPException(
            429,
            detail={
                "error": "proposal_daily_limit",
                "limit": _PROPOSAL_DAILY_PER_USER,
                "used": used_today,
            },
        )

    # Don't let an inspector propose a slug that already exists under
    # the chosen parent. Also reject same-label-EN to avoid near-dups.
    existing = parents.get(parent_slug, [])
    base = _normalize_proposed_slug(label_en)
    label_lower = label_en.lower()
    for ex in existing:
        if (ex.get("slug") or "").lower() == base.lower():
            raise HTTPException(409, "slug already exists under this parent")
        if (ex.get("label_en") or "").strip().lower() == label_lower:
            raise HTTPException(409, "label already exists under this parent")

    proposed_slug = f"pending:{_unique_proposed_slug(base)}"

    # Tolerate example_photo_id being a non-UUID (e.g. blank or a UI
    # sentinel) — store NULL rather than failing the insert.
    photo_id_clean: str | None = None
    if example_photo_id:
        try:
            uuid.UUID(example_photo_id)
            photo_id_clean = example_photo_id
        except (ValueError, TypeError):
            photo_id_clean = None

    db = get_db()
    try:
        row = (
            db.table("hse_class_proposals").insert({
                "proposed_by_user_key": user_key,
                "parent_slug": parent_slug,
                "proposed_slug": proposed_slug,
                "label_en": label_en,
                "label_vn": label_vn,
                "description": description,
                "example_photo_id": photo_id_clean,
                "status": "pending",
            }).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        if "hse_class_proposals" in str(e):
            raise HTTPException(503, "proposals not configured (run migration)")
        log.warning("proposal insert failed: %s", e)
        raise HTTPException(500, "proposal insert failed")

    rec = row[0] if row else {}
    body = {
        "id": rec.get("id"),
        "proposed_slug": proposed_slug,   # 'pending:<slug>' — UI stores this on the correction
        "parent_slug": parent_slug,
        "label_en": label_en,
        "label_vn": label_vn,
        "status": "pending",
        "used_today": used_today + 1,
        "limit": _PROPOSAL_DAILY_PER_USER,
    }
    resp = JSONResponse(content=body)
    _attach_user_cookie(resp, user_key)
    return resp


@app.get("/api/proposals/decisions")
def api_proposals_decisions(request: Request):
    """Pull decided proposals for this cookie that the user hasn't
    been told about yet. The first call returns up to 10 rows + flips
    notified_user=TRUE on those rows so the toast is one-shot.

    Called once per page load by index.html. Returns an empty list
    when the user has no pending notifications, so the call is cheap
    + safe to fire on every visit.
    """
    user_key = _get_or_set_user_key(request)
    db = get_db()
    try:
        rows = (
            db.table("hse_class_proposals")
              .select("id, parent_slug, label_en, label_vn, status, "
                      "approved_slug, reviewer_note, reviewed_at")
              .eq("proposed_by_user_key", user_key)
              .neq("status", "pending")
              .eq("notified_user", False)
              .order("reviewed_at", desc=True)
              .limit(10).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        if "hse_class_proposals" in str(e):
            return JSONResponse(content={"decisions": []})
        log.warning("proposal decisions lookup failed: %s", e)
        return JSONResponse(content={"decisions": []})

    if rows:
        ids = [r["id"] for r in rows]
        try:
            db.table("hse_class_proposals").update({
                "notified_user": True,
            }).in_("id", ids).execute()
        except Exception as e:  # noqa: BLE001
            log.warning("proposal notify-mark failed: %s", e)
            # Non-fatal — the user just sees the toast again next visit.

    resp = JSONResponse(content={"decisions": rows})
    _attach_user_cookie(resp, user_key)
    return resp


@app.get("/api/proposals/mine")
def api_proposals_mine(request: Request):
    """List THIS cookie's recent proposals (any status). Used by the
    Propose modal's 'your recent submissions' section so the inspector
    can see what they've already sent + the daily cap usage."""
    user_key = _get_or_set_user_key(request)
    db = get_db()
    try:
        rows = (
            db.table("hse_class_proposals")
              .select("id, parent_slug, proposed_slug, label_en, label_vn, "
                      "status, approved_slug, reviewer_note, "
                      "created_at, reviewed_at")
              .eq("proposed_by_user_key", user_key)
              .order("created_at", desc=True)
              .limit(20).execute().data or []
        )
    except Exception as e:  # noqa: BLE001
        if "hse_class_proposals" in str(e):
            return JSONResponse(content={
                "proposals": [],
                "used_today": 0,
                "limit": _PROPOSAL_DAILY_PER_USER,
            })
        log.warning("proposals/mine failed: %s", e)
        rows = []

    used_today = _proposal_count_today(user_key)
    body = {
        "proposals": rows,
        "used_today": used_today,
        "limit": _PROPOSAL_DAILY_PER_USER,
    }
    resp = JSONResponse(content=body)
    _attach_user_cookie(resp, user_key)
    return resp


@app.get("/api/usage/today")
def api_usage_today(request: Request):
    """Estimate today's OpenRouter spend by summing tokens × public list
    rate per model from the classifications table (UTC day boundary).

    Scope:
      - Admins: see global spend (all users, all batches) — same shape
        as before. The admin /admin panel uses this for the cost tile.
      - Signed-in non-admins: see their own spend, joined through
        photos.user_id. Lets each inspector watch their personal
        consumption without leaking everyone else's.
      - Pre-auth (only reachable when AUTH_REQUIRED=0 in dev): falls
        through to the legacy global view since there's no user_id to
        scope by.

    Cheap enough to call on every page load — bounded by today's row
    count, typically <500 rows.
    """
    if not DEFAULT_TENANT_ID:
        return {"date": "", "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "estimated_cost_usd": 0.0,
                "by_model": {}, "scope": "none"}
    db = get_db()
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    start_iso = f"{today.isoformat()}T00:00:00Z"

    # Scope decision: admin gets global, regular signed-in user gets
    # personal (joined via photos.user_id), anonymous (only possible
    # when the gate is disabled) gets the legacy global view.
    user = _get_session_user(request)
    is_admin = bool(user and user.get("is_admin")) or _admin_authed(request)
    user_id = user["id"] if user else None
    scope = "global" if is_admin else ("personal" if user_id else "global")

    try:
        if scope == "personal" and user_id:
            # Pull this user's photo ids for today, then classifications
            # filtered by those. Two short queries beat fetching all
            # classifications and filtering in Python.
            my_photo_ids_rows = (
                db.table("photos").select("id")
                  .eq("user_id", user_id)
                  .gte("uploaded_at", start_iso)
                  .limit(5000).execute().data or []
            )
            my_photo_ids = [r["id"] for r in my_photo_ids_rows]
            if not my_photo_ids:
                rows = []
            else:
                rows = (
                    db.table("classifications")
                      .select("model, input_tokens, output_tokens")
                      .in_("photo_id", my_photo_ids)
                      .gte("created_at", start_iso)
                      .limit(10000)
                      .execute().data or []
                )
        else:
            rows = (
                db.table("classifications")
                  .select("model, input_tokens, output_tokens")
                  .gte("created_at", start_iso)
                  .limit(10000)
                  .execute().data or []
            )
    except Exception as e:  # noqa: BLE001
        log.warning("usage query failed: %s", e)
        return {"date": today.isoformat(), "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "estimated_cost_usd": 0.0,
                "by_model": {}, "scope": scope, "error": str(e)[:120]}

    total_in = 0
    total_out = 0
    cost = 0.0
    by_model: dict[str, dict] = {}
    for r in rows:
        in_tok = int(r.get("input_tokens") or 0)
        out_tok = int(r.get("output_tokens") or 0)
        m = r.get("model") or "unknown"
        in_rate, out_rate = _price_for_model(m)
        c = (in_tok / 1e6) * in_rate + (out_tok / 1e6) * out_rate
        total_in += in_tok
        total_out += out_tok
        cost += c
        bm = by_model.setdefault(m, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        bm["calls"] += 1
        bm["in"] += in_tok
        bm["out"] += out_tok
        bm["cost"] += c
    for bm in by_model.values():
        bm["cost"] = round(bm["cost"], 4)
    return {
        "date": today.isoformat(),
        "calls": len(rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "estimated_cost_usd": round(cost, 4),
        "by_model": by_model,
        # Frontend uses this to label the ticker tooltip ("you today" vs
        # "everyone today"). Admins see the global figure even though
        # they're authenticated, so the label tracks the actual scope.
        "scope": scope,
    }


# ---------- phase 5: google oauth routes ----------

@app.get("/auth/login/google")
def auth_login_google(request: Request, next: str = "/"):
    """Kick off the Google OAuth flow.

    Generates a CSRF state token, stows it in a short-lived cookie,
    and redirects the browser to Google's authorize endpoint. The
    `next` query param survives the round-trip (encoded into state)
    so the callback can drop the user back where they came from.
    """
    if not _oauth_enabled():
        raise HTTPException(
            503,
            "Google sign-in is not configured (set GOOGLE_CLIENT_ID / "
            "GOOGLE_CLIENT_SECRET env vars).",
        )
    # State token format: <random>.<safe_next> — Google echoes the
    # whole string back; we split on '.' to verify random + recover next.
    raw = secrets.token_urlsafe(32)
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    # base64-ish safe chars only; restore on callback.
    encoded_next = uuid.uuid5(uuid.NAMESPACE_URL, safe_next).hex
    # We store the actual `next` separately keyed by the random part so
    # an attacker can't substitute a different next URL in the state.
    state = raw

    resp = RedirectResponse(_google_authorize_url(state), status_code=303)
    resp.set_cookie(
        _OAUTH_STATE_COOKIE_NAME,
        value=f"{raw}|{safe_next}",
        max_age=600,   # 10 min — long enough for the round-trip
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )
    return resp


@app.get("/auth/callback/google")
async def auth_callback_google(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """Google's redirect target. Verifies the state cookie, exchanges
    the auth code for tokens, fetches the user profile, upserts the
    users row, creates a session, retro-attributes pre-auth cookie
    data, and redirects back to the app.

    On any failure, drops a small inline HTML page with a 'Try again'
    link rather than a JSON 4xx — this is a top-level browser navigation,
    not a fetch from JS.
    """
    if not _oauth_enabled():
        raise HTTPException(503, "google oauth not configured")
    if error:
        return HTMLResponse(
            f"<h1>Sign-in cancelled</h1><p>Google reported: <code>{error}</code></p>"
            "<p><a href='/'>← Back to app</a></p>", status_code=400)
    if not code or not state:
        raise HTTPException(400, "missing code/state")

    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE_NAME) or ""
    raw, _, safe_next = cookie_state.partition("|")
    if not raw or not secrets.compare_digest(raw, state):
        log.warning("oauth state mismatch")
        return HTMLResponse(
            "<h1>Sign-in failed</h1><p>State token mismatch — try again.</p>"
            "<p><a href='/auth/login/google'>← Try again</a></p>", status_code=400)

    try:
        token_data = await _google_exchange_code(code)
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(401, "no access token in google response")
        profile = await _google_fetch_userinfo(access_token)
    except HTTPException:
        return HTMLResponse(
            "<h1>Sign-in failed</h1><p>Google rejected the auth code. Try again.</p>"
            "<p><a href='/auth/login/google'>← Try again</a></p>", status_code=401)

    # Persist user + session. Wrap so a transient DB failure doesn't
    # leave the user stranded — they get a retry link.
    try:
        user = _upsert_user("google", profile)
        session_token = _create_session(user["id"], request)
    except Exception as e:  # noqa: BLE001
        log.error("oauth callback persistence failed: %s", e)
        return HTMLResponse(
            "<h1>Sign-in failed</h1>"
            "<p>Couldn't save your session. The sysadmin has been notified.</p>"
            "<p><a href='/'>← Back to app</a></p>", status_code=500)

    # Retro-attribute the pre-auth cookie's data to this user. Best-effort.
    pre_auth_key = request.cookies.get(_USER_COOKIE_NAME)
    if pre_auth_key:
        _retro_attribute_user(user["id"], pre_auth_key)

    target = safe_next if safe_next and safe_next.startswith("/") and not safe_next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    _attach_session_cookie(resp, session_token)
    resp.delete_cookie(_OAUTH_STATE_COOKIE_NAME)
    return resp


@app.post("/auth/logout")
def auth_logout(request: Request):
    """Drop the session row + clear the cookie. The vai_uid cookie is
    NOT cleared — the user can keep working anonymously. Returns a
    303 redirect for HTML-form submissions; JS callers can ignore the
    body."""
    token = request.cookies.get(_SESSION_COOKIE_NAME)
    if token:
        try:
            get_db().table("sessions").delete().eq("token", token).execute()
        except Exception as e:  # noqa: BLE001
            log.warning("session delete failed: %s", e)
    resp = RedirectResponse("/", status_code=303)
    _clear_session_cookie(resp)
    return resp


@app.get("/auth/me")
def auth_me(request: Request):
    """Return the current signed-in user (or {authed: false}). Read by
    the frontend to render the user pill in the header. Cheap — single
    sessions+users join, indexed on token. Safe to call on every page
    load."""
    user = _get_session_user(request)
    if not user:
        return JSONResponse(content={
            "authed": False,
            "google_enabled": _oauth_enabled(),
            "azure_enabled": _azure_oauth_enabled(),
            "auth_required": _AUTH_REQUIRED,
            "login_url": "/auth/signin",
        })
    return JSONResponse(content={
        "authed": True,
        "id": user["id"],
        "email": user.get("email"),
        "name": user.get("name"),
        "picture_url": user.get("picture_url"),
        "is_admin": user.get("is_admin", False),
        "provider": user.get("provider"),
    })


# ---------- phase 7: azure oauth routes ----------

@app.get("/auth/login/azure")
def auth_login_azure(request: Request, next: str = "/"):
    """Kick off the Azure Entra ID OAuth flow. Mirrors /auth/login/google
    structure — generates a state token, stows it in a short-lived
    cookie, redirects to Microsoft's authorize endpoint."""
    if not _azure_oauth_enabled():
        raise HTTPException(
            503,
            "Microsoft sign-in is not configured (set AZURE_CLIENT_ID / "
            "AZURE_CLIENT_SECRET env vars).",
        )
    raw = secrets.token_urlsafe(32)
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    state = raw

    resp = RedirectResponse(_azure_authorize_url(state), status_code=303)
    resp.set_cookie(
        _OAUTH_STATE_COOKIE_NAME,
        value=f"{raw}|{safe_next}",
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )
    return resp


@app.get("/auth/callback/azure")
async def auth_callback_azure(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Microsoft's redirect target. Same shape as the Google callback —
    state check, code exchange, profile fetch, user upsert, session,
    retro-attribute, redirect home."""
    if not _azure_oauth_enabled():
        raise HTTPException(503, "azure oauth not configured")
    if error:
        msg = error_description or error
        return HTMLResponse(
            f"<h1>Sign-in cancelled</h1><p>Microsoft reported: <code>{msg}</code></p>"
            "<p><a href='/auth/signin'>← Back</a></p>", status_code=400)
    if not code or not state:
        raise HTTPException(400, "missing code/state")

    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE_NAME) or ""
    raw, _, safe_next = cookie_state.partition("|")
    if not raw or not secrets.compare_digest(raw, state):
        log.warning("azure oauth state mismatch")
        return HTMLResponse(
            "<h1>Sign-in failed</h1><p>State token mismatch — try again.</p>"
            "<p><a href='/auth/login/azure'>← Try again</a></p>", status_code=400)

    try:
        token_data = await _azure_exchange_code(code)
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(401, "no access token in azure response")
        profile = await _azure_fetch_userinfo(access_token)
    except HTTPException:
        return HTMLResponse(
            "<h1>Sign-in failed</h1><p>Microsoft rejected the auth code. Try again.</p>"
            "<p><a href='/auth/login/azure'>← Try again</a></p>", status_code=401)

    try:
        user = _upsert_user("azure", profile)
        session_token = _create_session(user["id"], request)
    except Exception as e:  # noqa: BLE001
        log.error("azure callback persistence failed: %s", e)
        return HTMLResponse(
            "<h1>Sign-in failed</h1>"
            "<p>Couldn't save your session. The sysadmin has been notified.</p>"
            "<p><a href='/'>← Back to app</a></p>", status_code=500)

    pre_auth_key = request.cookies.get(_USER_COOKIE_NAME)
    if pre_auth_key:
        _retro_attribute_user(user["id"], pre_auth_key)

    target = safe_next if safe_next and safe_next.startswith("/") and not safe_next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    _attach_session_cookie(resp, session_token)
    resp.delete_cookie(_OAUTH_STATE_COOKIE_NAME)
    return resp


# ---------- phase 7: sign-in chooser page ----------

@app.get("/auth/signin", response_class=HTMLResponse)
def auth_signin(request: Request, next: str = "/"):
    """Provider-chooser landing page. Anonymous users hit this when the
    auth gate redirects them. Shows whichever providers are configured;
    a single-provider deployment gets a single button. Bilingual."""
    # If they already have a valid session, no need to choose — bounce.
    if _get_session_user(request):
        return RedirectResponse(next or "/", status_code=303)

    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    from urllib.parse import quote
    n = quote(safe_next, safe="")

    google_btn = ""
    if _oauth_enabled():
        google_btn = f"""
        <a href="/auth/login/google?next={n}" class="provider-btn google">
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
          <span data-locale="en">Continue with Google</span><span data-locale="vn">Tiếp tục với Google</span>
        </a>"""
    azure_btn = ""
    if _azure_oauth_enabled():
        azure_btn = f"""
        <a href="/auth/login/azure?next={n}" class="provider-btn microsoft">
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <rect x="2"  y="2"  width="9" height="9" fill="#F25022"/>
            <rect x="13" y="2"  width="9" height="9" fill="#7FBA00"/>
            <rect x="2"  y="13" width="9" height="9" fill="#00A4EF"/>
            <rect x="13" y="13" width="9" height="9" fill="#FFB900"/>
          </svg>
          <span data-locale="en">Continue with Microsoft</span><span data-locale="vn">Tiếp tục với Microsoft</span>
        </a>"""

    none_msg = ""
    if not google_btn and not azure_btn:
        none_msg = (
            "<div class='no-providers'>"
            "Sign-in is not configured on this server. "
            "Set <code>GOOGLE_CLIENT_ID</code> or <code>AZURE_CLIENT_ID</code> "
            "in the environment.</div>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Sign in · Violation AI</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#000000">
<link rel="manifest" href="/static/manifest.json">
<link rel="apple-touch-icon" href="/static/icons/icon-180.png">
<style>
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, sans-serif;
       margin: 0; background: #000000; color: #ffffff; min-height: 100vh;
       -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
       display: flex; align-items: center; justify-content: center; padding: 16px;
       letter-spacing: -0.005em; }}
.card {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px; padding: 36px 30px; max-width: 400px; width: 100%;
        text-align: center; backdrop-filter: blur(8px); }}
.brand {{ display: inline-flex; align-items: center; gap: 10px;
         font-size: 17px; font-weight: 600; margin-bottom: 6px; color: #ffffff; }}
.brand .check {{ width: 28px; height: 28px; border: 1px solid rgba(255,255,255,0.4);
                border-radius: 999px; display: grid; place-items: center; color: #fff; }}
.brand .check svg {{ width: 14px; height: 14px; }}
.brand .accent {{ color: rgba(255,255,255,0.45); font-weight: 400; }}
h1 {{ font-size: 24px; margin: 14px 0 8px; font-weight: 600; letter-spacing: -0.025em; color: #ffffff; }}
.sub {{ color: rgba(255,255,255,0.55); font-size: 13px; margin: 0 0 28px; line-height: 1.5; }}
.providers {{ display: flex; flex-direction: column; gap: 10px; }}
.provider-btn {{ display: flex; align-items: center; justify-content: center; gap: 12px;
                padding: 13px 18px; border-radius: 999px; text-decoration: none;
                font-weight: 600; font-size: 14px; letter-spacing: 0.005em;
                transition: transform .04s, background .15s, border-color .15s, box-shadow .15s; }}
.provider-btn:active {{ transform: scale(0.98); }}
.provider-btn.google {{ background: #ffffff; color: #0f172a; border: 1px solid #ffffff; }}
.provider-btn.google:hover {{ background: rgba(255,255,255,0.92); box-shadow: 0 4px 16px rgba(255,255,255,0.08); }}
.provider-btn.microsoft {{ background: rgba(255,255,255,0.06); color: #ffffff;
                           border: 1px solid rgba(255,255,255,0.18); }}
.provider-btn.microsoft:hover {{ background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.28); }}
.foot {{ color: rgba(255,255,255,0.35); font-size: 11px; margin-top: 28px; line-height: 1.5; }}
.foot a {{ color: rgba(255,255,255,0.7); text-decoration: underline; text-decoration-color: rgba(255,255,255,0.25); }}
.locale-toggle {{ position: absolute; top: 16px; right: 16px; display: flex; gap: 0;
                 border: 1px solid rgba(255,255,255,0.18); border-radius: 999px; overflow: hidden; }}
.locale-toggle button {{ font: inherit; background: transparent; border: 0; cursor: pointer;
                        padding: 5px 12px; color: rgba(255,255,255,0.55); font-weight: 600;
                        font-size: 11px; letter-spacing: 0.04em; transition: background .15s, color .15s; }}
.locale-toggle button:hover {{ color: #ffffff; }}
.locale-toggle button.active {{ background: #ffffff; color: #000000; }}
.back-link {{ position: absolute; top: 16px; left: 16px; color: rgba(255,255,255,0.45);
             text-decoration: none; font-size: 12px; letter-spacing: 0.02em;
             padding: 5px 10px; border-radius: 999px; border: 1px solid transparent;
             transition: color .15s, border-color .15s; }}
.back-link:hover {{ color: #ffffff; border-color: rgba(255,255,255,0.18); }}
.no-providers {{ background: rgba(239,68,68,0.06); color: #fca5a5;
                 border: 1px solid rgba(239,68,68,0.25);
                 padding: 14px; border-radius: 10px; font-size: 12px; text-align: left;
                 line-height: 1.5; }}
.no-providers code {{ background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px;
                      font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: #fde68a; }}
[data-locale]:not(.show) {{ display: none; }}
</style></head><body>
<a class="back-link" href="/">← <span data-locale="en">Back</span><span data-locale="vn">Quay lại</span></a>
<div class="locale-toggle">
  <button id="loc-en" class="active" type="button">EN</button>
  <button id="loc-vn" type="button">VN</button>
</div>
<div class="card">
  <div class="brand">
    <span class="check"><svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg></span>
    <span>Violation <span class="accent">/ AI</span></span>
  </div>
  <h1><span data-locale="en">Sign in to continue</span><span data-locale="vn">Đăng nhập để tiếp tục</span></h1>
  <p class="sub">
    <span data-locale="en">AECIS HSE inspection tool. Pick your provider.</span>
    <span data-locale="vn">Công cụ kiểm tra HSE của AECIS. Chọn nhà cung cấp.</span>
  </p>
  <div class="providers">
    {google_btn}
    {azure_btn}
    {none_msg}
  </div>
  <p class="foot">
    <span data-locale="en">By signing in, you agree to use this tool for AECIS HSE inspection only.</span>
    <span data-locale="vn">Bằng cách đăng nhập, bạn đồng ý chỉ sử dụng công cụ này cho việc kiểm tra HSE của AECIS.</span>
  </p>
</div>
<script>
  // Lightweight EN/VN locale toggle. Persists in localStorage so the
  // sign-in page remembers the user's preference between visits.
  (function () {{
    const saved = localStorage.getItem("vai-lang") || "en";
    const enBtn = document.getElementById("loc-en");
    const vnBtn = document.getElementById("loc-vn");
    function apply(loc) {{
      document.querySelectorAll("[data-locale]").forEach(el => {{
        el.classList.toggle("show", el.dataset.locale === loc);
      }});
      enBtn.classList.toggle("active", loc === "en");
      vnBtn.classList.toggle("active", loc === "vn");
      localStorage.setItem("vai-lang", loc);
    }}
    enBtn.addEventListener("click", () => apply("en"));
    vnBtn.addEventListener("click", () => apply("vn"));
    apply(saved);
  }})();
</script>
</body></html>"""
    return HTMLResponse(html)


# ---------- admin (env-var-password gated; user is_admin also recognised in phase 5) ----------

@app.post("/admin/login")
def admin_login(request: Request, password: str = Form(...)):
    """Sets the admin cookie if the env-var password matches. The
    cookie value IS the password — simple and short-lived (12h).
    When real auth ships, this route gets replaced by Azure AD."""
    if not _ADMIN_PASSWORD:
        raise HTTPException(503, "admin disabled (set ADMIN_PASSWORD env)")
    if password != _ADMIN_PASSWORD:
        raise HTTPException(401, "wrong password")
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(
        _ADMIN_COOKIE_NAME,
        value=_ADMIN_PASSWORD,
        max_age=_ADMIN_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "1") not in ("0", "false", "no"),
    )
    return resp


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request):
    """Tiny HTML form for the admin password. Inline, no template
    needed — admin UX is intentionally sparse."""
    if _admin_authed(request):
        return RedirectResponse("/admin", status_code=303)
    if not _ADMIN_PASSWORD:
        return HTMLResponse(
            "<h1>admin disabled</h1><p>Set <code>ADMIN_PASSWORD</code> env "
            "var on the server to enable.</p>", status_code=503)
    return HTMLResponse("""
<!doctype html><html><head><meta charset=utf-8><title>Admin login</title>
<style>body{font:14px/1.4 system-ui;max-width:360px;margin:80px auto;padding:24px;background:#f8fafc}
h1{font-size:18px;margin:0 0 16px}
form{display:flex;gap:8px}
input{flex:1;padding:8px;border:1px solid #cbd5e1;border-radius:6px}
button{background:#0f172a;color:#fff;border:0;padding:8px 16px;border-radius:6px;cursor:pointer}
</style></head><body>
<h1>Admin login</h1>
<form method=POST action=/admin/login>
  <input type=password name=password placeholder=password autofocus required>
  <button type=submit>Sign in</button>
</form>
</body></html>""")


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request, days: int = 30, status: str = "pending"):
    """Unified admin control panel — stats + HSE-class proposal review
    on a single page. Replaces the old /admin/stats and /admin/proposals
    pages (both still 303 here for backwards compat).

    Layout: stats section first (compact), proposals queue below with
    the same status-filter tabs that /admin/proposals had. The proposals
    POST handlers (approve/reject/duplicate) redirect back here with
    the #proposals anchor so the admin lands on the queue, not the top.
    """
    if not _admin_authed(request):
        return RedirectResponse("/admin/login", status_code=303)

    # Both sections compute independently — proposals failure shouldn't
    # blank out stats. _build_proposals_inner already degrades gracefully
    # to its own error block when the table is missing.
    try:
        stats_html = _build_stats_inner(days)
    except Exception as e:  # noqa: BLE001
        log.warning("admin stats render failed: %s", e)
        stats_html = (
            "<section id='stats' class='adm-section'>"
            "<h2>Activity overview</h2>"
            f"<div class='adm-error'>Stats render failed: {e}</div></section>"
        )

    try:
        proposals_html = _build_proposals_inner(status)
    except Exception as e:  # noqa: BLE001
        log.warning("admin proposals render failed: %s", e)
        proposals_html = (
            "<section id='proposals' class='adm-section'>"
            "<h2>HSE class proposals</h2>"
            f"<div class='adm-error'>Proposals render failed: {e}</div></section>"
        )

    html = f"""<!doctype html>
<html><head><meta charset=utf-8>
<title>Admin · Violation AI</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root {{ --emerald: #10b981; --emerald-dark: #059669; --slate-900: #0f172a;
         --slate-700: #334155; --slate-500: #64748b; --slate-200: #e2e8f0;
         --slate-100: #f1f5f9; --slate-50: #f8fafc; --slate-400: #94a3b8;
         --rose: #ef4444; --indigo: #6366f1; --amber: #f59e0b; }}
* {{ box-sizing: border-box; }}
body {{ font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; padding: 0; background: var(--slate-50); color: var(--slate-900); }}
.adm-shell {{ max-width: 1200px; margin: 0 auto; padding: 0 16px 32px; }}
.adm-header {{ display: flex; align-items: center; justify-content: space-between;
              padding: 14px 16px; background: #fff; border-bottom: 1px solid var(--slate-200);
              position: sticky; top: 0; z-index: 5; }}
.adm-header h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
.adm-header h1 .accent {{ color: var(--emerald); }}
.adm-header-nav {{ display: flex; gap: 4px; }}
.adm-header-nav a {{ font-size: 12px; color: var(--slate-500); text-decoration: none;
                    padding: 4px 10px; border-radius: 6px; }}
.adm-header-nav a:hover {{ color: var(--slate-900); background: var(--slate-100); }}
.adm-jump {{ display: flex; gap: 4px; padding: 12px 16px; background: #fff;
            border-bottom: 1px solid var(--slate-200); margin: 0 0 16px; position: sticky;
            top: 47px; z-index: 4; max-width: 1200px; margin-left: auto; margin-right: auto; }}
.adm-jump a {{ font-size: 12px; color: var(--slate-500); text-decoration: none;
              padding: 6px 12px; border-radius: 999px; background: var(--slate-100); }}
.adm-jump a:hover {{ color: var(--slate-900); background: var(--slate-200); }}
.adm-section {{ background: #fff; border: 1px solid var(--slate-200); border-radius: 12px;
               padding: 18px 20px; margin: 16px 0; scroll-margin-top: 100px; }}
.adm-section-head {{ display: flex; align-items: baseline; gap: 12px; margin: 0 0 14px;
                    border-bottom: 1px solid var(--slate-100); padding-bottom: 10px; }}
.adm-section h2 {{ font-size: 15px; margin: 0; font-weight: 600; }}
.adm-section h3 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
                  color: var(--slate-500); margin: 22px 0 8px; font-weight: 600; }}
.adm-window {{ font-size: 12px; color: var(--slate-400); }}
.adm-error {{ padding: 12px; border-radius: 8px; background: #fef2f2; color: #991b1b;
             border: 1px solid #fecaca; font-size: 12px; }}
.adm-empty {{ text-align: center; color: var(--slate-400); padding: 12px; }}

.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }}
.stat {{ background: var(--slate-50); border: 1px solid var(--slate-100); border-radius: 8px; padding: 12px; }}
.k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--slate-500); }}
.v {{ font-size: 18px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }}

.adm-table-wrap {{ overflow-x: auto; }}
.stats-tbl, .props-tbl {{ border-collapse: collapse; width: 100%; }}
.stats-tbl th, .stats-tbl td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--slate-100);
                                font-variant-numeric: tabular-nums; font-size: 12px; }}
.props-tbl th, .props-tbl td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--slate-100);
                                vertical-align: top; font-size: 12px; }}
.stats-tbl th, .props-tbl th {{ background: var(--slate-50); color: var(--slate-700);
                                font-size: 11px; text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }}
.stats-tbl tr:last-child td, .props-tbl tr:last-child td {{ border-bottom: 0; }}
.mono {{ font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--slate-700); }}

.adm-tabs {{ display: flex; gap: 4px; margin: 0 0 12px; border-bottom: 1px solid var(--slate-200); flex-wrap: wrap; }}
.adm-tabs .tab {{ padding: 8px 14px; color: var(--slate-500); text-decoration: none;
                 border-bottom: 2px solid transparent; font-size: 13px; text-transform: capitalize; }}
.adm-tabs .tab.active {{ color: var(--slate-900); border-bottom-color: var(--emerald); font-weight: 600; }}
.adm-tabs .tab:hover {{ color: var(--slate-900); }}

.btn {{ border: 0; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; }}
.btn.approve {{ background: var(--emerald); color: #fff; }}
.btn.approve:hover {{ background: var(--emerald-dark); }}
.btn.reject {{ background: #fff; color: var(--rose); border: 1px solid var(--rose); }}
.btn.reject:hover {{ background: #fef2f2; }}
.btn.dup {{ background: #fff; color: var(--indigo); border: 1px solid var(--indigo); }}
.btn.dup:hover {{ background: #eef2ff; }}
code {{ font-family: ui-monospace, Menlo, monospace; background: var(--slate-100);
       padding: 1px 6px; border-radius: 4px; font-size: 11px; }}

@media (max-width: 640px) {{
  .adm-header h1 {{ font-size: 15px; }}
  .adm-section {{ padding: 14px; }}
}}
</style></head><body>
<header class="adm-header">
  <h1>Violation <span class="accent">AI</span> · Admin</h1>
  <nav class="adm-header-nav">
    <a href="/">← App</a>
    <a href="/api/usage/today" target="_blank" rel="noopener">Cost JSON</a>
  </nav>
</header>
<nav class="adm-jump">
  <a href="#stats">Activity</a>
  <a href="#proposals">Proposals</a>
</nav>
<div class="adm-shell">
{stats_html}
{proposals_html}
</div>
</body></html>"""
    return HTMLResponse(html)


def _build_stats_inner(days: int) -> str:
    """Compute the stats section's inner HTML for the unified /admin
    panel. Returns just the body content — the page shell + styles
    live in admin_panel. Same lazy roll-up as before — at our scale
    (a few thousand rows/day) the daily_user_stats materialized view
    is reserved for when scale actually demands it.
    """
    from datetime import datetime, timezone, timedelta
    db = get_db()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    start_iso = f"{start.isoformat()}T00:00:00Z"

    # Pull recent photos + corrections with user_key, group in Python.
    photos = (
        db.table("photos").select("id, user_key, uploaded_at, batch_id")
          .gte("uploaded_at", start_iso)
          .limit(50000).execute().data or []
    )
    corrs = (
        db.table("corrections").select("id, user_key, action, created_at")
          .gte("created_at", start_iso)
          .limit(50000).execute().data or []
    )

    # Per-day, per-user roll-up
    from collections import defaultdict
    by_day_user: dict[tuple, dict] = defaultdict(
        lambda: {"uploaded": 0, "reviewed": 0, "confirms": 0, "corrections": 0}
    )
    by_user_total: dict[str, dict] = defaultdict(
        lambda: {"uploaded": 0, "reviewed": 0, "confirms": 0, "corrections": 0,
                 "first_seen": None, "last_seen": None}
    )
    for p in photos:
        uk = p.get("user_key") or "(no key)"
        day = (p.get("uploaded_at") or "")[:10]
        if not day:
            continue
        by_day_user[(day, uk)]["uploaded"] += 1
        by_user_total[uk]["uploaded"] += 1
        if not by_user_total[uk]["first_seen"] or day < by_user_total[uk]["first_seen"]:
            by_user_total[uk]["first_seen"] = day
        if not by_user_total[uk]["last_seen"] or day > by_user_total[uk]["last_seen"]:
            by_user_total[uk]["last_seen"] = day
    for c in corrs:
        uk = c.get("user_key") or "(no key)"
        day = (c.get("created_at") or "")[:10]
        if not day:
            continue
        action = c.get("action") or ""
        by_day_user[(day, uk)]["reviewed"] += 1
        by_user_total[uk]["reviewed"] += 1
        if action == "confirm":
            by_day_user[(day, uk)]["confirms"] += 1
            by_user_total[uk]["confirms"] += 1
        elif action == "correct":
            by_day_user[(day, uk)]["corrections"] += 1
            by_user_total[uk]["corrections"] += 1
        if not by_user_total[uk]["first_seen"] or day < by_user_total[uk]["first_seen"]:
            by_user_total[uk]["first_seen"] = day
        if not by_user_total[uk]["last_seen"] or day > by_user_total[uk]["last_seen"]:
            by_user_total[uk]["last_seen"] = day

    # Headline numbers
    today = end.isoformat()
    today_users = {uk for (d, uk) in by_day_user if d == today}
    week_start = (end - timedelta(days=7)).isoformat()
    week_users = {uk for (d, uk) in by_day_user if d >= week_start}
    month_users = set(by_user_total.keys())

    headline = {
        "DAU (today)": len(today_users),
        "WAU (last 7d)": len(week_users),
        f"MAU (last {days}d)": len(month_users),
        "Photos uploaded": sum(u["uploaded"] for u in by_user_total.values()),
        "Photos reviewed": sum(u["reviewed"] for u in by_user_total.values()),
        "Confirm rate": (
            f"{sum(u['confirms'] for u in by_user_total.values())}/"
            f"{sum(u['reviewed'] for u in by_user_total.values()) or 1}"
        ),
    }

    # Per-day table
    days_sorted = sorted({d for (d, _) in by_day_user}, reverse=True)
    daily_rows = []
    for d in days_sorted[:days]:
        users_today = {uk for (dd, uk) in by_day_user if dd == d}
        ups = sum(u["uploaded"] for (dd, _), u in by_day_user.items() if dd == d)
        revs = sum(u["reviewed"] for (dd, _), u in by_day_user.items() if dd == d)
        cnfs = sum(u["confirms"] for (dd, _), u in by_day_user.items() if dd == d)
        cors = sum(u["corrections"] for (dd, _), u in by_day_user.items() if dd == d)
        daily_rows.append((d, len(users_today), ups, revs, cnfs, cors))

    # Per-user table (top 30 by uploads)
    user_rows = sorted(by_user_total.items(),
                       key=lambda kv: -kv[1]["uploaded"])[:30]

    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    head_html = "".join(
        f"<div class=stat><div class=k>{_esc(k)}</div><div class=v>{_esc(v)}</div></div>"
        for k, v in headline.items()
    )
    daily_html = "".join(
        f"<tr><td>{_esc(d)}</td><td>{n}</td><td>{u}</td><td>{r}</td><td>{c}</td><td>{x}</td></tr>"
        for d, n, u, r, c, x in daily_rows
    )
    user_html = "".join(
        f"<tr><td class=mono>{_esc(uk[:14])}…</td><td>{u['uploaded']}</td>"
        f"<td>{u['reviewed']}</td><td>{u['confirms']}</td>"
        f"<td>{u['corrections']}</td><td>{u['first_seen'] or ''}</td>"
        f"<td>{u['last_seen'] or ''}</td></tr>"
        for uk, u in user_rows
    )

    return f"""
<section id="stats" class="adm-section">
  <div class="adm-section-head">
    <h2>Activity overview</h2>
    <span class="adm-window">last {days} days</span>
  </div>
  <div class="stats-grid">{head_html}</div>
  <h3>Daily activity</h3>
  <div class="adm-table-wrap"><table class="stats-tbl">
    <thead><tr><th>Day</th><th>Users</th><th>Uploaded</th><th>Reviewed</th><th>Confirms</th><th>Corrections</th></tr></thead>
    <tbody>{daily_html or '<tr><td colspan=6 class=adm-empty>(no data)</td></tr>'}</tbody>
  </table></div>
  <h3>Top users · last {days} days</h3>
  <div class="adm-table-wrap"><table class="stats-tbl">
    <thead><tr><th>User key</th><th>Uploaded</th><th>Reviewed</th><th>Confirms</th><th>Corrections</th><th>First seen</th><th>Last seen</th></tr></thead>
    <tbody>{user_html or '<tr><td colspan=7 class=adm-empty>(no data)</td></tr>'}</tbody>
  </table></div>
</section>"""


@app.get("/admin/stats", include_in_schema=False)
def admin_stats(request: Request, days: int = 30):
    """Backwards-compat redirect — old bookmarks still work, just
    forward to the unified /admin panel."""
    return RedirectResponse(f"/admin?days={days}#stats", status_code=303)


# ---------- phase 4: hse-class proposals (admin side) ----------

def _append_to_fine_taxonomy(parent_slug: str, slug: str,
                             label_en: str, label_vn: str) -> None:
    """Idempotently append one approved sub-type to
    data/fine_hse_types_by_parent.json. Writes via temp file + rename
    so a crash mid-write doesn't leave a half-written JSON on disk.
    Caller is responsible for invalidating the in-memory cache."""
    fine_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
    tmp_path = fine_path.with_suffix(".json.tmp")
    if not fine_path.exists():
        raise RuntimeError(f"taxonomy file not found: {fine_path}")
    data = json.loads(fine_path.read_text(encoding="utf-8"))
    parents = data.setdefault("parents", {})
    bucket = parents.setdefault(parent_slug, [])

    # If a row with the same slug already exists, leave it (idempotent
    # re-approve doesn't double-write).
    if any((row.get("slug") or "") == slug for row in bucket):
        return

    bucket.append({
        "slug": slug,
        "label_en": label_en,
        "label_vn": label_vn,
        "primary_work_zone": "User-submitted",
    })
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(fine_path)


def _retro_patch_corrections(old_slug: str, new_slug: str) -> int:
    """Replace `pending:<x>` with the approved slug on any corrections
    that were saved while the proposal was pending. Returns the number
    of rows touched. Best-effort — failures are logged."""
    if not old_slug or not new_slug:
        return 0
    db = get_db()
    try:
        rows = (
            db.table("corrections").select("id")
              .eq("fine_hse_type_slug", old_slug)
              .limit(1000).execute().data or []
        )
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        db.table("corrections").update({
            "fine_hse_type_slug": new_slug,
        }).in_("id", ids).execute()
        return len(ids)
    except Exception as e:  # noqa: BLE001
        log.warning("retro-patch corrections failed: %s", e)
        return 0


def _build_proposals_inner(status: str) -> str:
    """Compute the proposals queue's inner HTML for the unified /admin
    panel. Default filter is `pending`; valid statuses also include
    approved | rejected | duplicate | all. Returns body content only;
    page shell + styles live in admin_panel.
    """
    db = get_db()
    valid = {"pending", "approved", "rejected", "duplicate", "all"}
    if status not in valid:
        status = "pending"

    try:
        q = (
            db.table("hse_class_proposals")
              .select("id, proposed_by_user_key, parent_slug, proposed_slug, "
                      "label_en, label_vn, description, example_photo_id, "
                      "status, reviewer_note, approved_slug, "
                      "created_at, reviewed_at, reviewed_by")
              .order("created_at", desc=True).limit(200)
        )
        if status != "all":
            q = q.eq("status", status)
        rows = q.execute().data or []
    except Exception as e:  # noqa: BLE001
        if "hse_class_proposals" in str(e):
            return HTMLResponse(
                "<h1>proposals not configured</h1>"
                "<p>Run <code>scripts/add_hse_class_proposals_table.py</code> "
                "and apply the SQL.</p>", status_code=503)
        raise

    # Load thumbnails for the example_photo_id refs in one batch query.
    photo_thumbs: dict[str, str] = {}
    photo_ids = [r["example_photo_id"] for r in rows if r.get("example_photo_id")]
    if photo_ids:
        try:
            phs = (
                db.table("photos").select("id, r2_thumb_key, r2_object_key")
                  .in_("id", photo_ids).execute().data or []
            )
            for p in phs:
                key = p.get("r2_thumb_key") or p.get("r2_object_key")
                if key:
                    photo_thumbs[p["id"]] = f"/r2/{key}"
        except Exception:  # noqa: BLE001
            pass

    def _esc(s: object) -> str:
        return (str(s or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    counts = {k: 0 for k in ("pending", "approved", "rejected", "duplicate")}
    try:
        all_rows = (
            db.table("hse_class_proposals").select("status")
              .limit(5000).execute().data or []
        )
        for r in all_rows:
            counts[r.get("status", "pending")] = counts.get(r.get("status", "pending"), 0) + 1
    except Exception:  # noqa: BLE001
        pass

    body_rows = []
    for r in rows:
        thumb = photo_thumbs.get(r.get("example_photo_id") or "")
        thumb_html = (
            f"<img src='{_esc(thumb)}' alt='' loading='lazy' "
            "style='width:96px;height:96px;object-fit:cover;border-radius:6px;border:1px solid #e2e8f0'>"
            if thumb else
            "<div style='width:96px;height:96px;border-radius:6px;border:1px dashed #cbd5e1;"
            "display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:11px'>no photo</div>"
        )
        st = r.get("status", "pending")
        st_color = {
            "pending": "#f59e0b",
            "approved": "#10b981",
            "rejected": "#ef4444",
            "duplicate": "#6366f1",
        }.get(st, "#64748b")
        if st == "pending":
            actions_html = (
                f"<form method=POST action='/admin/proposals/{r['id']}/approve' style='display:inline'>"
                "<button class='btn approve' type=submit>Approve</button></form> "
                f"<form method=POST action='/admin/proposals/{r['id']}/reject' "
                "style='display:inline-flex;gap:4px;align-items:center;margin-left:6px'>"
                "<input name=note placeholder='reason (optional)' "
                "style='padding:4px 6px;border:1px solid #cbd5e1;border-radius:4px;font-size:11px;width:140px'>"
                "<button class='btn reject' type=submit>Reject</button></form> "
                f"<form method=POST action='/admin/proposals/{r['id']}/duplicate' "
                "style='display:inline-flex;gap:4px;align-items:center;margin-left:6px'>"
                "<input name=note placeholder='canonical slug' required "
                "style='padding:4px 6px;border:1px solid #cbd5e1;border-radius:4px;font-size:11px;width:160px'>"
                "<button class='btn dup' type=submit>Duplicate</button></form>"
            )
        else:
            decided_at = (r.get("reviewed_at") or "")[:19].replace("T", " ")
            note = r.get("reviewer_note") or ""
            approved_slug = r.get("approved_slug") or ""
            extra = (
                f"<div style='font-size:11px;color:#64748b;margin-top:4px'>"
                f"slug: <code>{_esc(approved_slug)}</code></div>"
                if approved_slug else ""
            )
            actions_html = (
                f"<div style='font-size:11px;color:#64748b'>{_esc(decided_at)}</div>"
                f"<div style='font-size:12px;color:#0f172a;margin-top:4px'>{_esc(note)}</div>"
                + extra
            )

        desc = r.get("description") or ""
        body_rows.append(
            f"<tr><td>{thumb_html}</td>"
            f"<td><div style='font-weight:600'>{_esc(r['label_en'])}</div>"
            f"<div style='color:#475569;font-size:12px;margin-top:2px'>{_esc(r['label_vn'])}</div>"
            f"{f'<div style=color:#64748b;font-size:12px;margin-top:6px;line-height:1.4>{_esc(desc)}</div>' if desc else ''}"
            f"<div style='font-size:11px;color:#94a3b8;margin-top:8px'>"
            f"parent: <code>{_esc(r['parent_slug'])}</code> · "
            f"by <code>{_esc((r.get('proposed_by_user_key') or '')[:14])}…</code> · "
            f"{_esc((r.get('created_at') or '')[:19].replace('T', ' '))}"
            f"</div></td>"
            f"<td><span style='display:inline-block;padding:2px 8px;border-radius:999px;"
            f"background:{st_color}20;color:{st_color};font-size:11px;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:.04em'>{_esc(st)}</span></td>"
            f"<td style='min-width:280px'>{actions_html}</td></tr>"
        )

    rows_html = "".join(body_rows) or (
        "<tr><td colspan=4 style='text-align:center;color:#94a3b8;padding:32px'>"
        "(no proposals in this status)</td></tr>"
    )

    tabs = []
    for s in ("pending", "approved", "rejected", "duplicate", "all"):
        cnt = "" if s == "all" else f" ({counts.get(s, 0)})"
        tabs.append(
            f"<a href='/admin?status={s}#proposals' "
            f"class='tab{' active' if s == status else ''}'>{s}{cnt}</a>"
        )

    return f"""
<section id="proposals" class="adm-section">
  <div class="adm-section-head">
    <h2>HSE class proposals</h2>
    <span class="adm-window">{counts.get('pending', 0)} pending · {len(rows)} shown</span>
  </div>
  <div class="adm-tabs">{''.join(tabs)}</div>
  <div class="adm-table-wrap"><table class="props-tbl">
    <thead><tr><th>Photo</th><th>Proposal</th><th>Status</th><th>Action</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table></div>
</section>"""


@app.get("/admin/proposals", include_in_schema=False)
def admin_proposals(request: Request, status: str = "pending"):
    """Backwards-compat redirect — old bookmarks still work."""
    return RedirectResponse(f"/admin?status={status}#proposals", status_code=303)


@app.post("/admin/proposals/{proposal_id}/approve")
def admin_proposal_approve(proposal_id: str, request: Request):
    """Approve a proposal: append to fine_hse_types_by_parent.json,
    retro-patch any corrections still tagged with `pending:<slug>`,
    invalidate the in-memory taxonomy cache, and mark the row decided."""
    if not _admin_authed(request):
        return RedirectResponse("/admin/login", status_code=303)

    db = get_db()
    rows = (
        db.table("hse_class_proposals").select("*")
          .eq("id", proposal_id).limit(1).execute().data or []
    )
    if not rows:
        raise HTTPException(404, "proposal not found")
    p = rows[0]
    if p["status"] != "pending":
        return RedirectResponse("/admin#proposals", status_code=303)

    pending_slug = p["proposed_slug"]   # e.g. "pending:Worker_no_harness"
    final_slug = pending_slug.split(":", 1)[1] if ":" in pending_slug else pending_slug
    # Disambiguate against the live taxonomy too (in case another
    # parent has the same slug under a different bucket — shouldn't
    # happen but cheap to defend).
    if not hasattr(app.state, "fine_types"):
        fine_path = REPO_ROOT / "data" / "fine_hse_types_by_parent.json"
        try:
            app.state.fine_types = json.loads(
                fine_path.read_text(encoding="utf-8")
            ).get("parents", {})
        except Exception:  # noqa: BLE001
            app.state.fine_types = {}
    seen_slugs = {
        row.get("slug") for parent_rows in (app.state.fine_types or {}).values()
        for row in parent_rows
    }
    if final_slug in seen_slugs:
        suffix = 2
        while f"{final_slug}_{suffix}" in seen_slugs:
            suffix += 1
        final_slug = f"{final_slug}_{suffix}"

    try:
        _append_to_fine_taxonomy(
            p["parent_slug"], final_slug, p["label_en"], p["label_vn"]
        )
    except Exception as e:  # noqa: BLE001
        log.error("approve: taxonomy append failed: %s", e)
        raise HTTPException(500, "taxonomy file write failed")

    _reload_fine_types_cache()
    patched = _retro_patch_corrections(pending_slug, final_slug)

    from datetime import datetime, timezone
    db.table("hse_class_proposals").update({
        "status": "approved",
        "approved_slug": final_slug,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": "admin",
        "reviewer_note": (
            f"Approved as '{final_slug}'."
            + (f" Retro-patched {patched} corrections." if patched else "")
        ),
    }).eq("id", proposal_id).execute()

    return RedirectResponse("/admin#proposals", status_code=303)


@app.post("/admin/proposals/{proposal_id}/reject")
def admin_proposal_reject(
    proposal_id: str, request: Request, note: str | None = Form(None)
):
    """Mark a proposal rejected. The note (optional) is shown to the
    user in the decision toast."""
    if not _admin_authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    db = get_db()
    rows = (
        db.table("hse_class_proposals").select("status")
          .eq("id", proposal_id).limit(1).execute().data or []
    )
    if not rows:
        raise HTTPException(404, "proposal not found")
    if rows[0]["status"] != "pending":
        return RedirectResponse("/admin#proposals", status_code=303)

    from datetime import datetime, timezone
    db.table("hse_class_proposals").update({
        "status": "rejected",
        "reviewer_note": (note or "").strip() or "Rejected.",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": "admin",
    }).eq("id", proposal_id).execute()
    return RedirectResponse("/admin#proposals", status_code=303)


@app.post("/admin/proposals/{proposal_id}/duplicate")
def admin_proposal_duplicate(
    proposal_id: str, request: Request, note: str = Form(...)
):
    """Mark a proposal as a duplicate of an existing slug. The note is
    REQUIRED here — it should be the canonical slug the user should
    have used. Surfaced to the user in the decision toast."""
    if not _admin_authed(request):
        return RedirectResponse("/admin/login", status_code=303)
    canonical = (note or "").strip()
    if not canonical:
        raise HTTPException(400, "canonical slug required")

    db = get_db()
    rows = (
        db.table("hse_class_proposals").select("status, proposed_slug")
          .eq("id", proposal_id).limit(1).execute().data or []
    )
    if not rows:
        raise HTTPException(404, "proposal not found")
    if rows[0]["status"] != "pending":
        return RedirectResponse("/admin#proposals", status_code=303)

    # Retro-patch corrections that used the pending slug to the canonical
    # one — same intent as approve, just routing to an existing taxonomy
    # row instead of a new one.
    patched = _retro_patch_corrections(rows[0]["proposed_slug"], canonical)

    from datetime import datetime, timezone
    db.table("hse_class_proposals").update({
        "status": "duplicate",
        "approved_slug": canonical,
        "reviewer_note": (
            f"Marked as duplicate of '{canonical}'."
            + (f" Retro-patched {patched} corrections." if patched else "")
        ),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": "admin",
    }).eq("id", proposal_id).execute()
    return RedirectResponse("/admin#proposals", status_code=303)


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
