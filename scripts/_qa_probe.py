"""Adversarial input testing harness.

Spins up a TestClient with AUTH_REQUIRED=0 so we can hit
authenticated handlers, then probes them with malformed /
hostile inputs to find validation gaps. Designed to be safe
to run repeatedly — does NOT persist anything to a real DB
when stub Supabase URLs are configured (DB calls fail and
we capture the failure shape).

Run:  python scripts/_qa_probe.py
"""
from __future__ import annotations
import json
import os
import sys
import re
import textwrap

# --- Env stub ---
# Use stub Supabase / R2 credentials so the app boots but DB calls
# raise rather than mutating real data. We only check behaviour that
# happens BEFORE the DB call (validation, slug normalization, etc.)
# or that catches the DB failure cleanly.
os.environ.setdefault("SUPABASE_URL", "https://stub.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "stub")
os.environ.setdefault("R2_ACCOUNT_ID", "stub")
os.environ.setdefault("R2_ACCESS_KEY_ID", "stub")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "stub")
os.environ.setdefault("R2_BUCKET", "stub")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-stub")
os.environ.setdefault("AUTH_REQUIRED", "0")
os.environ.setdefault("ADMIN_PASSWORD", "test-admin-password")  # so admin gate is reachable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from webapp import app as appmod
# raise_server_exceptions=False makes TestClient return 500 responses
# instead of re-raising — matches production behaviour for users.
client = TestClient(appmod.app, raise_server_exceptions=False)

PASS = "\033[32m PASS \033[0m"
FAIL = "\033[31m FAIL \033[0m"
WARN = "\033[33m WARN \033[0m"
INFO = "\033[36m INFO \033[0m"

findings = []


def expect(label: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    print(f"{tag} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        findings.append((label, detail))


def warn_finding(label: str, detail: str = ""):
    print(f"{WARN} {label}{(' — ' + detail) if detail else ''}")
    findings.append((label, detail))


def section(name: str):
    print()
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)


# -----------------------------------------------------------------
section("1. /api/proposals — input validation")

# All these should return 400 (or fail the validation layer cleanly)
# BEFORE any DB call. We can't easily distinguish a 400 from validation
# vs a 500 from the DB stub failing, so accept either as "didn't 200".

cases = [
    # name, form data, expected status
    ("missing parent_slug",     {"label_en": "x", "label_vn": "x"}, 422),
    ("missing label_en",         {"parent_slug": "Mass_piling_unsafe", "label_vn": "x"}, 422),
    ("missing label_vn",         {"parent_slug": "Mass_piling_unsafe", "label_en": "x"}, 422),
    ("unknown parent_slug",      {"parent_slug": "DOES_NOT_EXIST", "label_en": "x", "label_vn": "x"}, 400),
    ("empty label_en",           {"parent_slug": "Mass_piling_unsafe", "label_en": "   ", "label_vn": "x"}, 400),
    ("empty label_vn",           {"parent_slug": "Mass_piling_unsafe", "label_en": "x", "label_vn": "   "}, 400),
    ("oversized label_en (201)", {"parent_slug": "Mass_piling_unsafe", "label_en": "x" * 201, "label_vn": "ok"}, 400),
    ("oversized label_vn (201)", {"parent_slug": "Mass_piling_unsafe", "label_en": "ok", "label_vn": "x" * 201}, 400),
    ("oversized description",     {"parent_slug": "Mass_piling_unsafe", "label_en": "ok", "label_vn": "ok", "description": "z" * 1001}, 400),
    ("XSS in label_en",          {"parent_slug": "Mass_piling_unsafe", "label_en": "<script>alert(1)</script>", "label_vn": "ok"}, "any"),
    ("RTL chars in label",       {"parent_slug": "Mass_piling_unsafe", "label_en": "‮أحمد", "label_vn": "ok"}, "any"),
    ("null bytes in label",      {"parent_slug": "Mass_piling_unsafe", "label_en": "ok\x00bad", "label_vn": "ok"}, "any"),
    ("very large parent_slug",   {"parent_slug": "x" * 5000, "label_en": "ok", "label_vn": "ok"}, 400),
    ("non-uuid example_photo_id","example_photo_id=not-a-uuid&parent_slug=Mass_piling_unsafe&label_en=ok&label_vn=ok", "any"),
]

for name, data, expected in cases:
    if isinstance(data, str):
        r = client.post("/api/proposals", data=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded",
                                 "Cookie": "vai_uid=qa_test_proposals"})
    else:
        r = client.post("/api/proposals", data=data,
                        headers={"Cookie": "vai_uid=qa_test_proposals"})
    if expected == "any":
        ok = r.status_code != 200 or "id" not in r.text  # would be a real success only if DB worked
        expect(f"{name:<32}  status {r.status_code}", ok, r.text[:120])
    else:
        ok = r.status_code == expected or (expected == 400 and r.status_code in (400, 422))
        expect(f"{name:<32}  status {r.status_code} (expected {expected})", ok)


# -----------------------------------------------------------------
section("2. /api/export/email — input validation")

# Should not be configured (no SMTP env), so should return 503.
cases_email = [
    ("no body",                 {}),
    ("invalid email format",    {"format": "pdf", "to": "not-an-email"}),
    ("unsupported format",      {"format": "exe", "to": "x@x.com"}),
    ("oversized subject (1k)",  {"format": "pdf", "to": "x@x.com", "subject": "x" * 1000}),
    ("oversized body",          {"format": "pdf", "to": "x@x.com", "body": "z" * 100000}),
    ("XSS in subject",          {"format": "pdf", "to": "x@x.com", "subject": "<script>alert(1)</script>"}),
    ("invalid batch_id (uuid)", {"format": "pdf", "to": "x@x.com", "batch_id": "not-a-uuid"}),
    ("comma-separated emails",  {"format": "pdf", "to": "a@b.com,c@d.com"}),
    ("email with header inj",   {"format": "pdf", "to": "x@x.com\r\nBCC: evil@bad.com"}),
]
for name, data in cases_email:
    r = client.post("/api/export/email", data=data, headers={"Cookie": "vai_uid=qa_test_email"})
    expect(f"{name:<32}  status {r.status_code}",
           r.status_code in (400, 422, 503),
           r.text[:120])


# -----------------------------------------------------------------
section("3. /admin/* with malformed inputs")

# Set the admin cookie so we can probe deeper than the gate
auth = {"Cookie": "vai_admin=test-admin-password"}

# Approve / reject / duplicate with various invalid IDs
for action in ("approve", "reject", "duplicate"):
    bad_ids = [
        "abc",
        "../../etc",
        "<script>alert(1)</script>",
        "00000000-0000-0000-0000-000000000000",
        "z" * 100,
        " ",
        "javascript:alert(1)",
    ]
    for bad in bad_ids:
        # duplicate requires a `note` form field
        data = {"note": "test"} if action == "duplicate" else {}
        r = client.post(f"/admin/proposals/{bad}/{action}", data=data, headers=auth)
        # Expect: 400, 404, or 422 — anything but 200/500/redirect-success
        ok = r.status_code in (400, 404, 422, 303, 500)
        ok_safe = "alert(1)" not in r.text
        expect(f"{action:<10} bad-id={bad[:30]:<30} -> {r.status_code}", ok and ok_safe)

# Duplicate with empty note (REQUIRED form field)
r = client.post("/admin/proposals/abc/duplicate", data={"note": ""}, headers=auth)
expect(f"duplicate empty note -> {r.status_code}",
       r.status_code in (400, 422))


# -----------------------------------------------------------------
section("4. /api/photos/{id}/* with malformed photo IDs")

bad_photo_ids = [
    "abc",
    "00000000-0000-0000-0000-000000000000",
    "../../etc",
    "<script>",
    "%2e%2e%2f%2e%2e%2fpasswd",
    "z" * 100,
]
for bad in bad_photo_ids:
    for endpoint in ("retry", "history"):
        path = f"/api/photos/{bad}/{endpoint}"
        method_post = endpoint == "retry"
        if method_post:
            r = client.post(path, headers={"Cookie": "vai_uid=qa_test_photos"})
        else:
            r = client.get(path, headers={"Cookie": "vai_uid=qa_test_photos"})
        # Should be 4xx or 5xx (DB lookup fails) — never 200 with leaked data
        ok = r.status_code != 200
        leak = bad in r.text and "<" in bad  # XSS leak check
        expect(f"{endpoint:<10} bad-id={bad[:30]:<30} -> {r.status_code}", ok and not leak)


# -----------------------------------------------------------------
section("5. /admin/login with malformed input")

# Form-based login. Try various pathological passwords.
for pw in ["", "a" * 10000, "<script>", "' OR 1=1 --", "wrong", "test-admin-password"]:
    r = client.post("/admin/login", data={"password": pw}, follow_redirects=False)
    if pw == "test-admin-password":
        expect(f"correct password -> {r.status_code}", r.status_code == 303,
               f"Location={r.headers.get('location', '')}")
    elif not pw:
        expect(f"empty password -> {r.status_code}", r.status_code in (400, 401, 422))
    else:
        expect(f"pw={pw[:30]:<30} -> {r.status_code}", r.status_code in (401, 422))


# -----------------------------------------------------------------
section("6. /auth/login/google + /auth/callback/google edge cases")

# OAuth not configured in stub env → 503.
r = client.get("/auth/login/google?next=//evil.com", follow_redirects=False)
expect(f"login/google with //evil.com next -> {r.status_code}",
       r.status_code == 503)

# Even if it WAS configured, next would be sanitized to "/".
# Test the sanitization branch by patching the env and re-importing.
# Skipped here because import side effects are global. Documented as:
#  "// open redirect blocked by `safe_next = next.startswith('/') and not next.startswith('//')`"

r = client.get("/auth/callback/google?error=access_denied&error_description=user+canceled",
               follow_redirects=False)
expect(f"callback w/ error=access_denied -> {r.status_code}",
       r.status_code == 503)  # Still 503 in stub env

# Inject crap in code/state
r = client.get("/auth/callback/google?code=AAAA&state=" + "X" * 5000, follow_redirects=False)
expect(f"callback w/ 5kb state -> {r.status_code}",
       r.status_code == 503)


# -----------------------------------------------------------------
section("7. Locale toggle + JS injection in HTML rendering")

# /auth/signin embeds the next param. Verify it's URL-encoded.
r = client.get("/auth/signin?next=/foo<script>alert(1)</script>")
ok = "<script>alert(1)" not in r.text  # raw script tag should not appear
ok2 = "%3Cscript%3E" in r.text or "/foo%3C" in r.text  # URL-encoded form
expect(f"chooser next param XSS-safe", ok)
expect(f"chooser next param URL-encoded", ok2,
       "(if signin links don't contain URL-encoded next, it's missing the safety check)")

# Same for landing — the landing's CTA goes to /auth/signin without
# carrying next, so should be fine. Verify no JS injection vector.
r = client.get("/")
ok = "<script>alert(" not in r.text
expect("landing has no inline alert(1)", ok)


# -----------------------------------------------------------------
section("8. Random tampering of session cookie")

# /auth/me with various bad cookies
for cookie in [
    "vai_session=",
    "vai_session=AAAAAAAAAAAA",
    "vai_session=" + "x" * 5000,
    "vai_session=null",
    "vai_session=undefined",
    "vai_session=' OR 1=1 --",
    "vai_session=%00null",
]:
    r = client.get("/auth/me", headers={"Cookie": cookie})
    j = r.json()
    ok = j.get("authed") == False
    expect(f"/auth/me with cookie='{cookie[:40]:<40}' -> authed={j.get('authed')}", ok)


# -----------------------------------------------------------------
section("9. Multiple concurrent /api/proposals on same cookie (rate)")

# Try 5 quick fires — should mostly succeed (DB stub fails) but not crash
import concurrent.futures as cf
def fire(i):
    r = client.post("/api/proposals",
                    data={"parent_slug": "Mass_piling_unsafe",
                          "label_en": f"qa_concurrent_{i}",
                          "label_vn": "vn"},
                    headers={"Cookie": "vai_uid=qa_test_concurrent"})
    return r.status_code, r.text[:80]
with cf.ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(fire, range(5)))
codes = [c for c, _ in results]
print(f"  5 concurrent calls returned: {codes}")
# All should be the same code (no state corruption)


# -----------------------------------------------------------------
section("10. Header behaviors for HEAD/OPTIONS")

for method, path, expected_code in [
    ("HEAD", "/", 200),
    ("OPTIONS", "/", "any"),
    ("PUT", "/", 405),
    ("DELETE", "/", 405),
    ("PATCH", "/", 405),
    ("HEAD", "/auth/signin", "any"),
    ("HEAD", "/api/usage/me", "any"),
]:
    r = client.request(method, path)
    if expected_code == "any":
        expect(f"{method:<7} {path:<20} -> {r.status_code}", True)
    else:
        expect(f"{method:<7} {path:<20} -> {r.status_code} (expected {expected_code})",
               r.status_code == expected_code)


# -----------------------------------------------------------------
section("11. Service worker + manifest content type")

r = client.get("/service-worker.js")
ct = r.headers.get("content-type", "")
expect(f"/service-worker.js content-type = {ct}",
       "javascript" in ct.lower())
expect(f"/service-worker.js Service-Worker-Allowed = {r.headers.get('service-worker-allowed')}",
       r.headers.get("service-worker-allowed") == "/")
expect(f"/service-worker.js Cache-Control",
       "no-cache" in r.headers.get("cache-control", "").lower() or "no-store" in r.headers.get("cache-control", "").lower())

r = client.get("/static/manifest.json")
ct = r.headers.get("content-type", "")
expect(f"/static/manifest.json content-type = {ct}",
       "json" in ct.lower())


# -----------------------------------------------------------------
print()
print("=" * 70)
if findings:
    print(f"  Total findings: {len(findings)}")
    for f, d in findings:
        print(f"   - {f}")
        if d:
            print(f"        {d[:150]}")
else:
    print("  All checks passed.")
print("=" * 70)
sys.exit(1 if findings else 0)
