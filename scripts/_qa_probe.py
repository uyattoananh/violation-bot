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
# After the UUID-validator fix:
#   non-UUID -> 400 (clean)
#   valid UUID but not in DB -> 404
#   path-traversal IDs -> 404 (FastAPI router rejects)
for action in ("approve", "reject", "duplicate"):
    bad_ids = [
        ("abc", 400),                         # not a uuid
        ("../../etc", 404),                   # sliced into wrong route
        ("<script>alert(1)</script>", 404),   # router rejects bracket chars in path
        ("00000000-0000-0000-0000-000000000000", 503),  # valid UUID, stub-DB unreachable -> 503
        ("z" * 100, 400),                     # too long, fails uuid check
        (" ", 400),                           # whitespace
        ("javascript:alert(1)", 400),         # not a uuid
    ]
    for bad, expected in bad_ids:
        data = {"note": "test"} if action == "duplicate" else {}
        r = client.post(f"/admin/proposals/{bad}/{action}", data=data, headers=auth)
        ok = r.status_code == expected
        ok_safe = "alert(1)" not in r.text
        expect(f"{action:<10} bad-id={bad[:30]:<30} -> {r.status_code} (expected {expected})",
               ok and ok_safe)

# Duplicate with empty note (REQUIRED form field)
r = client.post("/admin/proposals/abc/duplicate", data={"note": ""}, headers=auth)
expect(f"duplicate empty note -> {r.status_code}",
       r.status_code in (400, 422))


# -----------------------------------------------------------------
section("4. /api/photos/{id}/* with malformed photo IDs")

# After UUID-validator fix:
#   non-UUID -> 400 (cleanly rejected before any DB call)
#   valid UUID -> hits DB; with stub Supabase that fails -> 500 (history) / clean fallback
#   path-traversal IDs -> 404 from the router itself
bad_photo_ids = [
    # valid UUID -> history degrades gracefully to {history:[],error:...}
    # at 200 (existing try/except), retry surfaces the DB failure as 500.
    ("abc",      400, 400),
    ("00000000-0000-0000-0000-000000000000", 200, 500),
    ("../../etc",  404, 404),
    ("<script>",   400, 400),
    ("%2e%2e%2f%2e%2e%2fpasswd", 404, 404),
    ("z" * 100,    400, 400),
]
for bad, hist_expected, retry_expected in bad_photo_ids:
    for endpoint, expected in (("retry", retry_expected), ("history", hist_expected)):
        path = f"/api/photos/{bad}/{endpoint}"
        if endpoint == "retry":
            r = client.post(path, headers={"Cookie": "vai_uid=qa_test_photos"})
        else:
            r = client.get(path, headers={"Cookie": "vai_uid=qa_test_photos"})
        ok = r.status_code == expected
        leak = bad in r.text and "<" in bad  # XSS leak check
        expect(f"{endpoint:<10} bad-id={bad[:30]:<30} -> {r.status_code} (expected {expected})",
               ok and not leak)


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
expect(f"chooser next param XSS-safe", ok)
# URL-encoding only verifiable when OAuth env is set (otherwise no
# provider buttons render = no `next=` links to inspect). Skip in
# stub env.
# Look for an actual <a class="provider-btn"> rather than the CSS rule
# (the CSS class .provider-btn is in the stylesheet regardless).
if 'class="provider-btn google"' in r.text or 'class="provider-btn microsoft"' in r.text:
    ok2 = "%3Cscript%3E" in r.text or "/foo%3C" in r.text
    expect(f"chooser next param URL-encoded", ok2,
           "(signin links should contain URL-encoded next)")
else:
    print(f"{INFO} chooser next= URL-encoding (skipped — no OAuth env in test)")

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
section("11.4 DELETE endpoints — UUID + ownership gate")

# DELETE /api/photos/{id} should reject:
#   - non-UUID -> 400
#   - non-existent UUID -> 404
#   - existing UUID owned by someone else -> 403 (can't test without
#     real DB rows; we settle for the not-found path here)
for bad, expected in [
    ("abc", 400),
    ("../../etc", 404),
    ("00000000-0000-0000-0000-000000000000", 503),  # valid uuid; stub DB
                                                     # unreachable -> 503
    ("z" * 100, 400),
    ("<script>alert(1)</script>", 404),
]:
    r = client.delete(f"/api/photos/{bad}",
                      headers={"Cookie": "vai_uid=qa_test_delete"})
    expect(f"DELETE photo {bad[:30]:<30} -> {r.status_code} (expected {expected})",
           r.status_code == expected)

# DELETE /api/batches/{id} same shape.
for bad, expected in [
    ("abc", 400),
    ("00000000-0000-0000-0000-000000000000", 500),  # valid uuid; tenant
                                                     # config check fires
                                                     # before ownership in
                                                     # stub env
    ("../../etc", 404),
]:
    r = client.delete(f"/api/batches/{bad}",
                      headers={"Cookie": "vai_uid=qa_test_delete"})
    expect(f"DELETE batch {bad[:30]:<30} -> {r.status_code} (expected {expected})",
           r.status_code == expected)

# clear-reviews same.
for bad, expected in [
    ("abc", 400),
    ("../../etc", 404),
    ("00000000-0000-0000-0000-000000000000", 500),  # stub-DB unreachable
                                                     # surfaces as 500
]:
    r = client.post(f"/api/batches/{bad}/clear-reviews",
                    headers={"Cookie": "vai_uid=qa_test_clear"})
    expect(f"clear-reviews {bad[:30]:<30} -> {r.status_code} (expected {expected})",
           r.status_code == expected)


# -----------------------------------------------------------------
section("11.5 Upload guard rails (size + count caps)")

# Build a fake file that's too many in count. The handler should
# return 413 BEFORE reading any bytes.
import io
many_files = [
    ("files", (f"f{i}.jpg", io.BytesIO(b"\xff\xd8\xff" + b"x" * 200), "image/jpeg"))
    for i in range(60)
]
r = client.post("/api/upload", files=many_files,
                headers={"Cookie": "vai_uid=qa_test_upload_count"})
expect(f"60 files in one POST -> {r.status_code}", r.status_code == 413,
       r.text[:120])

# Single oversized file: 30 MB of zeros (above 25 MB per-file cap).
# In stub env, we 500 on the tenant-missing check before we get to
# the size cap (which is per-file, post-read). Real prod env has a
# tenant configured and the request returns 200 with rejected[]
# containing reason="too_large". Document the test-env limitation:
big = b"\x00" * (30 * 1024 * 1024)
big_files = [("files", ("big.jpg", io.BytesIO(big), "image/jpeg"))]
r = client.post("/api/upload", files=big_files,
                headers={"Cookie": "vai_uid=qa_test_upload_size"})
# 500 = stub tenant; 200/413/422/503 = real validation. Both ok for QA.
ok = r.status_code in (200, 413, 422, 500, 503)
expect(f"30 MB single-file POST -> {r.status_code} (size cap is per-file post-tenant-check)",
       ok, r.text[:140])


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
