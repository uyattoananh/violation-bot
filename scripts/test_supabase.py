"""Smoke-test Supabase credentials + schema + seed state.

Checks (in order):
  1. Can connect with the service-role key
  2. Required tables exist (tenants, projects, photos, classifications,
     corrections, classify_jobs, locations, hse_types)
  3. Taxonomy is seeded — counts match taxonomy.json
  4. At least one tenant + project exists; if none, OFFER to create them

Exits 0 on green. Any failure tells you exactly which step broke.

Usage:
  python scripts/test_supabase.py
  python scripts/test_supabase.py --bootstrap   # create a tenant+project if missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


REQUIRED_TABLES = [
    "tenants", "projects", "users",
    "photos", "classifications", "corrections", "classify_jobs",
    "locations", "hse_types",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true",
                    help="Create a default tenant + SVN project if none exist.")
    ap.add_argument("--tenant-name", type=str, default="Demo PM Company")
    ap.add_argument("--project-code", type=str, default="SVN")
    ap.add_argument("--project-name", type=str, default="Sun Valley Nha Trang")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key and "REPLACE_ME" not in url and "REPLACE_ME" not in key):
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env first.",
              file=sys.stderr)
        return 2

    try:
        from supabase import create_client
    except ImportError:
        print("supabase-py not installed. Run: pip install supabase", file=sys.stderr)
        return 2

    db = create_client(url, key)
    print("[1/4] connected to", url)

    # Step 2: verify tables
    missing = []
    for t in REQUIRED_TABLES:
        try:
            db.table(t).select("*", count="exact").limit(0).execute()
        except Exception as e:  # noqa: BLE001
            missing.append((t, str(e)[:120]))
    if missing:
        print(f"[2/4] missing/unreadable tables:")
        for name, err in missing:
            print(f"        {name:<20} {err}")
        print("        → did you paste schema.sql into the SQL editor?")
        return 1
    print(f"[2/4] all {len(REQUIRED_TABLES)} required tables present")

    # Step 3: taxonomy seed check
    tax = json.loads((REPO_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    expect_loc = len(tax["locations"])
    expect_hse = len(tax["hse_types"])
    loc_rows = db.table("locations").select("slug", count="exact").execute()
    hse_rows = db.table("hse_types").select("slug", count="exact").execute()
    have_loc = loc_rows.count or 0
    have_hse = hse_rows.count or 0
    if have_loc < expect_loc or have_hse < expect_hse:
        print(f"[3/4] taxonomy under-seeded: locations {have_loc}/{expect_loc}, "
              f"hse_types {have_hse}/{expect_hse}")
        print("        → run: python scripts/seed_taxonomy_to_supabase.py")
        return 1
    print(f"[3/4] taxonomy seeded: {have_loc} locations, {have_hse} hse_types")

    # Step 4: tenant + project bootstrap
    tenants = db.table("tenants").select("id, name").execute().data or []
    projects = db.table("projects").select("id, tenant_id, code, name").execute().data or []
    if tenants and projects:
        print(f"[4/4] tenants={len(tenants)}  projects={len(projects)}  (ready)")
        for t in tenants[:3]:
            print(f"        tenant {t['id']}  {t['name']!r}")
        for p in projects[:5]:
            print(f"        project {p['id']}  {p['code']}  {p['name']!r}")
        print()
        print("Supabase OK — paste these UUIDs into the Upload form.")
        return 0

    if not args.bootstrap:
        print(f"[4/4] no tenants/projects yet.")
        print("        → re-run with --bootstrap to create a demo tenant + project")
        return 1

    # Create demo tenant + project
    tenant = db.table("tenants").insert({"name": args.tenant_name}).execute().data[0]
    print(f"[4/4] created tenant {tenant['id']}  {tenant['name']!r}")
    project = db.table("projects").insert({
        "tenant_id": tenant["id"],
        "code": args.project_code,
        "name": args.project_name,
    }).execute().data[0]
    print(f"      created project {project['id']}  {project['code']}  {project['name']!r}")
    print()
    print("Supabase OK — copy these two UUIDs for the Upload form:")
    print(f"  tenant_id  = {tenant['id']}")
    print(f"  project_id = {project['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
