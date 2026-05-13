"""Download HSE-disciplined photos from AECIS's public S3 bucket.

Reads Issue_Gen/Issue_Gen/result_after_query.csv, filters to
DisciplineID = 10, constructs S3 URLs from each row's FilePath
column using the base URL in Issue_Gen/PUBLIC_S3_URL.txt (or the
AECIS_PHOTO_S3_BASE env var), and downloads each photo to
Issue_Gen/photos/<filepath>.

Resumable — skips files already present at full size. Manifests
each download into Issue_Gen/photos/manifest.jsonl so a follow-up
Playwright-driven assign script knows which photo maps to which
AECIS issue.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/seed_download_aecis_photos.py [--limit N] [--workers N]

Defaults: --workers 8, no limit (~2,400 photos).
"""
import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "Issue_Gen" / "Issue_Gen" / "result_after_query.csv"
PHOTO_ROOT = REPO_ROOT / "Issue_Gen" / "photos"
MANIFEST = PHOTO_ROOT / "manifest.jsonl"
URL_FILE = REPO_ROOT / "Issue_Gen" / "PUBLIC_S3_URL.txt"


def resolve_s3_base() -> str:
    """Env var > Issue_Gen/PUBLIC_S3_URL.txt. Trailing slash stripped."""
    env = os.environ.get("AECIS_PHOTO_S3_BASE")
    if env:
        return env.rstrip("/")
    if URL_FILE.exists():
        for line in URL_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                return s.rstrip("/")
    sys.stderr.write(
        "ERROR: no S3 base URL configured. Set AECIS_PHOTO_S3_BASE or "
        "paste URL into Issue_Gen/PUBLIC_S3_URL.txt.\n"
    )
    sys.exit(2)


def iter_hse_rows(limit: int = 0):
    """Yield {issue_id, project_id, issue_name, description, filepath}
    for each HSE-disciplined photo row (DisciplineID=10)."""
    if not CSV_PATH.exists():
        sys.stderr.write(f"ERROR: missing {CSV_PATH}\n")
        sys.exit(2)
    n = 0
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 70:
                continue
            if row[6] != "10":
                continue
            filepath = (row[61] or "").strip()
            if not filepath:
                continue
            yield {
                "issue_id": row[0],
                "project_id": row[2],
                "issue_name": row[4],
                "description": row[12],
                "filepath": filepath,
            }
            n += 1
            if limit and n >= limit:
                return


def download_one(url: str, dest: Path, timeout: int = 30) -> tuple[str, int]:
    """Returns (status, bytes). status in {ok, skip, fail}."""
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", dest.stat().st_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hse-seeder/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.rename(dest)
        return ("ok", len(data))
    except Exception as e:  # noqa: BLE001
        if tmp.exists():
            try: tmp.unlink()
            except: pass
        sys.stderr.write(f"  FAIL {dest.name}: {e}\n")
        return ("fail", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--dry-run", action="store_true", help="print URLs, no download")
    args = ap.parse_args()

    base = resolve_s3_base()
    sys.stdout.write(f"S3 base: {base}\n")

    rows = list(iter_hse_rows(args.limit))
    sys.stdout.write(f"HSE rows to fetch: {len(rows)}\n")

    if args.dry_run:
        for r in rows[:20]:
            sys.stdout.write(f"  {base}/{r['filepath']}\n")
        if len(rows) > 20:
            sys.stdout.write(f"  ... {len(rows) - 20} more\n")
        return

    PHOTO_ROOT.mkdir(parents=True, exist_ok=True)
    # Append-mode manifest so resumed runs accumulate entries.
    manifest_f = MANIFEST.open("a", encoding="utf-8")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    bytes_total = 0
    t0 = time.perf_counter()

    def task(row):
        url = f"{base}/{row['filepath']}"
        dest = PHOTO_ROOT / row["filepath"]
        status, n_bytes = download_one(url, dest)
        return row, status, n_bytes

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row, status, n_bytes in ex.map(task, rows):
            counts[status] += 1
            bytes_total += n_bytes
            if status == "ok":
                manifest_f.write(json.dumps({
                    "filepath": row["filepath"],
                    "issue_id": row["issue_id"],
                    "project_id": row["project_id"],
                    "issue_name": row["issue_name"],
                    "description": row["description"],
                    "bytes": n_bytes,
                }, ensure_ascii=False) + "\n")
            done = counts["ok"] + counts["skip"] + counts["fail"]
            if done % 50 == 0:
                sys.stdout.write(
                    f"  {done}/{len(rows)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}\n"
                )

    manifest_f.close()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(
        f"\nDone in {elapsed:.1f}s — ok={counts['ok']} skip={counts['skip']} "
        f"fail={counts['fail']}  ({bytes_total / 1024 / 1024:.1f} MB written)\n"
        f"Manifest: {MANIFEST}\n"
    )


if __name__ == "__main__":
    main()
