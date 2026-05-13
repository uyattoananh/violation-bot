"""Download HSE-disciplined photos from AECIS's photo bucket.

LOCAL-ONLY: this script is meant to run on a workstation, not on
the VPS service. Issue_Gen/ is bundled in the repo but isn't
deployed to /root/violation-bot, and writing 2,400+ files into
the running service's working tree is a poor fit for production.
Use `scripts/seed_run_local.py` as the entry point if you want
download + assign in one shot.

Reads Issue_Gen/Issue_Gen/result_after_query.csv, filters to
DisciplineID = 10, builds a URL per row via one of four signer
modes (SecureLink / IAM / manifest / unsigned — see
build_aecis_signer), and downloads each photo to
Issue_Gen/photos/<filepath>.

Resumable — skips files already present at full size. Manifests
each download into Issue_Gen/photos/manifest.jsonl so the
Playwright-driven assigner knows which photo maps to which
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


def _build_securelink(file_path: str, user_title: str = "") -> str | None:
    """AECIS SecureLink URL builder — Issue_Gen/method.txt §4.2.

    Returns the api.upload SecureLink URL for the given keyName.
    The endpoint validates the MD5 hash and either streams the
    file or redirects to a fresh AWS presigned URL. None if config
    is missing (caller falls through to the next signer mode).
    """
    secret = os.environ.get("AECIS_NGINX_HASH_KEY")
    base = os.environ.get("AECIS_LINK_API_URL")
    if not secret or not base or not file_path:
        return None
    import hashlib, math, time as _t
    from urllib.parse import quote as _q
    key = file_path.replace("\\\\", "/").replace("\\", "/")
    minutes = int(os.environ.get("AECIS_NGINX_EXPIRE_MIN", "60"))
    # Round down to whole minutes; the C# implementation does the
    # same and the server-side validator hashes against this exact
    # number — off-by-one seconds invalidate the URL.
    expire_min = int(math.floor((_t.time() + minutes * 60) / 60) * 60)
    raw = f"{key}.{expire_min}.{secret}"
    md5 = hashlib.md5(raw.encode("ascii")).hexdigest().upper()
    return (
        f"{base.rstrip('/')}/Files/SecureLink"
        f"?keyName={_q(key, safe='')}"
        f"&expiredTime={expire_min}"
        f"&md5hash={md5}"
        f"&rename={_q(user_title or '', safe='')}"
    )


def build_aecis_signer():
    """Returns (signer_fn, mode_label).
    signer_fn(filepath, user_title) -> URL string ("" if not buildable).

    Four modes, picked in documented-preference order:

      1. SECURELINK   — AECIS_NGINX_HASH_KEY + AECIS_LINK_API_URL
                        set. THE documented integration path
                        (Issue_Gen/method.txt §4.2). Builds URLs
                        against AECIS's api.upload proxy; no IAM
                        credentials needed.
      2. IAM signing  — AECIS_S3_BUCKET + AECIS_S3_REGION +
                        AECIS_S3_ACCESS_KEY_ID + AECIS_S3_SECRET_ACCESS_KEY.
                        Only useful if AECIS hands over a read-
                        only key pair (off-path per their doc).
      3. MANIFEST     — AECIS_PRESIGNED_MANIFEST=<path/to/file.json>.
                        AECIS pre-signs URLs for us, we look them
                        up by filepath. Right call for a one-shot
                        seed when AECIS won't share secrets.
      4. UNSIGNED     — fallback. <base>/<filepath>. 403s on the
                        real private bucket; kept for dry-runs.
    """
    if os.environ.get("AECIS_NGINX_HASH_KEY") and os.environ.get("AECIS_LINK_API_URL"):
        minutes = int(os.environ.get("AECIS_NGINX_EXPIRE_MIN", "60"))
        def _securelink(filepath: str, user_title: str = "") -> str:
            return _build_securelink(filepath, user_title) or ""
        return _securelink, f"securelink (TTL {minutes}m, base={os.environ['AECIS_LINK_API_URL']})"

    bucket = os.environ.get("AECIS_S3_BUCKET")
    region = os.environ.get("AECIS_S3_REGION")
    ak = os.environ.get("AECIS_S3_ACCESS_KEY_ID")
    sk = os.environ.get("AECIS_S3_SECRET_ACCESS_KEY")
    manifest_path = os.environ.get("AECIS_PRESIGNED_MANIFEST")

    if all([bucket, region, ak, sk]):
        import boto3
        from botocore.config import Config
        client = boto3.client(
            "s3", region_name=region,
            aws_access_key_id=ak, aws_secret_access_key=sk,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
        expires = int(os.environ.get("AECIS_PRESIGN_EXPIRES", "3600"))
        expires = max(60, min(expires, 604800))
        def _sign(filepath: str, user_title: str = "") -> str:
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": filepath},
                ExpiresIn=expires,
            )
        return _sign, f"iam-signed (TTL {expires}s)"

    if manifest_path:
        from pathlib import Path as _P
        p = _P(manifest_path)
        if not p.exists():
            sys.stderr.write(f"ERROR: manifest not found at {manifest_path}\n")
            sys.exit(2)
        urls = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(urls, list):
            urls = {
                r.get("filepath", ""): (r.get("url") or r.get("s3_url") or "")
                for r in urls if isinstance(r, dict)
            }
        def _lookup(filepath: str, user_title: str = "") -> str:
            return urls.get(filepath, "")
        return _lookup, f"manifest ({len(urls)} entries)"

    base = resolve_s3_base()
    def _unsigned(filepath: str, user_title: str = "") -> str:
        return f"{base}/{filepath}"
    return _unsigned, "UNSIGNED — will 403 against a private bucket"


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
                "user_title": (row[59] or "").strip(),
            }
            n += 1
            if limit and n >= limit:
                return


def download_one(url: str, dest: Path, timeout: int = 30) -> tuple[str, int]:
    """Returns (status, bytes). status in {ok, skip, fail}.

    urllib.request follows 302 redirects by default (HTTPRedirectHandler
    is in the default opener). That matters because AECIS's SecureLink
    endpoint may either stream the file inline OR 302 to a fresh AWS
    presigned URL — both flows resolve transparently here."""
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

    signer, mode = build_aecis_signer()
    sys.stdout.write(f"URL mode: {mode}\n")

    rows = list(iter_hse_rows(args.limit))
    sys.stdout.write(f"HSE rows to fetch: {len(rows)}\n")

    if args.dry_run:
        for r in rows[:20]:
            url = signer(r["filepath"], r.get("user_title", "")) or "<no URL — signer returned empty>"
            sys.stdout.write(f"  {url[:200]}\n")
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
        url = signer(row["filepath"], row.get("user_title", ""))
        if not url:
            sys.stderr.write(f"  SKIP {row['filepath']}: signer returned empty URL\n")
            return row, "fail", 0
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
