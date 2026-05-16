"""Thin wrapper that loads AECIS IAM credentials from
Issue_Gen/rnd-user_accessKeys.csv into process env, sets the known
bucket + region, then exec's seed_download_aecis_photos.py with all
forwarded CLI args.

The credentials live in a gitignored CSV (see .gitignore:
Issue_Gen/*accessKeys*). This wrapper exists so the downloader can
be invoked without the operator having to set AECIS_S3_* env vars
in their shell each time — and without the values ever appearing
in logs or shell history.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/seed_download_with_iam.py \\
        --disciplines all --limit 50 --dry-run
    ./.venv-webapp/Scripts/python.exe scripts/seed_download_with_iam.py \\
        --disciplines all
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYS_CSV = REPO_ROOT / "Issue_Gen" / "rnd-user_accessKeys.csv"


def load_iam_into_env() -> None:
    """AWS console exports CSV with header:
       Access key ID,Secret access key
    """
    if not KEYS_CSV.exists():
        sys.stderr.write(
            f"ERROR: {KEYS_CSV} missing. This wrapper expects the AECIS "
            "IAM read-only key CSV (gitignored).\n"
        )
        sys.exit(2)
    with KEYS_CSV.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rec = next(r, None)
    if not rec:
        sys.stderr.write(f"ERROR: {KEYS_CSV} has no data row.\n")
        sys.exit(2)
    # Normalize keys (the AWS export uses spaces)
    norm = {k.strip().lower(): (v or "").strip() for k, v in rec.items()}
    ak = norm.get("access key id") or norm.get("accesskeyid")
    sk = norm.get("secret access key") or norm.get("secretaccesskey")
    if not ak or not sk:
        sys.stderr.write(f"ERROR: could not find access key + secret in {KEYS_CSV}.\n")
        sys.exit(2)
    # AECIS bucket + region — fixed per the db manager's method.txt
    os.environ.setdefault("AECIS_S3_BUCKET", "aecis-app")
    os.environ.setdefault("AECIS_S3_REGION", "ap-southeast-1")
    os.environ["AECIS_S3_ACCESS_KEY_ID"] = ak
    os.environ["AECIS_S3_SECRET_ACCESS_KEY"] = sk
    # AECIS_PHOTO_S3_BASE — virtual-hosted addressing
    os.environ.setdefault(
        "AECIS_PHOTO_S3_BASE",
        f"https://{os.environ['AECIS_S3_BUCKET']}.s3.{os.environ['AECIS_S3_REGION']}.amazonaws.com",
    )


def main() -> int:
    load_iam_into_env()
    # Replace this process with the downloader; argv preserves CLI args.
    downloader = REPO_ROOT / "scripts" / "seed_download_aecis_photos.py"
    argv = [sys.executable, str(downloader), *sys.argv[1:]]
    # On Windows, os.execv doesn't behave like POSIX; use subprocess instead.
    import subprocess
    return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main())
