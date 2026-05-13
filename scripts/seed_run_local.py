"""One-shot local orchestrator for the AECIS seed pipeline.

LOCAL-ONLY: this script does everything on the workstation that runs
it. No VPS deploy, no production code path. The pipeline:

  1. Sanity-check environment (Issue_Gen/ present, .venv-webapp/, etc.)
  2. Start a local uvicorn dev server on 127.0.0.1:<port> with
     AUTH_REQUIRED=0 (skipped if a server is already listening there).
  3. Run seed_download_aecis_photos.py to pull HSE photos via the
     AECIS SecureLink proxy (or fallback IAM-signed) into
     Issue_Gen/photos/.
  4. Run seed_assign_via_playwright.py against the local dev server
     to create batches and upload chunks of N photos at a time.
  5. Stop the dev server if we started it.

Why local-first: the Issue_Gen/ dataset bundle isn't shipped to the
VPS by default (it's bundled in the repo but excluded from the
deploy checkout); pulling 2,400+ photos through the live service's
classify worker would also dominate its OpenRouter spend for the
day. Running locally points at the same Supabase + R2 + OpenRouter
infrastructure through the dev .env, but keeps the heavy lifting
(disk I/O, Playwright orchestration, retry loops) off the VPS.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/seed_run_local.py \\
        [--port 8765] \\
        [--limit-photos N] \\
        [--chunk-size 20] \\
        [--auto-confirm] [--confirm-threshold 0.85] \\
        [--skip-download] [--skip-assign] \\
        [--dry-run]

Env vars consumed:
    AECIS_NGINX_HASH_KEY + AECIS_LINK_API_URL   SecureLink mode (preferred)
    AECIS_S3_*                                  IAM-signed fallback
    AECIS_PRESIGNED_MANIFEST                    pre-signed JSON manifest
    AUTH_REQUIRED=0 is forced regardless of host environment.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv-webapp" / "Scripts" / "python.exe"
ISSUE_GEN_CSV = REPO_ROOT / "Issue_Gen" / "Issue_Gen" / "result_after_query.csv"
PHOTOS_DIR = REPO_ROOT / "Issue_Gen" / "photos"
DOWNLOADER = REPO_ROOT / "scripts" / "seed_download_aecis_photos.py"
ASSIGNER = REPO_ROOT / "scripts" / "seed_assign_via_playwright.py"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_env(args) -> None:
    """Validate the workstation is ready before kicking off downloads."""
    errs = []
    if not PYTHON.exists():
        errs.append(f"venv python missing: {PYTHON}")
    if not ISSUE_GEN_CSV.exists():
        errs.append(f"seed CSV missing: {ISSUE_GEN_CSV}")
    if not DOWNLOADER.exists():
        errs.append(f"downloader missing: {DOWNLOADER}")
    if not ASSIGNER.exists():
        errs.append(f"assigner missing: {ASSIGNER}")

    if not args.skip_download:
        # Confirm one of the signer modes is configured. Mirrors the
        # priority list in scripts/seed_download_aecis_photos.py
        # build_aecis_signer().
        signer_ok = (
            (os.environ.get("AECIS_NGINX_HASH_KEY") and os.environ.get("AECIS_LINK_API_URL"))
            or all(os.environ.get(k) for k in (
                "AECIS_S3_BUCKET", "AECIS_S3_REGION",
                "AECIS_S3_ACCESS_KEY_ID", "AECIS_S3_SECRET_ACCESS_KEY"
            ))
            or os.environ.get("AECIS_PRESIGNED_MANIFEST")
        )
        if not signer_ok:
            errs.append(
                "No AECIS access configured. Set ONE of:\n"
                "  • AECIS_NGINX_HASH_KEY + AECIS_LINK_API_URL  (SecureLink, recommended)\n"
                "  • AECIS_S3_BUCKET + AECIS_S3_REGION + AECIS_S3_ACCESS_KEY_ID + AECIS_S3_SECRET_ACCESS_KEY  (IAM)\n"
                "  • AECIS_PRESIGNED_MANIFEST=path/to/manifest.json  (pre-signed bundle)"
            )

    if errs:
        sys.stderr.write("Environment problems:\n")
        for e in errs:
            sys.stderr.write(f"  - {e}\n")
        sys.exit(2)


def _start_dev_server(port: int):
    """Spawn uvicorn in the background, wait for /api/batches to
    respond, return the Popen handle. Caller stops it on exit."""
    sys.stdout.write(f"Starting local dev server on 127.0.0.1:{port} (AUTH_REQUIRED=0)…\n")
    env = {**os.environ, "AUTH_REQUIRED": "0"}
    # Force UTF-8 stdout from uvicorn so its logs don't choke on the
    # Vietnamese accents in seed photo metadata when proxied through.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "webapp.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait up to 30 s for the port to open.
    for _ in range(60):
        if _port_open("127.0.0.1", port):
            sys.stdout.write("  ready.\n")
            return proc
        time.sleep(0.5)
    proc.terminate()
    sys.stderr.write("ERROR: dev server failed to come up on time.\n")
    sys.exit(2)


def _run(label: str, argv: list[str]) -> int:
    """Run a subprocess streaming its output, return exit code."""
    sys.stdout.write(f"\n=== {label} ===\n")
    sys.stdout.write(f"$ {' '.join(argv)}\n")
    return subprocess.call(argv, cwd=str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765, help="local dev server port")
    ap.add_argument("--limit-photos", type=int, default=0,
                    help="cap photos downloaded + assigned (0 = all ~2,444)")
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="photos per batch in the assign stage")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="auto-confirm high-conf AI picks during assign")
    ap.add_argument("--confirm-threshold", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel download workers")
    ap.add_argument("--skip-download", action="store_true",
                    help="re-use photos already in Issue_Gen/photos/")
    ap.add_argument("--skip-assign", action="store_true",
                    help="download only; don't upload to local server")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, run no fetches/uploads")
    args = ap.parse_args()

    _check_env(args)

    if args.skip_download and args.skip_assign:
        sys.stderr.write("Both --skip-download and --skip-assign — nothing to do.\n")
        sys.exit(2)

    # === Stage 1: download ===
    if not args.skip_download:
        dl_args = [str(PYTHON), str(DOWNLOADER),
                   "--workers", str(args.workers)]
        if args.limit_photos:
            dl_args += ["--limit", str(args.limit_photos)]
        if args.dry_run:
            dl_args += ["--dry-run"]
        rc = _run("Stage 1 — Download photos from AECIS", dl_args)
        if rc != 0:
            sys.stderr.write(f"downloader exited {rc}; aborting.\n")
            sys.exit(rc)

    # === Stage 2: assign (Playwright via local dev server) ===
    if args.skip_assign:
        sys.stdout.write("\n[--skip-assign] stopping after download.\n")
        return

    # Spin up dev server only if nothing's listening on the port.
    started_server = False
    server_proc = None
    if _port_open("127.0.0.1", args.port):
        sys.stdout.write(f"Reusing existing server on 127.0.0.1:{args.port}.\n")
    else:
        server_proc = _start_dev_server(args.port)
        started_server = True

    try:
        as_args = [str(PYTHON), str(ASSIGNER),
                   "--base", f"http://127.0.0.1:{args.port}/",
                   "--chunk-size", str(args.chunk_size)]
        if args.limit_photos:
            as_args += ["--limit", str(args.limit_photos)]
        if args.auto_confirm:
            as_args += ["--auto-confirm",
                        "--confirm-threshold", str(args.confirm_threshold)]
        if args.dry_run:
            as_args += ["--dry-run"]
        rc = _run("Stage 2 — Assign via Playwright (local UI)", as_args)
    finally:
        if started_server and server_proc and server_proc.poll() is None:
            sys.stdout.write("\nStopping dev server we started…\n")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    if rc != 0:
        sys.stderr.write(f"assigner exited {rc}.\n")
        sys.exit(rc)

    sys.stdout.write("\nSeed pipeline complete.\n")


if __name__ == "__main__":
    main()
