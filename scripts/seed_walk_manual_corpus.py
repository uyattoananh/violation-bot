"""Walk ~/Desktop/aecis-violations/ and compute sha256 for every
*.jpg. Output a {sha256: relative_path} map so the eval endpoint
can look up the 3,015 photo_embeddings rows with label_source=manual.

The `manual` rows were originally embedded by auto_seed_from_disk.py
using source_paths like `autoseed/SLPXA/135004/32.jpg`. The folder
naming convention has changed since (current folders look like
`20230404-RMIT-DN-Sp_xp_...`), so the path-based lookup no longer
works. Sha256 lookup does — the file bytes haven't moved.

Output: tmp/manual_corpus_sha_map.json

Usage:
  ./.venv-webapp/Scripts/python.exe scripts/seed_walk_manual_corpus.py
  ./.venv-webapp/Scripts/python.exe scripts/seed_walk_manual_corpus.py --root /custom/path
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"
OUT_MAP = REPO / "tmp" / "manual_corpus_sha_map.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"corpus root (default: {DEFAULT_ROOT})")
    ap.add_argument("--extensions", default="jpg,jpeg,png",
                    help="comma-separated suffixes to scan")
    args = ap.parse_args()

    if not args.root.exists():
        sys.stderr.write(f"ERROR: corpus root missing at {args.root}\n")
        return 2

    suffixes = {f".{e.strip().lower()}" for e in args.extensions.split(",") if e.strip()}
    sha_map: dict[str, str] = {}
    t0 = time.perf_counter()
    n_scanned = n_dupe = 0
    for p in args.root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
        n_scanned += 1
        try:
            sha = _sha256(p)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  fail {p}: {e}\n")
            continue
        rel = str(p.relative_to(args.root)).replace("\\", "/")
        if sha in sha_map:
            n_dupe += 1
            continue
        sha_map[sha] = rel
        if n_scanned % 500 == 0:
            print(f"  scanned {n_scanned}, unique sha={len(sha_map)} "
                  f"(elapsed {time.perf_counter()-t0:.0f}s)")

    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_MAP.with_suffix(OUT_MAP.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "root": str(args.root),
        "scanned": n_scanned,
        "duplicates": n_dupe,
        "unique_sha": len(sha_map),
        "sha_to_relpath": sha_map,
    }, indent=2), encoding="utf-8")
    tmp.replace(OUT_MAP)
    print(f"\nscanned {n_scanned} files, {len(sha_map)} unique shas "
          f"({n_dupe} dupes) in {time.perf_counter()-t0:.0f}s")
    print(f"out: {OUT_MAP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
