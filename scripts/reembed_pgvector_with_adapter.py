"""Re-embed every pgvector row from its source photo on disk through the
adapter-aware `embed_image()`. This is the safe way to rebuild the index
after training a new CLIP adapter: every row gets exactly one fresh CLIP
pass + one adapter apply, regardless of what state the pgvector column
was in beforehand (partial updates from earlier runs, mixed states, etc.).

Call after `scripts/finetune_clip.py` saves `src/clip_adapter.pt`. From
this point forward, query and index embeddings live in the same adapter-
specialized space.

Pipeline:
  1. Scan ~/Desktop/aecis-violations — build sha256 → disk_path index
     from each metadata.json's `photos[].sha256` + `photos[].file`
  2. Fetch pgvector shas
  3. For each sha: embed_image(disk_path) (runs CLIP + adapter)
  4. UPDATE photo_embeddings.embedding

Sequential with retry — parallel HTTP/2 updates overwhelm Supabase's
stream limit.

Usage:
  python scripts/reembed_pgvector_with_adapter.py
  python scripts/reembed_pgvector_with_adapter.py --dry-run
  python scripts/reembed_pgvector_with_adapter.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("reembed")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DEFAULT_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _build_sha_index(root: Path) -> dict[str, Path]:
    """Walk the archive and build {sha256: Path} by reading each metadata.json's
    photos[] array. Only photos whose file actually exists are included."""
    idx: dict[str, Path] = {}
    n_meta = 0
    for meta_path in root.glob("*/*/metadata.json"):
        n_meta += 1
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for ph in (meta.get("photos") or []):
            sha = ph.get("sha256")
            fname = ph.get("file")
            if not sha or not fname:
                continue
            img = meta_path.parent / fname
            if img.exists():
                idx.setdefault(sha, img)
    log.info("Scanned %d metadata.json files — indexed %d unique sha256→disk paths",
             n_meta, len(idx))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        log.error("SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required"); return 2

    from src.clip_adapter import load_adapter, ADAPTER_PATH
    if load_adapter() is None:
        log.warning("No adapter at %s — re-embedding with raw CLIP (rollback mode)", ADAPTER_PATH)

    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_ROOT
    if not root.exists():
        log.error("Archive root not found: %s", root); return 1

    # 1. Build sha → disk path index
    sha_to_path = _build_sha_index(root)

    from supabase import create_client
    db = create_client(url, key)

    # 2. Fetch pgvector shas
    rows: list[dict] = []
    offset = 0
    while True:
        batch = (
            db.table("photo_embeddings")
              .select("sha256")
              .range(offset, offset + 499).execute().data or []
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 500:
            break
        offset += 500
    log.info("Fetched %d pgvector rows", len(rows))

    sha_list = [r["sha256"] for r in rows]
    missing = [s for s in sha_list if s not in sha_to_path]
    present = [s for s in sha_list if s in sha_to_path]
    log.info("sha→disk mapped for %d/%d rows (%d missing)",
             len(present), len(sha_list), len(missing))
    if missing:
        log.warning("  first few missing: %s", [s[:10] for s in missing[:5]])

    if args.limit:
        present = present[:args.limit]
        log.info("Limited to %d", len(present))

    if args.dry_run:
        log.info("dry-run: would re-embed %d rows", len(present))
        return 0

    # 3. Re-embed via embed_image (adapter-aware) + UPDATE
    from PIL import Image
    from src.embeddings import _load_model, _apply_adapter
    import numpy as np

    model = _load_model()
    done = 0
    failed = 0

    for idx, sha in enumerate(present, 1):
        img_path = sha_to_path[sha]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            log.warning("decode failed for %s (%s): %s", sha[:10], img_path, e)
            failed += 1
            continue

        try:
            vec = model.encode([img], normalize_embeddings=True, convert_to_numpy=True)[0]
            vec = _apply_adapter(vec.astype(np.float32).reshape(1, -1))[0]
        except Exception as e:  # noqa: BLE001
            log.warning("embed failed for %s: %s", sha[:10], e)
            failed += 1
            continue

        ok = False
        for attempt in range(3):
            try:
                db.table("photo_embeddings").update(
                    {"embedding": vec.tolist()}
                ).eq("sha256", sha).execute()
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log.warning("update failed for %s: %s", sha[:10], e)
                else:
                    time.sleep(0.5 * (attempt + 1))
        if ok:
            done += 1
        else:
            failed += 1

        if idx % 50 == 0 or idx == len(present):
            log.info("[%d/%d] done=%d failed=%d", idx, len(present), done, failed)

    log.info("Done. %d updated  %d failed  %d missing-from-disk",
             done, failed, len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
