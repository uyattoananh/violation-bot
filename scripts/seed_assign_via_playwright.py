"""Drive the HSE Detector UI via Playwright to upload + auto-assign
the AECIS seed photos downloaded by seed_download_aecis_photos.py.

Flow per chunk of N photos:
  1. Open the app shell (AUTH_REQUIRED=0 for local, or pre-auth
     cookie for VPS).
  2. Click "Start new inspection" → new batch.
  3. Drop N photo files onto the dropzone (one shot, programmatic
     <input type=file> set).
  4. Wait for all N to finish classifying.
  5. If --auto-confirm: tap Confirm on every card that landed with
     high confidence (>= --confirm-threshold). Lower-confidence
     cards are left as pending so a human reviews them.
  6. Move to next chunk.

The manifest from seed_download_aecis_photos.py lets us stamp each
batch with the source AECIS project + issue id range so the seeded
data is traceable back to its origin.

Usage:
    ./.venv-webapp/Scripts/python.exe scripts/seed_assign_via_playwright.py \\
      [--base http://127.0.0.1:8765/] \\
      [--chunk-size 20] \\
      [--limit N] \\
      [--auto-confirm] [--confirm-threshold 0.85] \\
      [--dry-run]

Defaults: --base http://127.0.0.1:8765/, --chunk-size 20.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTO_ROOT = REPO_ROOT / "Issue_Gen" / "photos"
MANIFEST = PHOTO_ROOT / "manifest.jsonl"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_manifest(limit: int = 0):
    """Return list of {filepath, issue_id, project_id, issue_name,
    description, bytes} dicts from the downloader's manifest."""
    if not MANIFEST.exists():
        sys.stderr.write(
            f"ERROR: manifest missing at {MANIFEST}. Run "
            "seed_download_aecis_photos.py first.\n"
        )
        sys.exit(2)
    out = []
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            local = PHOTO_ROOT / rec["filepath"]
            if not local.exists():
                continue
            rec["local_path"] = str(local)
            out.append(rec)
            if limit and len(out) >= limit:
                break
    return out


async def upload_chunk(page, photos, *, auto_confirm: bool, confirm_threshold: float, label: str):
    """Upload one chunk of photos via the dropzone, wait for
    classification, optionally auto-confirm high-confidence picks.
    Returns counts: {uploaded, classified, confirmed, low_conf}."""
    counts = {"uploaded": 0, "classified": 0, "confirmed": 0, "low_conf": 0}

    # Click "Start new inspection" CTA. The selector is the empty-state
    # button on the list view OR the in-batch "+ Add" button.
    new_btn = page.locator("#btn-new-batch-from-list, .btn-new-batch").first
    await new_btn.click(timeout=10_000)
    await page.wait_for_timeout(400)

    # Rename the batch to the chunk label (so the seeded batches are
    # tagged with their source AECIS project id range).
    label_input = page.locator("#batch-label-input")
    if await label_input.count():
        await label_input.fill(label)
        await label_input.press("Tab")

    # Drop files onto the hidden <input type=file>. The picker
    # uses a labeled "Choose photos" / "Take photo" button on
    # mobile, but on desktop a single <input> behind the dropzone
    # accepts a programmatic .set_input_files() call.
    file_input = page.locator('input[type="file"]').first
    paths = [p["local_path"] for p in photos]
    await file_input.set_input_files(paths)
    counts["uploaded"] = len(paths)

    # Wait for cards to mount + classification to finish. The
    # classification worker takes ~1-3 s per photo; with N parallel
    # workers it's roughly N seconds for a chunk of 20.
    # Strategy: poll for predicted state on each .card, time out at
    # 5 minutes for the chunk.
    deadline = asyncio.get_event_loop().time() + 300
    classified = 0
    while asyncio.get_event_loop().time() < deadline:
        classified = await page.locator(".card[data-status='predicted'], .card[data-status='reviewed']").count()
        if classified >= len(paths):
            break
        await page.wait_for_timeout(1500)
    counts["classified"] = classified

    if not auto_confirm:
        return counts

    # Auto-confirm: tap .btn-confirm on every card whose conf-pct
    # text shows >= confirm_threshold * 100. The conf-pct element
    # text reads "· 92%" for a 92% prediction; we parse the integer.
    cards = await page.locator(".card[data-status='predicted']").all()
    threshold_pct = int(confirm_threshold * 100)
    for card in cards:
        try:
            pct_text = (await card.locator(".conf-pct").inner_text(timeout=2_000)).strip()
        except Exception:
            counts["low_conf"] += 1
            continue
        # "· 92%" → "92" → 92
        digits = "".join(c for c in pct_text if c.isdigit())
        if not digits:
            counts["low_conf"] += 1
            continue
        if int(digits) < threshold_pct:
            counts["low_conf"] += 1
            continue
        try:
            await card.locator(".btn-confirm").click(timeout=5_000)
            counts["confirmed"] += 1
            await page.wait_for_timeout(150)
        except Exception:
            counts["low_conf"] += 1

    return counts


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765/", help="webapp base URL")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap total photos (0 = all)")
    ap.add_argument("--auto-confirm", action="store_true", help="confirm high-conf picks")
    ap.add_argument("--confirm-threshold", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true", help="print plan, don't run Playwright")
    args = ap.parse_args()

    photos = load_manifest(args.limit)
    if not photos:
        sys.stderr.write("ERROR: no photos in manifest.\n")
        sys.exit(2)

    sys.stdout.write(
        f"Photos to assign:  {len(photos)}\n"
        f"Chunk size:        {args.chunk_size}\n"
        f"Auto-confirm:      {args.auto_confirm} (>={args.confirm_threshold * 100:.0f}%)\n"
    )
    if args.dry_run:
        for i, p in enumerate(photos[:5]):
            sys.stdout.write(f"  [{i}] issue {p['issue_id']} ({p['project_id']}) "
                             f"-- {p['issue_name'][:50]}\n")
        sys.stdout.write(f"  ... ({len(photos)} total)\n")
        return

    from playwright.async_api import async_playwright

    total = {"uploaded": 0, "classified": 0, "confirmed": 0, "low_conf": 0}
    t0 = time.perf_counter()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await ctx.new_page()
        await page.goto(args.base, wait_until="networkidle")

        for i in range(0, len(photos), args.chunk_size):
            chunk = photos[i : i + args.chunk_size]
            first_issue = chunk[0]["issue_id"]
            last_issue = chunk[-1]["issue_id"]
            project = chunk[0]["project_id"]
            label = f"AECIS seed P_{project} ({first_issue}–{last_issue})"
            sys.stdout.write(f"\n--- chunk {i // args.chunk_size + 1}: "
                             f"{len(chunk)} photos, label='{label}' ---\n")
            counts = await upload_chunk(
                page, chunk,
                auto_confirm=args.auto_confirm,
                confirm_threshold=args.confirm_threshold,
                label=label,
            )
            for k in total:
                total[k] += counts[k]
            sys.stdout.write(
                f"  uploaded={counts['uploaded']} "
                f"classified={counts['classified']} "
                f"confirmed={counts['confirmed']} "
                f"low_conf={counts['low_conf']}\n"
            )
            # Navigate back to the list before the next chunk.
            await page.evaluate("() => history.pushState({}, '', '/')")
            await page.evaluate("() => window.dispatchEvent(new PopStateEvent('popstate'))")
            await page.wait_for_timeout(500)

        await browser.close()

    elapsed = time.perf_counter() - t0
    sys.stdout.write(
        f"\nDone in {elapsed:.1f}s — total: uploaded={total['uploaded']} "
        f"classified={total['classified']} confirmed={total['confirmed']} "
        f"low_conf={total['low_conf']}\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
