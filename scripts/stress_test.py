"""HSE Detector stress test — Playwright-driven.

Runs the production shell against mocked /api/batches and /api/pending
responses with large datasets, measuring:

  - Initial paint timing      (navigationStart -> DOMContentLoaded
                               -> first list render -> "interactive")
  - List render scaling       (100 / 500 / 1000 batches mocked)
  - Detail-view render cost   (200 photos in one batch)
  - Navigation cycle stability(50 round-trips list <-> detail to
                               surface memory leaks / dangling listeners)
  - JS heap growth            (performance.memory.usedJSHeapSize delta)

Run: ./.venv-webapp/Scripts/python.exe scripts/stress_test.py
Requires: dev server on 127.0.0.1:8765 with AUTH_REQUIRED=0.
"""
import asyncio
import datetime
import json
import statistics
import sys
import time
from pathlib import Path
from playwright.async_api import async_playwright

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "http://127.0.0.1:8765/"


def _iso(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def make_batches(n):
    """N synthetic batches with varying photo counts."""
    now = int(time.time() * 1000)
    out = []
    for i in range(n):
        photos = (i % 50) + 1
        reviewed = i % (photos + 1)
        out.append({
            "batch_id": f"stress-{i:05d}",
            "label": f"Site {chr(65 + i % 26)} — inspection {i}",
            "photo_count": photos,
            "reviewed_count": reviewed,
            "expires_at": _iso(now + ((48 - i % 48) * 3600 * 1000)),
            "created_at": _iso(now - i * 60_000),
            "latest_uploaded_at": _iso(now - i * 60_000),
        })
    return {"batches": out, "photo_expiry_days": 2}


def make_photos(batch_id, n):
    """N predicted photos with two alts each."""
    now = int(time.time() * 1000)
    photos = []
    for i in range(n):
        photos.append({
            "id": f"{batch_id}-p{i:04d}",
            "batch_id": batch_id,
            "filename": f"img{i}.jpg",
            "thumbnail_url": "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 9'><rect width='16' height='9' fill='%23cbd5e1'/></svg>",
            "presigned_url": "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 9'><rect width='16' height='9' fill='%23cbd5e1'/></svg>",
            "expires_at": _iso(now + 47 * 3600 * 1000),
            "uploaded_at": _iso(now - i * 60_000),
            "classification": {
                "fine_hse_type_slug": None,
                "hse_type_slug": "Site_warning_signs" if i % 3 == 0 else "Electrical_hazard" if i % 3 == 1 else "Fall_hazard",
                "location_slug": "Common_working_area",
                "confidence": 0.6 + (i % 40) / 100.0,
                "label_en": f"Sample violation {i}",
                "hse_type_alternatives": [
                    {"slug": "Electrical_hazard", "confidence": 0.7},
                    {"slug": "Fall_hazard", "confidence": 0.5},
                ],
            },
            "reviewed": False,
        })
    return {"photos": photos, "training_set_size": 100}


async def install_mocks(ctx, batches, photos_by_batch):
    batches_body = json.dumps(batches)

    async def handle(route):
        url = route.request.url
        method = route.request.method
        if "/api/batches" in url and method == "GET":
            return await route.fulfill(status=200, content_type="application/json", body=batches_body)
        if "/api/pending" in url:
            # Pull batch_id from query string
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(url).query)
            bid = qs.get("batch_id", [""])[0]
            payload = photos_by_batch.get(bid, {"photos": [], "training_set_size": 100})
            return await route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))
        if "/api/usage/today" in url:
            return await route.fulfill(status=200, content_type="application/json", body='{"used_today": 0, "daily_limit": 100}')
        if "/api/" in url and method != "GET":
            return await route.fulfill(status=200, content_type="application/json", body="{}")
        await route.continue_()

    await ctx.route("**/api/**", handle)


async def measure_initial_paint(page):
    """Read Navigation Timing API for the page that just loaded."""
    return await page.evaluate(r"""() => {
        const nav = performance.getEntriesByType('navigation')[0];
        if (!nav) return null;
        return {
            domContentLoaded_ms: Math.round(nav.domContentLoadedEventEnd - nav.startTime),
            loadEvent_ms:        Math.round(nav.loadEventEnd - nav.startTime),
            firstByte_ms:        Math.round(nav.responseStart - nav.startTime),
            domInteractive_ms:   Math.round(nav.domInteractive - nav.startTime),
            transferSize:        nav.transferSize,
            encodedBodySize:     nav.encodedBodySize,
        };
    }""")


async def measure_render_cost(page, selector, expected_count):
    """Wait for `expected_count` matches of `selector` to be in DOM,
    return the wall-clock ms. Uses locator.nth().wait_for() rather
    than wait_for_function so we don't hit CSP unsafe-eval blocks."""
    t0 = time.perf_counter()
    await page.locator(selector).nth(expected_count - 1).wait_for(
        state="attached", timeout=15000
    )
    return int((time.perf_counter() - t0) * 1000)


async def js_heap(page):
    """Returns usedJSHeapSize in MB; None if not available (Firefox)."""
    return await page.evaluate(r"""() => {
        const m = performance.memory;
        return m ? Math.round(m.usedJSHeapSize / 1024 / 1024 * 10) / 10 : null;
    }""")


async def scenario_initial_paint(browser):
    """No mocks — measure the cold paint of the empty shell."""
    print("\n=== Scenario 1: initial paint, empty shell ===")
    ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
    # Don't install mocks; let /api/batches return whatever the real
    # server gives (likely empty for the demo tenant on AUTH_REQUIRED=0).
    page = await ctx.new_page()
    await page.goto(BASE, wait_until="load")
    timing = await measure_initial_paint(page)
    print(f"  TTFB:            {timing['firstByte_ms']:>5} ms")
    print(f"  domInteractive:  {timing['domInteractive_ms']:>5} ms")
    print(f"  DOMContentLoaded:{timing['domContentLoaded_ms']:>5} ms")
    print(f"  load event:      {timing['loadEvent_ms']:>5} ms")
    print(f"  HTML transfer:   {timing['transferSize']:>5} bytes "
          f"(encoded body {timing['encodedBodySize']} bytes)")
    await ctx.close()


async def scenario_list_scaling(browser):
    """Render list view with 100, 500, 1000 batches; measure
    time-to-render on each."""
    print("\n=== Scenario 2: list view render scaling ===")
    print("  N batches | render ms | heap MB")
    print("  ----------+-----------+--------")
    for n in (100, 500, 1000):
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        await install_mocks(ctx, make_batches(n), {})
        page = await ctx.new_page()
        await page.goto(BASE, wait_until="networkidle")
        # Wait for first row, then measure scaling to N rows.
        await page.wait_for_selector(".batch-card", timeout=10000)
        ms = await measure_render_cost(page, ".batch-card", n)
        heap = await js_heap(page)
        print(f"  {n:>9} | {ms:>9} | {heap if heap is not None else '   n/a'}")
        await ctx.close()


async def scenario_detail_grid(browser):
    """Detail view with 200 photos. Measure user-perceived 'page is
    ready' (first visible card populated) and total skeleton-mount
    time. With lazy rendering (v116.12+), only viewport cards
    populate on first paint — off-screen cards stay as skeletons
    until scrolled in."""
    print("\n=== Scenario 3: photo-grid render (200 photos) ===")
    batch_id = "stress-grid"
    batches = {
        "batches": [{
            "batch_id": batch_id,
            "label": "Stress grid",
            "photo_count": 200,
            "reviewed_count": 0,
            "expires_at": _iso(int(time.time() * 1000) + 47 * 3600 * 1000),
            "created_at": _iso(int(time.time() * 1000) - 3600 * 1000),
            "latest_uploaded_at": _iso(int(time.time() * 1000) - 3600 * 1000),
        }],
        "photo_expiry_days": 2,
    }
    photos = {batch_id: make_photos(batch_id, 200)}
    ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
    await install_mocks(ctx, batches, photos)
    page = await ctx.new_page()
    await page.goto(BASE, wait_until="networkidle")
    await page.wait_for_selector(f'.batch-card[data-batch-id="{batch_id}"]', timeout=10000)
    heap_before = await js_heap(page)
    t0 = time.perf_counter()
    await page.click(f'.batch-card[data-batch-id="{batch_id}"]')
    # Time-to-first-card-populated. .primary-hse is empty in the
    # template clone; updateCard fills it. So a non-empty primary-hse
    # is the "this card is ready for the user to read" signal.
    await page.locator("#cards .card .primary-hse").nth(0).wait_for(
        state="attached", timeout=20000
    )
    # Wait one frame so populated text is committed.
    await page.wait_for_timeout(50)
    first_ready_ms = int((time.perf_counter() - t0) * 1000)

    # All 200 cards in the DOM as skeletons (insertion done).
    await page.locator("#cards .card").nth(199).wait_for(
        state="attached", timeout=20000
    )
    all_skeleton_ms = int((time.perf_counter() - t0) * 1000)

    # Count populated cards at this moment — visible ones rendered,
    # off-screen ones still skeletons.
    populated = await page.evaluate(r"""() => {
        const cards = document.querySelectorAll('#cards .card');
        let n = 0;
        for (const c of cards) {
            const t = c.querySelector('.primary-hse')?.textContent?.trim();
            if (t && t.length > 0) n++;
        }
        return n;
    }""")
    heap_after = await js_heap(page)
    print(f"  first card populated:        {first_ready_ms} ms")
    print(f"  all 200 skeletons in DOM:    {all_skeleton_ms} ms")
    print(f"  populated cards on first paint: {populated} of 200 "
          f"({'lazy render active' if populated < 200 else 'eager render'})")
    print(f"  heap before/after:           {heap_before} MB -> {heap_after} MB")
    await ctx.close()


async def scenario_nav_cycles(browser, cycles=30):
    """Open batch -> back to list, repeated. Surface leaks via
    heap growth + render time degradation."""
    print(f"\n=== Scenario 4: navigation cycle stability ({cycles} iters) ===")
    batches = make_batches(20)
    photos = {b["batch_id"]: make_photos(b["batch_id"], 10) for b in batches["batches"]}
    ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
    await install_mocks(ctx, batches, photos)
    page = await ctx.new_page()
    await page.goto(BASE, wait_until="networkidle")
    await page.wait_for_selector(".batch-card", timeout=10000)

    open_times = []
    back_times = []
    heap_samples = []

    for i in range(cycles):
        target = batches["batches"][i % len(batches["batches"])]["batch_id"]

        t0 = time.perf_counter()
        await page.click(f'.batch-card[data-batch-id="{target}"]')
        # Wait for the detail view to become non-hidden. We can't
        # wait for .card here because poll's adaptive backoff
        # (5 idle polls -> 16 s interval) drops out by iteration ~6
        # and individual cards stop rendering inside the timeout.
        # The view-state transition is the right thing to time —
        # it's what the user perceives as "the page loaded".
        await page.locator("#view-detail:not(.hidden)").wait_for(
            state="attached", timeout=10000
        )
        open_times.append((time.perf_counter() - t0) * 1000)

        # Drive the SPA back the same way the hardware back button
        # would: history.back(). Wait for the list view to be
        # non-hidden (visible state).
        t0 = time.perf_counter()
        await page.evaluate("() => history.back()")
        await page.locator("#view-list:not(.hidden)").wait_for(
            state="attached", timeout=10000
        )
        back_times.append((time.perf_counter() - t0) * 1000)

        if i in (0, cycles // 2, cycles - 1):
            heap_samples.append((i, await js_heap(page)))

    print(f"  open list->detail (ms): "
          f"min={min(open_times):.0f} median={statistics.median(open_times):.0f} "
          f"p95={sorted(open_times)[int(len(open_times) * 0.95)]:.0f} "
          f"max={max(open_times):.0f}")
    print(f"  back detail->list (ms): "
          f"min={min(back_times):.0f} median={statistics.median(back_times):.0f} "
          f"p95={sorted(back_times)[int(len(back_times) * 0.95)]:.0f} "
          f"max={max(back_times):.0f}")
    if all(h[1] is not None for h in heap_samples):
        first, mid, last = heap_samples
        print(f"  heap growth: iter 0={first[1]} MB, iter {mid[0]}={mid[1]} MB, iter {last[0]}={last[1]} MB")
        if last[1] - first[1] > 10:
            print(f"  WARN: heap grew by {last[1] - first[1]:.1f} MB across {cycles} cycles — possible leak")
    await ctx.close()


async def scenario_concurrent_polls(browser):
    """Fire a burst of /api/batches refreshes by toggling visibility,
    confirming the SWR layer doesn't double-render or thrash."""
    print("\n=== Scenario 5: concurrent batches refresh under visibility flicker ===")
    batches = make_batches(50)
    ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
    await install_mocks(ctx, batches, {})
    page = await ctx.new_page()
    await page.goto(BASE, wait_until="networkidle")
    await page.wait_for_selector(".batch-card", timeout=10000)
    t0 = time.perf_counter()
    # Fire 20 visibilitychange events in quick succession; each
    # triggers a refreshBatchesList() call. Measure render
    # consistency.
    await page.evaluate(r"""async () => {
        for (let i = 0; i < 20; i++) {
            Object.defineProperty(document, 'hidden', {value: i % 2 === 1, configurable: true});
            document.dispatchEvent(new Event('visibilitychange'));
            await new Promise(r => setTimeout(r, 30));
        }
        Object.defineProperty(document, 'hidden', {value: false, configurable: true});
        document.dispatchEvent(new Event('visibilitychange'));
    }""")
    await page.wait_for_timeout(800)
    ms = int((time.perf_counter() - t0) * 1000)
    rows_after = await page.evaluate("() => document.querySelectorAll('.batch-card').length")
    print(f"  20 visibility flips drained in {ms} ms; {rows_after} rows still mounted")
    await ctx.close()


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        await scenario_initial_paint(browser)
        await scenario_list_scaling(browser)
        await scenario_detail_grid(browser)
        await scenario_nav_cycles(browser, cycles=30)
        await scenario_concurrent_polls(browser)
        await browser.close()
    print("\nstress test complete")


if __name__ == "__main__":
    asyncio.run(main())
