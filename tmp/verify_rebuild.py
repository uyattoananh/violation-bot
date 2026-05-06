"""Capture both viewports for each commit during the rebuild."""
import asyncio, sys
from pathlib import Path
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
SHOTS = REPO / "tmp" / "rebuild"


async def shot_viewports(label, populate=None):
    SHOTS.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for name, w, h, mobile in [
            ("mobile", 375, 667, True),
            ("desktop", 1280, 800, False),
        ]:
            ctx = await browser.new_context(
                viewport={"width": w, "height": h},
                device_scale_factor=2 if mobile else 1,
                is_mobile=mobile, has_touch=mobile,
            )
            page = await ctx.new_page()
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            await page.goto("http://127.0.0.1:8088/", wait_until="networkidle")
            await page.wait_for_timeout(400)
            if populate:
                try:
                    await populate(page)
                    await page.wait_for_timeout(400)
                except Exception as e:
                    print(f"  [{name}] populate failed: {e}")
            await page.screenshot(path=str(SHOTS / f"{label}-{name}.png"), full_page=True)
            if errs:
                print(f"  [{name}] JS ERRORS:")
                for e in errs[:3]:
                    print(f"    {e[:160]}")
            else:
                print(f"  [{name}] OK ({w}x{h})")
            await ctx.close()
        await browser.close()


# Populator helpers
async def populate_detail_with_card(page):
    try: await page.click("#btn-new-batch-from-list", timeout=2000)
    except: pass
    await page.wait_for_timeout(400)
    await page.evaluate("""() => {
        const tmpl = document.getElementById('card-template');
        const cards = document.getElementById('cards');
        if (!tmpl || !cards) return;
        for (let i = 0; i < 3; i++) {
            const node = tmpl.content.firstElementChild.cloneNode(true);
            node.dataset.photoId = 'fake-' + i;
            node.dataset.status = 'predicted';
            cards.appendChild(node);
            node.querySelector('.state-pending')?.classList.add('hidden');
            node.querySelector('.state-predicted')?.classList.remove('hidden');
            const fineEl = node.querySelector('.primary-fine');
            if (fineEl) {
                fineEl.textContent = 'Site warning signs / barricades missing';
                fineEl.classList.remove('hidden');
            }
            const hseEl = node.querySelector('.primary-hse');
            if (hseEl) hseEl.textContent = 'Site warning signs missing';
            const conf = node.querySelector('.conf-pct');
            if (conf) conf.textContent = '· 90%';
            const pill = node.querySelector('.status-pill');
            if (pill) {
                pill.textContent = '✓';
                pill.className = 'status-pill text-xs font-bold leading-none w-5 h-5 grid place-items-center rounded-full bg-emerald-100 text-emerald-700';
                pill.classList.remove('hidden');
            }
            node.classList.add('band-high');
        }
        document.getElementById('empty-state')?.classList.add('hidden');
        document.getElementById('loading-state')?.classList.add('hidden');
        document.getElementById('toolbar')?.classList.remove('hidden');
        document.getElementById('toolbar2')?.classList.remove('hidden');
        const sub = document.getElementById('batch-subtitle');
        if (sub) sub.textContent = '12 photos · 4 reviewed';
    }""")


async def populate_picker_open(page):
    await populate_detail_with_card(page)
    await page.evaluate("""() => {
        const m = document.getElementById('fp-modal');
        if (!m) return;
        m.classList.remove('hidden');
        const list = document.querySelector('.fp-list');
        if (!list) return;
        list.innerHTML = '';
        const samples = [
            {label: 'Common eating / rest area unhygienic', count: 2},
            {label: 'Concrete pump / pour hazard', count: 7},
            {label: 'Confined space hazard', count: 9},
            {label: 'Drinking water area unhygienic', count: 2},
            {label: 'Electrical hazard (wiring, panel, equipment)', count: 36},
            {label: 'Excavation / pit / deep hole hazard', count: 24},
            {label: 'Fire prevention deficiency', count: 10},
            {label: 'Site warning signs / barricades missing or damaged', count: 11},
        ];
        for (const s of samples) {
            const btn = document.createElement('button');
            btn.className = 'fp-item';
            btn.innerHTML = `<div class="flex items-center gap-2"><div class="flex-1 min-w-0"><div class="font-medium">${s.label}</div><div class="fp-meta">${s.count} sub-types</div></div><svg class="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5"/></svg></div>`;
            list.appendChild(btn);
        }
    }""")


async def populate_list(page):
    await page.evaluate(r"""() => {
        const tmpl = document.getElementById('batch-card-template');
        const grid = document.getElementById('batches-grid');
        const empty = document.getElementById('batches-empty');
        const view = document.getElementById('view-list');
        if (!tmpl || !grid) return;
        view?.classList.remove('hidden');
        document.getElementById('view-batch')?.classList.add('hidden');
        document.getElementById('view-landing')?.classList.add('hidden');
        empty?.classList.add('hidden');
        grid.classList.remove('hidden');
        grid.innerHTML = '';
        const samples = [
            {label: 'Site A — North wing rebar inspection', photos: 24, reviewed: 18, expiry: 'Expires 38h', when: 'Today', pct: '75%'},
            {label: 'Crane lift area — boom radius check', photos: 12, reviewed: 12, expiry: '', when: 'Yesterday', pct: '100%'},
            {label: 'Scaffolding deck level 4', photos: 47, reviewed: 9, expiry: 'Expires 12h', when: '2 days ago', pct: '19%'},
            {label: 'Excavation pit shoring', photos: 8, reviewed: 0, expiry: 'Expires 47h', when: '3 days ago', pct: '0%'},
            {label: 'Confined space entry — tank 7', photos: 31, reviewed: 31, expiry: '', when: '4 days ago', pct: '100%'},
        ];
        for (const s of samples) {
            const node = tmpl.content.firstElementChild.cloneNode(true);
            node.querySelector('.batch-card-label').textContent = s.label;
            node.querySelector('.batch-card-photos').textContent = `${s.photos} photos`;
            if (s.reviewed) node.querySelector('.batch-card-reviewed').textContent = `· ${s.reviewed} reviewed`;
            node.querySelector('.batch-card-expiry-text').textContent = s.expiry;
            if (!s.expiry) node.querySelector('.batch-card-expiry').classList.add('hidden');
            node.querySelector('.batch-card-when').textContent = `· ${s.when}`;
            node.querySelector('.batch-card-pct').textContent = s.pct;
            grid.appendChild(node);
        }
    }""")


SCENARIOS = {
    "list": None,
    "list-populated": populate_list,
    "detail-empty": (lambda p: p.click("#btn-new-batch-from-list", timeout=2000)),
    "detail-card": populate_detail_with_card,
    "picker-open": populate_picker_open,
}


async def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "list"
    pop = SCENARIOS.get(label)
    print(f"=== {label} ===")
    await shot_viewports(label, pop)


if __name__ == "__main__":
    asyncio.run(main())
