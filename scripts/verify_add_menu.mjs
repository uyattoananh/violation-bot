// Visual + assertion check for v118.10: [+ Add] removed; the detail-view
// dropzone stays minimized-but-visible (Take photo / Choose photos menu)
// once the batch has photos. Dev DB is empty, so we drive the detail
// view + data-state directly to exercise both states.
import { createRequire } from 'module';
import fs from 'fs';

const BASE = process.env.BASE_URL || 'http://127.0.0.1:8765';
const SHOTS = 'tmp/addmenu-shots';
fs.mkdirSync(SHOTS, { recursive: true });

const DIRS = [
  'C:/Users/lang/.claude/skills/chrome-devtools/scripts/lib/',
  'C:/Users/lang/.claude/skills/chrome-devtools/scripts/',
  'C:/Users/lang/.claude/skills/chrome-devtools/',
];
let puppeteer = null;
for (const d of DIRS) { try { puppeteer = createRequire('file:///' + d + 'n.js')('puppeteer'); break; } catch (_) {} }
if (!puppeteer) puppeteer = (await import('puppeteer')).default;

const out = { checks: [], pass: true };
const rec = (name, ok, detail) => { out.checks.push({ name, ok, detail }); if (!ok) out.pass = false; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} — ${detail}`); };

const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox', '--disable-setuid-sandbox'] });
try {
  const page = await browser.newPage();
  // Bypass any HTTP/service-worker cache so the test reflects disk truth
  // (the SW caches /static/app.css?v=... by URL; a prior dev-server run
  // can leave a stale copy that masks fresh edits).
  await page.setCacheEnabled(false);
  await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
  await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 30000 });
  // Unregister any service worker + clear caches, then hard-reload so the
  // page's app.css comes straight from the dev server, not the SW cache.
  await page.evaluate(async () => {
    if (navigator.serviceWorker) {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map(r => r.unregister()));
    }
    if (window.caches) {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
    }
  });
  await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 30000 });

  // 1. The [+ Add] button must be gone from the DOM entirely.
  const addBtn = await page.$('#btn-add-photos');
  rec('[+ Add] button removed', addBtn === null, addBtn === null ? '#btn-add-photos not in DOM' : '#btn-add-photos STILL present');

  // Force the detail view visible + dropzone panel shown (empty DB has no
  // batch, so we reveal the static detail markup directly).
  await page.evaluate(() => {
    document.querySelectorAll('#view-landing, #view-list, #view-error').forEach(v => v && v.classList.add('hidden'));
    const d = document.getElementById('view-detail');
    if (d) d.classList.remove('hidden');
    const panel = document.getElementById('dropzone-panel');
    if (panel) panel.classList.remove('hidden');
  });

  const stateProbe = async (state) => {
    await page.evaluate((s) => {
      // Re-assert the detail view + panel visibility every probe — a poll()
      // tick (4s interval) can re-hide them between probes / viewport swaps.
      document.querySelectorAll('#view-landing, #view-list, #view-error').forEach(v => v && v.classList.add('hidden'));
      document.getElementById('view-detail')?.classList.remove('hidden');
      document.getElementById('dropzone-panel')?.classList.remove('hidden');
      const dz = document.getElementById('dropzone');
      if (dz) dz.dataset.state = s;
    }, state);
    await new Promise(r => setTimeout(r, 150));
    return page.evaluate(() => {
      const vis = (el) => { if (!el) return false; const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
      const actions = document.querySelector('.dz-mobile-actions');
      const hero = document.querySelector('.dz-hero-hint');
      const cam = document.getElementById('btn-camera');
      const choose = document.getElementById('btn-choose');
      return {
        actionsVisible: vis(actions),
        heroVisible: vis(hero),
        camVisible: vis(cam),
        chooseVisible: vis(choose),
      };
    });
  };

  // 2. has-photos (minimized): hero hidden, both action buttons visible.
  const hp = await stateProbe('has-photos');
  await page.screenshot({ path: `${SHOTS}/has-photos.png` });
  rec('minimized menu: hero hidden', hp.heroVisible === false, `heroVisible=${hp.heroVisible}`);
  rec('minimized menu: Take photo + Choose photos visible',
      hp.camVisible && hp.chooseVisible,
      `camVisible=${hp.camVisible} chooseVisible=${hp.chooseVisible} actionsVisible=${hp.actionsVisible}`);

  // 3. empty: full hero shows AND the buttons are still there.
  const em = await stateProbe('empty');
  await page.screenshot({ path: `${SHOTS}/empty.png` });
  rec('empty state: full hero visible', em.heroVisible === true, `heroVisible=${em.heroVisible}`);
  rec('empty state: action buttons present', em.camVisible && em.chooseVisible, `camVisible=${em.camVisible} chooseVisible=${em.chooseVisible}`);

  // 4. Desktop viewport: buttons must ALSO be visible in minimized state
  //    (formerly sm:hidden). Re-probe at 1280px.
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
  const hpDesk = await stateProbe('has-photos');
  await page.screenshot({ path: `${SHOTS}/has-photos-desktop.png` });
  rec('desktop minimized: buttons visible', hpDesk.camVisible && hpDesk.chooseVisible,
      `camVisible=${hpDesk.camVisible} chooseVisible=${hpDesk.chooseVisible}`);

  fs.writeFileSync('tmp/verify_add_menu_result.json', JSON.stringify(out, null, 2));
  console.log('\n' + (out.pass ? 'ALL CHECKS PASSED' : 'SOME CHECKS FAILED'));
  console.log('screenshots in ' + SHOTS);
} catch (e) {
  console.error('PROBE ERROR:', e.stack || e.message);
  out.pass = false; out.fatal = String(e.message || e);
  fs.writeFileSync('tmp/verify_add_menu_result.json', JSON.stringify(out, null, 2));
} finally {
  await browser.close();
}
process.exit(out.pass ? 0 : 1);
