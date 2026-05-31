/* UI bug-audit harness.
 *
 * Runs the inspector workflow end-to-end through Puppeteer and:
 *   1. Captures every console.error / pageerror / failed network
 *   2. At each scripted step, clicks the actual intended target
 *      AND clicks a random set of non-destructive sibling elements
 *      (visible buttons, links, chips) to surface latent bugs where
 *      the app misbehaves under "fat-finger" / out-of-order taps.
 *   3. Screenshots each step for visual diff review.
 *
 * Not a regression test — it's an exploration tool. Failures here
 * are findings to investigate, not assertions that gate CI.
 *
 * Usage (from project root, dev server already running on 8765):
 *   node scripts/ui_audit.mjs                          # default run
 *   node scripts/ui_audit.mjs --noisy 5                # 5 random clicks per step
 *   node scripts/ui_audit.mjs --base http://127.0.0.1:8765
 */
import { getBrowser, getPage, disconnectBrowser, outputJSON } from "file:///C:/Users/lang/.claude/skills/chrome-devtools/scripts/lib/browser.js";
import fs from "fs";
import path from "path";

const argv = process.argv.slice(2);
function arg(key, fallback) {
  const i = argv.indexOf(key);
  return i >= 0 ? argv[i + 1] : fallback;
}

const BASE_URL    = arg("--base", "http://127.0.0.1:8765");
const NOISY_CLICKS = parseInt(arg("--noisy", "3"), 10);
const OUT_DIR     = path.resolve(".claude/chrome-devtools/screenshots/audit");
fs.mkdirSync(OUT_DIR, { recursive: true });

// Selectors of things we MUST NOT click on while exploring — clicking
// them mid-audit would destroy state we still want to test against.
const DANGER_SELECTORS = [
  "form[action='/auth/logout'] button",
  "[data-destructive='1']",
  ".batch-delete-btn",
  ".bulk-delete-btn",
  "#confirm-delete",
];

function ts() { return new Date().toISOString().replace(/[:.]/g, "-"); }

async function snapshot(page, label, findings) {
  const fn = path.join(OUT_DIR, `audit-${ts()}-${label}.png`);
  await page.screenshot({ path: fn });
  findings.steps.push({ label, screenshot: fn });
}

async function clickRandomNonDestructive(page, count, findings, stepLabel) {
  if (count <= 0) return;
  const candidates = await page.evaluate((dangerSels) => {
    const dangerSet = new Set();
    dangerSels.forEach(sel => {
      try { document.querySelectorAll(sel).forEach(n => dangerSet.add(n)); }
      catch (_) {}
    });
    const isVisible = (el) => {
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return false;
      const cs = getComputedStyle(el);
      if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0") return false;
      return true;
    };
    const out = [];
    document.querySelectorAll("button, [role='button'], a, [role='menuitem']").forEach((el, idx) => {
      if (dangerSet.has(el)) return;
      if (!isVisible(el)) return;
      // Skip submit-type buttons inside forms with action= (avoid POSTing things)
      if (el.tagName === "BUTTON" && el.type === "submit") return;
      out.push({ idx, tag: el.tagName, id: el.id || "", text: (el.textContent || "").trim().slice(0, 40) });
    });
    return out;
  }, DANGER_SELECTORS);

  for (let i = 0; i < count && candidates.length > 0; i++) {
    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    try {
      // Click by re-resolving the index (DOM may have shifted)
      await page.evaluate((idx) => {
        const all = document.querySelectorAll("button, [role='button'], a, [role='menuitem']");
        if (all[idx]) all[idx].click();
      }, pick.idx);
      // Brief settle
      await new Promise(r => setTimeout(r, 250));
      findings.adversarial.push({ step: stepLabel, clicked: pick });
    } catch (e) {
      // best-effort — if a click navigates away, capture and bail this batch
      findings.adversarial.push({ step: stepLabel, clicked: pick, error: String(e).slice(0, 120) });
      break;
    }
  }
}

async function main() {
  const browser = await getBrowser();
  const page = await getPage(browser);

  const findings = {
    base_url: BASE_URL,
    started_at: new Date().toISOString(),
    console_errors: [],
    page_errors: [],
    failed_requests: [],
    steps: [],
    adversarial: [],
  };

  page.on("console", m => {
    if (m.type() === "error" || m.type() === "warn") {
      findings.console_errors.push({ type: m.type(), text: m.text() });
    }
  });
  page.on("pageerror", e => findings.page_errors.push(e.message));
  page.on("requestfailed", req => {
    const reason = req.failure()?.errorText || "unknown";
    if (reason !== "net::ERR_ABORTED") {
      findings.failed_requests.push({ url: req.url(), reason });
    }
  });
  page.on("response", async (resp) => {
    if (resp.status() >= 500 && resp.url().includes(BASE_URL)) {
      findings.failed_requests.push({ url: resp.url(), status: resp.status() });
    }
  });

  // ===== Step 1: main batches landing =====
  await page.goto(BASE_URL, { waitUntil: "networkidle2" });
  await new Promise(r => setTimeout(r, 1500));
  await snapshot(page, "01-landing", findings);
  await clickRandomNonDestructive(page, NOISY_CLICKS, findings, "01-landing");

  // ===== Step 2: open an existing batch (new flow — v117.9 removed
  // the "+ Start new inspection" button; the dropzone now drives
  // batch creation on upload). If no batch exists yet, just stay
  // on the list view and skip the detail-view assertion.
  const opened = await page.evaluate(() => {
    const firstBatchRow = document.querySelector("#batches-grid > *");
    if (firstBatchRow) {
      firstBatchRow.click();
      return true;
    }
    return false;
  });
  await new Promise(r => setTimeout(r, 2000));
  await snapshot(page, "02-batch-detail-or-list", findings);

  // v118.10 — The [+ Add] toggle button (#btn-add-photos) was removed.
  // Replaced by a permanent Take photo / Choose photos menu inside the
  // detail-view dropzone. Assertion now: that button must NOT exist, and
  // when a batch is open the permanent action buttons must be present.
  findings.assertions = findings.assertions || [];
  const addBtnGone = await page.evaluate(() =>
    document.getElementById("btn-add-photos") === null);
  findings.assertions.push({
    name: "legacy-add-button-removed",
    pass: addBtnGone,
    detail: { btn_add_photos_in_dom: !addBtnGone },
  });
  if (opened) {
    const detailState = await page.evaluate(() => {
      const vis = (el) => {
        if (!el) return false;
        const r = el.getBoundingClientRect();
        const cs = getComputedStyle(el);
        return cs.display !== "none" && cs.visibility !== "hidden" && r.width > 0 && r.height > 0;
      };
      return {
        cam_visible: vis(document.getElementById("btn-camera")),
        choose_visible: vis(document.getElementById("btn-choose")),
        photo_cards_count: document.querySelectorAll(".card").length,
      };
    });
    findings.assertions.push({
      name: "permanent-add-menu-present",
      pass: detailState.cam_visible && detailState.choose_visible,
      detail: detailState,
    });
  } else {
    findings.assertions.push({
      name: "permanent-add-menu-present",
      skipped: true,
      reason: "no batches in the test database — assertion skipped",
    });
  }

  await clickRandomNonDestructive(page, NOISY_CLICKS, findings, "02-empty-batch");
  await snapshot(page, "03-empty-batch-after-noisy-clicks", findings);

  // Assertion: the upload affordance must STILL be visible after fat-finger
  // clicks. v117.13 two-instance architecture: the inspection-list view shows
  // the always-mounted #dropzone-list (never hidden), while an opened batch
  // shows the detail-view #dropzone-panel (auto-expanded by poll() for empty
  // batches). Check whichever element belongs to the current view — on an
  // empty test DB no batch opens, so we stay on the list view.
  // A noisy click may have navigated the page; let it settle before probing.
  await new Promise(r => setTimeout(r, 500));
  let dzStillVisible = false;
  try {
    dzStillVisible = await page.evaluate((onDetail) => {
      const id = onDetail ? "dropzone-panel" : "dropzone-list";
      const dz = document.getElementById(id);
      if (!dz) return false;
      const styleHidden = dz.classList.contains("hidden")
        || getComputedStyle(dz).display === "none";
      return !styleHidden && dz.getBoundingClientRect().width > 0;
    }, opened);
  } catch (e) {
    // Context destroyed by an in-flight navigation — re-probe once after settle.
    await new Promise(r => setTimeout(r, 800));
    dzStillVisible = await page.evaluate((onDetail) => {
      const id = onDetail ? "dropzone-panel" : "dropzone-list";
      const dz = document.getElementById(id);
      if (!dz) return false;
      const styleHidden = dz.classList.contains("hidden")
        || getComputedStyle(dz).display === "none";
      return !styleHidden && dz.getBoundingClientRect().width > 0;
    }, opened).catch(() => false);
  }
  findings.assertions.push({
    name: "dropzone-survives-noisy-clicks",
    pass: !!dzStillVisible,
    detail: { dz_visible: dzStillVisible, view: opened ? "detail" : "list" },
  });

  // ===== Step 3: navigate back to all inspections =====
  await page.evaluate(() => {
    const anchors = Array.from(document.querySelectorAll("a, button"));
    const back = anchors.find(a => /all inspections/i.test(a.textContent || ""));
    if (back) back.click();
  });
  await new Promise(r => setTimeout(r, 1500));
  await snapshot(page, "04-back-to-list", findings);

  // ===== Step 4: header language picker (read-only) =====
  await page.evaluate(() => document.getElementById("lang-btn")?.click());
  await new Promise(r => setTimeout(r, 400));
  await snapshot(page, "05-lang-menu-open", findings);
  // Close it
  await page.evaluate(() => document.body.click());
  await new Promise(r => setTimeout(r, 300));

  findings.finished_at = new Date().toISOString();
  // Write the report
  const reportPath = path.join(OUT_DIR, `report-${ts()}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(findings, null, 2), "utf-8");
  outputJSON({
    report: reportPath,
    steps: findings.steps.length,
    adversarial_clicks: findings.adversarial.length,
    console_errors: findings.console_errors.length,
    page_errors: findings.page_errors.length,
    failed_requests: findings.failed_requests.length,
    assertions: findings.assertions || [],
  });
  await disconnectBrowser();
}

main();
