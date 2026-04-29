"""Render an evaluation_*.json into a browsable HTML side-by-side audit page.

The single-label exact-match accuracy metric in evaluate_rag.py is too
narrow for fine-grained classification — many photos have multiple
valid fine labels, so the model picking a different valid label gets
counted as wrong even when it's actually right.

This script generates a one-page HTML where each row shows:
  [photo thumbnail] [parent: pred / gt] [fine: pred / gt] [✓ ✗ ?]

A human reviews the page and can click ✓ / ? / ✗ on each row to record
their judgment ("model's pick is a valid label for this photo, even if
not exact-match GT"). The script doesn't persist clicks — it just
provides a fast visual scan tool.

Open the output HTML in any browser. Photos load from local disk via
file:// URLs (the eval JSON has relative paths from the dataset root).

Usage:
  python scripts/render_eval_audit.py evaluation_flash_finegrained.json
  python scripts/render_eval_audit.py <eval.json> --out audit.html --root /path/to/dataset
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path.home() / "Desktop" / "aecis-violations"


def _photo_uri(rel_path: str, root: Path) -> str:
    """Build a file:// URI to the local photo. Survives spaces and
    Windows backslashes by URL-quoting the path component."""
    p = (root / rel_path).resolve()
    # Forward-slash, then URI-quote everything except the slashes
    posix = str(p).replace("\\", "/")
    return "file:///" + quote(posix, safe="/:")


def _render_row(idx: int, r: dict, root: Path) -> str:
    img = _photo_uri(r.get("image", ""), root)
    parent_pred = html.escape(r.get("pred_hse") or "—")
    parent_gt = html.escape(r.get("gt_hse") or "—")
    parent_match = r.get("hse_match")
    fine_pred = r.get("pred_fine") or ""
    fine_gt = r.get("gt_fine") or ""
    fine_emitted = bool(r.get("fine_emitted"))
    fine_match = r.get("fine_match")
    rationale = html.escape(r.get("rationale") or "")
    fine_conf = r.get("fine_conf") or 0
    hse_conf = r.get("hse_conf") or 0

    parent_cls = "match" if parent_match else "miss"
    fine_cls = "match" if fine_match else ("miss" if fine_emitted else "skip")

    # Highlight semantically: green if parent matches, yellow if fine
    # was emitted but doesn't exact-match GT (could still be valid label),
    # gray if Stage 2 abstained, red if parent itself is wrong.
    return f"""<tr class="row-{parent_cls}">
  <td class="idx">{idx}</td>
  <td class="thumb"><img src="{img}" loading="lazy" /></td>
  <td class="labels">
    <div class="row-block">
      <div class="kind">parent ({hse_conf:.2f})</div>
      <div class="pair">
        <span class="pred">{parent_pred}</span>
        <span class="vs">vs</span>
        <span class="gt">{parent_gt}</span>
      </div>
    </div>
    <div class="row-block">
      <div class="kind">fine ({fine_conf:.2f}) {('' if fine_emitted else '— abstained')}</div>
      <div class="pair">
        <span class="pred">{html.escape(fine_pred or '(none)')}</span>
        <span class="vs">vs</span>
        <span class="gt">{html.escape(fine_gt or '(none)')}</span>
      </div>
    </div>
    {f'<div class="rationale">{rationale}</div>' if rationale else ''}
  </td>
  <td class="judge">
    <button class="j-correct" onclick="judge(this, '{idx}', 'correct')">✓ valid</button>
    <button class="j-different" onclick="judge(this, '{idx}', 'different')">⚠ different but valid</button>
    <button class="j-wrong" onclick="judge(this, '{idx}', 'wrong')">✗ wrong</button>
  </td>
</tr>"""


def _render(eval_data: dict, root: Path) -> str:
    rows = eval_data.get("results", [])
    n = len(rows)
    summary = f"""
    <div class="meta">
      <div><b>Sample size:</b> {n} photos</div>
      <div><b>Model:</b> {html.escape(eval_data.get('model', '?'))}</div>
      <div><b>Parent accuracy:</b> {eval_data.get('hse_accuracy', 0)*100:.1f}%</div>
      <div><b>Fine emit rate:</b> {eval_data.get('fine_emit_rate', 0)*100:.1f}%</div>
      <div><b>Fine accuracy when emitted (exact-match):</b> {eval_data.get('fine_accuracy_when_emitted', 0)*100:.1f}%</div>
    </div>
    """

    body = "\n".join(_render_row(i + 1, r, root) for i, r in enumerate(rows))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Eval audit — {n} photos</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font: 14px/1.4 -apple-system, system-ui, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
  header {{ position: sticky; top: 0; background: #fff; padding: 12px 24px; border-bottom: 1px solid #e2e8f0; z-index: 10; }}
  .meta {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #475569; }}
  .meta b {{ color: #0f172a; margin-right: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 12px 16px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  .idx {{ width: 40px; text-align: center; color: #94a3b8; font-variant-numeric: tabular-nums; }}
  .thumb {{ width: 220px; }}
  .thumb img {{ width: 100%; height: auto; max-height: 200px; object-fit: cover; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .labels {{ width: 60%; }}
  .row-block {{ margin-bottom: 8px; }}
  .kind {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 2px; }}
  .pair {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; }}
  .pred {{ font-weight: 600; }}
  .gt {{ color: #64748b; }}
  .vs {{ color: #cbd5e1; margin: 0 6px; font-size: 10px; }}
  .rationale {{ font-style: italic; color: #64748b; font-size: 12px; margin-top: 4px; }}
  .row-match {{ background: #f0fdf4; }}
  .row-miss {{ background: #fef2f2; }}
  .judge {{ width: 200px; white-space: nowrap; }}
  .judge button {{ display: block; width: 100%; margin-bottom: 4px; padding: 5px 8px; border: 1px solid #cbd5e1; background: #fff; border-radius: 4px; font-size: 12px; cursor: pointer; text-align: left; }}
  .judge button:hover {{ background: #f1f5f9; }}
  .judge button.active {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
  .summary-bar {{ position: fixed; bottom: 16px; right: 16px; background: #fff; padding: 12px 16px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); font-variant-numeric: tabular-nums; font-size: 12px; }}
  .summary-bar > div {{ margin: 2px 0; }}
  .copy-btn {{ margin-top: 8px; background: #0f172a; color: #fff; border: 0; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer; }}
</style>
</head>
<body>
<header>
  <h1 style="margin: 0 0 6px 0; font-size: 18px;">Evaluation audit — {n} photos</h1>
  {summary}
</header>
<table>
  <tbody>
    {body}
  </tbody>
</table>
<div class="summary-bar" id="summary-bar">
  <div><b>Judged:</b> <span id="n-judged">0</span> / {n}</div>
  <div>✓ valid: <span id="n-correct">0</span></div>
  <div>⚠ different but valid: <span id="n-different">0</span></div>
  <div>✗ wrong: <span id="n-wrong">0</span></div>
  <button class="copy-btn" onclick="copyResults()">Copy judgments JSON</button>
</div>
<script>
const judgments = {{}};
function judge(btn, idx, verdict) {{
  // Toggle off if same verdict clicked again
  if (judgments[idx] === verdict) {{
    delete judgments[idx];
    btn.classList.remove("active");
  }} else {{
    judgments[idx] = verdict;
    btn.parentElement.querySelectorAll("button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  }}
  updateSummary();
}}
function updateSummary() {{
  const total = Object.keys(judgments).length;
  const c = Object.values(judgments).filter(v => v === "correct").length;
  const d = Object.values(judgments).filter(v => v === "different").length;
  const w = Object.values(judgments).filter(v => v === "wrong").length;
  document.getElementById("n-judged").textContent = total;
  document.getElementById("n-correct").textContent = c;
  document.getElementById("n-different").textContent = d;
  document.getElementById("n-wrong").textContent = w;
}}
function copyResults() {{
  const blob = JSON.stringify(judgments, null, 2);
  navigator.clipboard.writeText(blob);
  alert("Judgments copied to clipboard.");
}}
</script>
</body>
</html>
"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", type=str,
                    help="Path to evaluation_*.json (output of evaluate_rag.py)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output HTML path. Default: <input>.audit.html")
    ap.add_argument("--root", type=str, default=str(DEFAULT_DATASET_ROOT),
                    help="Dataset root for resolving relative photo paths")
    args = ap.parse_args()

    in_path = Path(args.eval_json).resolve()
    out_path = Path(args.out).resolve() if args.out else in_path.with_suffix(".audit.html")
    root = Path(args.root).resolve()

    eval_data = json.loads(in_path.read_text(encoding="utf-8"))
    html_doc = _render(eval_data, root)
    out_path.write_text(html_doc, encoding="utf-8")

    n = len(eval_data.get("results", []))
    print(f"Wrote audit page: {out_path}")
    print(f"  {n} rows, photos resolve from {root}")
    print(f"Open in browser:")
    print(f"  start \"\" \"{out_path}\"   # Windows")
    print(f"  open \"{out_path}\"          # macOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
