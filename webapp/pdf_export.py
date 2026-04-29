"""Build a printable PDF report from a batch of classified photos.

Layout:
  - Cover page: title, date range, batch label, totals, top HSE types
  - One photo per page: thumbnail (~4" wide), violation details, AI vs
    final label (when corrected), inspector note (when present)
  - Page footer: generated-at + page N of M

The "one page per photo" format is what AECIS HSE clients expect — they
hand it to safety officers, project managers, or print and sign. Packing
multiple photos per page makes it harder to read and reference in
on-site discussions.

Unicode (Vietnamese diacritics): tries to register DejaVuSans from the
local system. Falls back to Helvetica which can't render VN text — at
which point VN strings get sanitized to ASCII so we don't emit boxes.
This is a v1 trade-off; embedding a bundled font is straightforward but
adds ~600KB to the repo, deferred until needed.
"""
from __future__ import annotations

import io
import logging
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ---- font registration ----

_FONT_NORMAL = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONT_REGISTERED = False


def _register_unicode_font() -> tuple[str, str]:
    """Try to register a unicode TTF so VN diacritics render. Returns
    (normal_name, bold_name). Caches the result via a module-level flag."""
    global _FONT_NORMAL, _FONT_BOLD, _FONT_REGISTERED
    if _FONT_REGISTERED:
        return _FONT_NORMAL, _FONT_BOLD
    _FONT_REGISTERED = True

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Search common locations for a unicode TTF. Order matters — prefer
    # DejaVuSans (full coverage, sane metrics) over the others.
    candidates_normal = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    normal_path = next((p for p in candidates_normal if os.path.exists(p)), None)
    bold_path = next((p for p in candidates_bold if os.path.exists(p)), None)
    if not normal_path:
        log.info("PDF: no unicode font found, using Helvetica (VN diacritics will be ASCIIfied)")
        return _FONT_NORMAL, _FONT_BOLD
    try:
        pdfmetrics.registerFont(TTFont("ReportNormal", normal_path))
        if bold_path:
            pdfmetrics.registerFont(TTFont("ReportBold", bold_path))
            _FONT_BOLD = "ReportBold"
        else:
            _FONT_BOLD = "ReportNormal"
        _FONT_NORMAL = "ReportNormal"
        log.info("PDF: registered unicode font from %s", normal_path)
    except Exception as e:  # noqa: BLE001
        log.warning("PDF: font registration failed (%s) — falling back to Helvetica", e)
    return _FONT_NORMAL, _FONT_BOLD


def _ascii_safe(s: str | None) -> str:
    """Strip diacritics and drop non-ASCII chars. Used when no unicode
    font registered, so we don't emit literal box glyphs for VN text."""
    if not s:
        return ""
    nf = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nf if not unicodedata.combining(c)).encode("ascii", "ignore").decode("ascii")


def _safe(s: str | None, has_unicode_font: bool) -> str:
    return (s or "") if has_unicode_font else _ascii_safe(s)


# ---- image fetch + downscale ----

def _fetch_thumb(r2_client, bucket: str, key: str, max_w: int = 1200) -> bytes | None:
    """Pull the original from R2 and downscale so embedding it in the PDF
    doesn't balloon the file. 1200px wide @ 4 inch print = 300 DPI which
    is overkill for screen viewing and fine for paper. Returns JPEG bytes
    suitable for reportlab's Image flowable, or None if R2 errors."""
    try:
        obj = r2_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
    except Exception as e:  # noqa: BLE001
        log.warning("PDF: R2 fetch failed for %s: %s", key, e)
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")  # drop alpha so JPEG encoder is happy
            w, h = img.size
            if w > max_w:
                ratio = max_w / w
                img = img.resize((max_w, int(h * ratio)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.warning("PDF: image normalize failed for %s: %s — passing raw", key, e)
        return raw


# ---- main builder ----

def build_violation_pdf(
    rows: list[dict[str, Any]],
    *,
    batch_label: str | None = None,
    batch_id: str | None = None,
    r2_client=None,
    r2_bucket: str = "",
    project_label: str = "",
) -> bytes:
    """Return a complete PDF as bytes. `rows` are export rows already
    enriched with label_en / label_vn / fine_hse_type_label fields by
    _enrich_with_labels in app.py."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    normal_font, bold_font = _register_unicode_font()
    has_unicode = (normal_font != "Helvetica")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontName=normal_font, fontSize=10, leading=13, textColor=colors.HexColor("#0f172a"),
    )
    body_dim = ParagraphStyle(
        "BodyDim", parent=body,
        textColor=colors.HexColor("#64748b"),
    )
    h1 = ParagraphStyle(
        "H1", parent=body,
        fontName=bold_font, fontSize=22, leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2", parent=body,
        fontName=bold_font, fontSize=14, leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    label_em = ParagraphStyle(
        "LabelEm", parent=body,
        fontName=bold_font, fontSize=11, leading=14,
        textColor=colors.HexColor("#047857"),  # emerald-700
    )
    pill_text = ParagraphStyle(
        "Pill", parent=body,
        fontSize=8, leading=10, textColor=colors.HexColor("#475569"),
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Violation Report",
        author="Violation AI",
    )

    story: list = []

    # --- cover page ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(rows)
    reviewed = sum(1 for r in rows if r.get("reviewed"))
    confirmed = sum(1 for r in rows if r.get("review_action") == "confirm")
    corrected = sum(1 for r in rows if r.get("review_action") == "correct")

    story.append(Paragraph("Violation Report", h1))
    story.append(Paragraph(_safe(batch_label or "(unlabeled batch)", has_unicode), h2))
    story.append(Spacer(1, 0.15 * inch))

    cover_meta = [
        ["Generated", now],
        ["Project", project_label or "AECIS HSE"],
        ["Batch ID", (batch_id or "—")[:36]],
        ["Photos", str(total)],
        ["Reviewed", f"{reviewed} of {total}  ({confirmed} confirmed, {corrected} corrected)"],
    ]
    cover_table = Table(cover_meta, colWidths=[1.4 * inch, 5.5 * inch])
    cover_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), bold_font),
        ("FONTNAME", (1, 0), (1, -1), normal_font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e8f0")),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.3 * inch))

    # Top-violation distribution. Use the FINE label when present (the
    # specific AECIS sub-type), fall back to parent. This makes the cover
    # page far more useful than "30 photos of Housekeeping_general" —
    # safety officers want to see "Smoking_or_cigarette_butts: 8,
    # Materials_left_in_disarray: 5, ..." with real specificity.
    from collections import Counter
    hse_counter: Counter[str] = Counter()
    for r in rows:
        fine_label = r.get("final_fine_hse_type_label_en") or r.get("final_fine_hse_type_slug")
        parent_label = r.get("final_hse_type_label_en") or r.get("ai_hse_type_label_en") or r.get("final_hse_type_slug")
        label = fine_label or parent_label
        if label:
            hse_counter[label] += 1
    if hse_counter:
        story.append(Paragraph("Top violations", h2))
        top_data = [["Violation", "Count", "Share"]]
        for label, n in hse_counter.most_common(10):
            top_data.append([
                _safe(label, has_unicode),
                str(n),
                f"{n / total * 100:.0f}%",
            ])
        top_table = Table(top_data, colWidths=[4.5 * inch, 1.0 * inch, 1.0 * inch])
        top_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 1), (0, -1), normal_font),
        ]))
        story.append(top_table)

    story.append(PageBreak())

    # --- one photo per page ---
    for i, r in enumerate(rows, 1):
        img_bytes = None
        if r2_client and r.get("storage_key"):
            img_bytes = _fetch_thumb(r2_client, r2_bucket, r["storage_key"])
        if img_bytes:
            try:
                img_buf = io.BytesIO(img_bytes)
                # Image flowable max-w 6.0", max-h 4.0" — preserves aspect
                pil = None
                try:
                    from PIL import Image as PILImage
                    pil = PILImage.open(io.BytesIO(img_bytes))
                    iw, ih = pil.size
                except Exception:  # noqa: BLE001
                    iw, ih = (1200, 800)
                target_w = 6.0 * inch
                target_h = 4.0 * inch
                ratio = min(target_w / iw, target_h / ih) if iw and ih else 1
                w_pts = (iw or 1200) * ratio
                h_pts = (ih or 800) * ratio
                story.append(RLImage(img_buf, width=w_pts, height=h_pts))
            except Exception as e:  # noqa: BLE001
                log.warning("PDF: image embed failed for row %d: %s", i, e)
                story.append(Paragraph("(image unavailable)", body_dim))
        else:
            story.append(Paragraph("(image unavailable)", body_dim))

        story.append(Spacer(1, 0.15 * inch))

        # Photo header line: filename + date
        fname = r.get("original_filename") or "(unnamed)"
        uploaded = r.get("uploaded_at") or ""
        if uploaded and "T" in uploaded:
            uploaded = uploaded.split("T")[0]
        story.append(Paragraph(
            f"<b>{i} / {total}</b> &nbsp;·&nbsp; {_safe(fname, has_unicode)} &nbsp;·&nbsp; {uploaded}",
            body_dim,
        ))
        story.append(Spacer(1, 0.05 * inch))

        # Final label = the inspector's truth (corrections override AI)
        final_hse = r.get("final_hse_type_label_en") or r.get("final_hse_type_slug") or "—"
        final_loc = r.get("final_location_label_en") or r.get("final_location_slug") or "—"
        final_fine = r.get("final_fine_hse_type_label_en") or r.get("final_fine_hse_type_slug") or ""
        review_action = r.get("review_action") or ""
        review_note = r.get("review_note") or ""

        # Status pill text
        if r.get("reviewed"):
            status = "Confirmed by inspector" if review_action == "confirm" else "Corrected by inspector"
        else:
            status = "AI prediction (not yet reviewed)"

        # Detail table (final classification + AI prediction + status).
        # When a fine sub-type is present, show it as the headline
        # "Violation" — that's the AECIS-canonical specific item the
        # safety officer wants to read first. Parent class becomes a
        # smaller "Category" subtitle. When no fine is set, parent is
        # the primary line (matches the previous behaviour).
        ai_hse = r.get("ai_hse_type_label_en") or r.get("ai_hse_type_slug") or "—"
        ai_conf = r.get("ai_hse_confidence")
        ai_conf_str = f"{ai_conf:.0%}" if isinstance(ai_conf, (int, float)) else "—"

        detail: list[list] = []
        if final_fine:
            detail.append([
                Paragraph("<b>Violation</b>", label_em),
                Paragraph(_safe(final_fine, has_unicode), body),
            ])
            detail.append([
                Paragraph("<b>Category</b>", label_em),
                Paragraph(_safe(final_hse, has_unicode), body_dim),
            ])
        else:
            detail.append([
                Paragraph("<b>Violation type</b>", label_em),
                Paragraph(_safe(final_hse, has_unicode), body),
            ])
        detail.extend([
            [Paragraph("<b>Location</b>", label_em), Paragraph(_safe(final_loc, has_unicode), body)],
            [Paragraph("<b>Status</b>", label_em), Paragraph(status, body)],
            [Paragraph("<b>AI prediction</b>", label_em),
             Paragraph(f"{_safe(ai_hse, has_unicode)} &nbsp;<font color='#94a3b8'>({ai_conf_str})</font>", body)],
        ])
        if review_note:
            detail.append([
                Paragraph("<b>Note</b>", label_em),
                Paragraph(_safe(review_note, has_unicode), body),
            ])
        detail_table = Table(detail, colWidths=[1.4 * inch, 5.5 * inch])
        detail_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#e2e8f0")),
        ]))
        story.append(detail_table)

        # Page break between photos (last one inside loop is fine —
        # SimpleDocTemplate ignores trailing PageBreaks).
        if i < total:
            story.append(PageBreak())

    # --- footer with page numbers ---
    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(normal_font, 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        page_num = canvas.getPageNumber()
        # Footer left: generation timestamp; right: page N
        canvas.drawString(0.6 * inch, 0.4 * inch, f"Generated {now}")
        canvas.drawRightString(LETTER[0] - 0.6 * inch, 0.4 * inch, f"Page {page_num}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
