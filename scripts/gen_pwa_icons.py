"""Generate PWA + Apple-touch icons for HSE Detector.

The original assets in webapp/static/icons/ were the emerald
check-mark from the pre-rebrand "Violation AI" days. After we
renamed to HSE Detector and switched the visual language to
drafting-paper / ink, those PNGs were the last piece still
shouting the old brand. This script regenerates all four:

    icon-180.png            (apple-touch-icon, iOS home-screen)
    icon-192.png            (Android Chrome legacy)
    icon-512.png            (Android Chrome high-density)
    icon-maskable-512.png   (Android adaptive-icon "any" + "maskable")

Design:
    - Paper background (#f5f3eb) with a faint heavy-grid line so
      the icon reads as drafting paper at home-screen size.
    - Bold black "HSE" wordmark, slightly tracked-out.
    - Drafting-marker orange accent underline beneath HSE.
    - Subtle ink hairline frame for definition against light
      home-screen wallpapers.
    - Maskable variant keeps the wordmark inside the 80% safe
      zone so Android's circle / rounded-square mask doesn't
      clip the H or E.

Usage:
    python scripts/gen_pwa_icons.py
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = REPO_ROOT / "webapp" / "static" / "icons"

# Drafting-paper palette — same hex values as the landing page CSS.
PAPER       = (245, 243, 235, 255)   # #f5f3eb
PAPER_WARM  = (239, 236, 225, 255)
INK         = (10,  10,  10,  255)   # #0a0a0a
INK_HAIR    = (10,  10,  10,  102)   # rgba(10,10,10,0.40) hairline
RULE_FAINT  = (10,  10,  10,  18)    # rgba(10,10,10,0.07) fine grid
RULE_HEAVY  = (10,  10,  10,  33)    # rgba(10,10,10,0.13) heavy grid
ACCENT      = (180, 83,  9,   255)   # #b45309 drafting-marker orange

# Sizes — apple-touch (180) is its own file, the rest map to manifest entries.
TARGETS = {
    "icon-180.png":          (180, "any"),
    "icon-192.png":          (192, "any"),
    "icon-512.png":          (512, "any"),
    "icon-maskable-512.png": (512, "maskable"),
}

# Bold sans for the HSE wordmark. arialbd is universal on Windows; the
# scaled PNG output looks the same on the deployed Linux server because
# we ship pre-rendered PNGs.
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def _load_font(target_px: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, target_px)
    # Last-resort fallback — bitmap font (small, ugly, but won't crash).
    return ImageFont.load_default()


def _draw_grid(draw: ImageDraw.ImageDraw, size: int, *, fine_step: int, heavy_step: int) -> None:
    """Faint drafting-paper grid. Lines every fine_step pixels (color
    `RULE_FAINT`), with heavier lines every heavy_step pixels."""
    for x in range(0, size, fine_step):
        draw.line([(x, 0), (x, size)], fill=RULE_FAINT, width=1)
    for y in range(0, size, fine_step):
        draw.line([(0, y), (size, y)], fill=RULE_FAINT, width=1)
    for x in range(0, size, heavy_step):
        draw.line([(x, 0), (x, size)], fill=RULE_HEAVY, width=1)
    for y in range(0, size, heavy_step):
        draw.line([(0, y), (size, y)], fill=RULE_HEAVY, width=1)


def render(size: int, kind: str) -> Image.Image:
    """Render the icon at the requested pixel size.

    `kind` is `"any"` for square / rounded-square use, or `"maskable"`
    for Android adaptive icons (keeps the wordmark inside the 80%
    safe zone). Both variants share the same paper background +
    grid + wordmark.
    """
    img = Image.new("RGBA", (size, size), PAPER)
    draw = ImageDraw.Draw(img)

    # Drafting-paper grid scaled to icon size. Heavy grid every 5
    # fine cells (mirrors the landing-page 24/120 ratio).
    fine = max(8, size // 16)
    heavy = fine * 5
    _draw_grid(draw, size, fine_step=fine, heavy_step=heavy)

    # Subtle ink hairline frame. Maskable icons skip this — the mask
    # itself provides the visible boundary.
    if kind == "any":
        inset = max(2, size // 64)
        draw.rectangle(
            [(inset, inset), (size - 1 - inset, size - 1 - inset)],
            outline=INK_HAIR, width=max(1, size // 256),
        )

    # Wordmark. For maskable, shrink so it stays inside the 80% safe
    # zone (Android crops to circle / rounded-square / etc.).
    safe_pct = 0.62 if kind == "maskable" else 0.72
    target_height = int(size * safe_pct * 0.55)   # cap glyph height
    font = _load_font(target_height)
    text = "HSE"

    # Measure & center.
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # Pillow's textbbox includes ascender padding; recenter using the
    # tight bbox so the glyphs sit visually centered.
    cx = size / 2 - tw / 2 - bbox[0]
    cy = size / 2 - th / 2 - bbox[1] - size * 0.04   # nudge up to leave room for accent underline
    draw.text((cx, cy), text, font=font, fill=INK)

    # Drafting-marker orange underline beneath the wordmark — the
    # single chromatic accent against the monochrome paper/ink.
    underline_y = cy + th + size * 0.04
    underline_x0 = size / 2 - tw / 2.6
    underline_x1 = size / 2 + tw / 2.6
    underline_height = max(2, size // 48)
    draw.rectangle(
        [(underline_x0, underline_y),
         (underline_x1, underline_y + underline_height)],
        fill=ACCENT,
    )

    return img


def main() -> int:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for name, (size, kind) in TARGETS.items():
        img = render(size, kind)
        out = ICON_DIR / name
        img.save(out, format="PNG", optimize=True)
        print(f"  wrote {out.relative_to(REPO_ROOT)}  ({size}x{size}, {kind})")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
