"""Generate the PWA icon set from scratch using Pillow.

The webapp's existing favicon is an inline SVG (emerald check-circle).
Browsers want raster icons for PWA install (manifest.icons[]) and for
Apple touch icons. This script draws those PNGs deterministically so
the icons stay in sync with the webapp's color scheme without needing
an external design step.

Outputs into webapp/static/icons/ :
  icon-192.png         # PWA manifest required minimum
  icon-512.png         # PWA manifest required for splash + install
  icon-180.png         # Apple touch icon (iOS home-screen)
  icon-maskable-512.png  # Android adaptive icon (full-bleed safe area)

Idempotent. Re-run any time to regenerate.

Usage:
  python scripts/generate_pwa_icons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "webapp" / "static" / "icons"


# Brand colors — lifted from existing _base.html theme-color + emerald-500.
EMERALD_500 = "#10b981"
WHITE = "#ffffff"


def _make_icon(size: int, *, maskable: bool = False) -> "Image.Image":
    """Draw the violation-AI icon at `size` pixels.

    maskable: when True, use a SAFE ZONE — Android adaptive icons crop
    the corners (e.g. into a circle), so the visible content needs to
    fit within the inner 80% of the canvas. We draw the check-circle
    smaller so cropping doesn't clip it.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Maskable: keep the brand circle inside the inner 80% (safe zone)
    # and fill the outer 20% with the same brand color so adaptive
    # crops still see continuous color.
    if maskable:
        d.rectangle((0, 0, size, size), fill=EMERALD_500)
        circle_pad = int(size * 0.10)
        # The "inner" emblem is just the white check on the same emerald
        # background. So no separate circle needed for maskable — the
        # whole canvas IS the circle.
    else:
        # Standard icon: emerald-500 filled circle on transparent bg
        d.ellipse((0, 0, size, size), fill=EMERALD_500)

    # White checkmark — 3-segment polyline approximating the SVG's
    # M5 13l4 4L19 7 path mapped into 0..size space.
    # SVG viewBox was 0..24, path is (5,13) -> (9,17) -> (19,7).
    inset = 0.20 if not maskable else 0.30   # smaller for maskable safe zone
    def _p(sx, sy):
        # SVG path coords are 0..24. Center the check in the available area.
        # Fraction of size: x=5/24 -> 0.208, x=9/24 -> 0.375, x=19/24 -> 0.792
        # y=13/24 -> 0.542, y=17/24 -> 0.708, y=7/24 -> 0.292
        # Then squish into [inset, 1-inset]
        usable = 1 - 2 * inset
        return (
            inset * size + sx * usable * size,
            inset * size + sy * usable * size,
        )
    pts = [_p(5/24, 13/24), _p(9/24, 17/24), _p(19/24, 7/24)]
    line_w = max(2, size // 10)
    # Use joint='curve' so the corner of the V doesn't show a notch
    d.line(pts, fill=WHITE, width=line_w, joint="curve")

    # Round caps at endpoints (Pillow's line doesn't draw round caps;
    # add small filled circles)
    cap_r = line_w // 2
    for px, py in (pts[0], pts[2]):
        d.ellipse(
            (px - cap_r, py - cap_r, px + cap_r, py + cap_r),
            fill=WHITE,
        )

    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        (192, False, "icon-192.png"),
        (512, False, "icon-512.png"),
        (180, False, "icon-180.png"),          # apple-touch-icon
        (512, True,  "icon-maskable-512.png"), # Android adaptive
    ]
    for size, maskable, name in targets:
        img = _make_icon(size, maskable=maskable)
        out = OUT_DIR / name
        img.save(out, "PNG", optimize=True)
        print(f"  wrote {out.relative_to(REPO_ROOT)}  ({size}px"
              f"{', maskable' if maskable else ''})")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
