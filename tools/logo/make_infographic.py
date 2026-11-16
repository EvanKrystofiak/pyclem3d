"""Build the "How pyCLEM3D works" infographic shown in the README, in the logo's style.

usage: python tools/logo/make_infographic.py ARCHIVO_800_WOFF [OUT_DIR]

Writes pyclem3d_how_it_works.svg and .png (2400 px wide) to OUT_DIR, by default docs/.
Needs fontTools, resvg-py and Pillow. Headings are Archivo ExtraBold outlines (the font
is in the family logo kit); body text is Segoe UI, so this runs on Windows as written.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marks as M
import resvg_py
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "docs")
n = M.n
INK, MUTED, RULE = "#1c212b", "#566072", "#d9dee6"
BODY_FONT = "Segoe UI, Arial, Helvetica, sans-serif"
ARCHIVO = TTFont(FONT)
BODY_METRICS = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 100)


def heavy(text, x, y, size, fill, anchor="start"):
    """Archivo ExtraBold as outlines (the family's name face)."""
    gs, cmap = ARCHIVO.getGlyphSet(), ARCHIVO.getBestCmap()
    k = size / ARCHIVO["head"].unitsPerEm
    width = sum(gs[cmap[ord(c)]].width for c in text) * k
    ox = x - (width if anchor == "end" else width / 2 if anchor == "middle" else 0)
    d = []
    for c in text:
        g = gs[cmap[ord(c)]]
        pen = SVGPathPen(gs)
        g.draw(TransformPen(pen, (k, 0, 0, -k, ox, y)))
        d.append(pen.getCommands())
        ox += g.width * k
    return '<path fill="{}" d="{}"/>'.format(fill, " ".join(d))


def wrap(text, size, width):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if BODY_METRICS.getlength(t) * size / 100 <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def body(text, x, y, size, width, fill=MUTED, lead=1.38):
    out = []
    for i, line in enumerate(wrap(text, size, width)):
        out.append(
            f'<text x="{n(x)}" y="{n(y + i * size * lead)}" font-family="{BODY_FONT}" font-size="{n(size)}" fill="{fill}">{line}</text>'
        )
    return "".join(out)


# ---- the scene every tile shares: one EM slice with three mitochondria ---------- #
MITOS = [(42, 41, 38, 17, -22), (68, 64, 44, 20, 24), (41, 87, 34, 15, -4)]  # cx cy w h tilt
SLICE = (18, 18, 84)  # x y size


def turned(inner, cx, cy, deg, dx=0, dy=0):
    return f'<g transform="translate({n(dx)} {n(dy)}) rotate({n(deg)} {n(cx)} {n(cy)})">{inner}</g>'


def slice_rect(fill, edge=M.GRID_EDGE, x=None, y=None, s=None):
    x0, y0, s0 = SLICE
    x, y, s = (x0 if x is None else x), (y0 if y is None else y), (s0 if s is None else s)
    return f'<rect x="{n(x)}" y="{n(y)}" width="{n(s)}" height="{n(s)}" rx="3" fill="{fill}" stroke="{edge}" stroke-width="2"/>'


def clip(uid):
    x, y, s = SLICE
    return f'<defs><clipPath id="{uid}"><rect x="{n(x + 1)}" y="{n(y + 1)}" width="{n(s - 2)}" height="{n(s - 2)}" rx="2"/></clipPath></defs>'


def nucleus():
    return f'<circle cx="100" cy="20" r="22" fill="none" stroke="{M.RIM}" stroke-width="2"/>'


def mitos(style, uid=None, cristae_col=None, dx=0, dy=0, grow=0):
    out = []
    for i, (cx, cy, w, h, deg) in enumerate(MITOS):
        inner = M.capsule(cx, cy, w + grow, h + grow, style)
        if cristae_col:
            cid = "%s-m%d" % (uid, i)
            xs = [cx - w * 0.24, cx, cx + w * 0.24]
            inner += (
                '<defs><clipPath id="{}">{}</clipPath></defs><g clip-path="url(#{})">{}</g>'.format(
                    cid,
                    M.capsule(cx, cy, w, h, ""),
                    cid,
                    M.cristae(cx, cy, h, xs, cristae_col, bw=3, frac=0.6),
                )
            )
        out.append(turned(inner, cx, cy, deg, dx, dy))
    return "".join(out)


def tile_volumes():
    """1. The two inputs: an EM stack and a confocal stack."""
    b = [M.tile()]
    for x0, y0, lm in ((15, 30, False), (64, 30, True)):
        s = 36
        for d in (8, 4):
            b.append(slice_rect(M.TILE, x=x0 + d, y=y0 - d, s=s))
        b.append(slice_rect(M.TILE if lm else M.GRID_FILL, x=x0, y=y0, s=s))
        for cx, cy, w, h, deg in ((x0 + 14, y0 + 12, 18, 8, -22), (x0 + 22, y0 + 25, 20, 9, 20)):
            if lm:
                style = f'fill="{M.ACC}" fill-opacity="0.32" stroke="{M.ACC}" stroke-width="2"'
            else:
                style = f'fill="none" stroke="{M.RIM}" stroke-width="2"'
            b.append(turned(M.capsule(cx, cy, w, h, style), cx, cy, deg))
    b.append(heavy("EM", 37, 94, 13, M.TEXT_DARK_BG, "middle"))
    b.append(heavy("LM", 86, 94, 13, M.ACC, "middle"))
    return "".join(b)


def tile_segment():
    """2. Organelles segmented in the EM: the mask."""
    return "".join(
        [
            clip("ig2"),
            M.tile(),
            slice_rect(M.GRID_FILL),
            '<g clip-path="url(#ig2)">',
            nucleus(),
            mitos(f'fill="{M.TEXT_DARK_BG}" stroke="none"'),
            "</g>",
        ]
    )


def tile_blur():
    """3. The mask blurred with the confocal PSF: synthetic fluorescence."""
    return "".join(
        [
            clip("ig3"),
            '<defs><filter id="ig3-blur" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur stdDeviation="4"/></filter></defs>',
            M.tile(),
            slice_rect(M.TILE),
            '<g clip-path="url(#ig3)"><g filter="url(#ig3-blur)">',
            mitos(f'fill="{M.ACC}" stroke="none"', grow=3),
            "</g>",
            mitos(f'fill="none" stroke="{M.RIM}" stroke-width="1.5" stroke-dasharray="3 3"'),
            "</g>",
        ]
    )


def tile_match():
    """4. The real fluorescence is moved until it fits the synthetic one."""
    dx, dy = 11, -9
    arrow = (
        f'<path d="M92 76 L84 84 M84 77.5 V84 H90.5" fill="none" stroke="{M.TICK}" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    return "".join(
        [
            clip("ig4"),
            M.tile(),
            slice_rect(M.TILE),
            '<g clip-path="url(#ig4)">',
            mitos(f'fill="none" stroke="{M.RIM}" stroke-width="1.5" stroke-dasharray="3 3"'),
            mitos(
                f'fill="{M.ACC}" fill-opacity="0.32" stroke="{M.ACC}" stroke-width="3"',
                dx=dx,
                dy=dy,
            ),
            "</g>",
            arrow,
        ]
    )


def tile_fused():
    """5. Fluorescence on the EM, through the whole stack."""
    x, y, s = SLICE
    b = [clip("ig5"), M.tile()]
    for d in (8, 4):
        b.append(slice_rect(M.TILE, x=x + d, y=y - d + 6, s=s - 8))
    b.append(slice_rect(M.GRID_FILL, x=x, y=y + 6, s=s - 8))
    b.append('<g clip-path="url(#ig5)"><g transform="translate(-3 4) scale(0.92)">')
    b.append(nucleus())
    b.append(
        mitos(
            f'fill="{M.ACC}" fill-opacity="0.32" stroke="{M.ACC}" stroke-width="3"',
            uid="ig5",
            cristae_col=M.TEXT_DARK_BG,
        )
    )
    b.append("</g></g>")
    return "".join(b)


STEPS = [
    (
        tile_volumes,
        "Two volumes",
        "A FIB-SEM or SBF-SEM stack and a confocal z-stack of the same cells.",
    ),
    (
        tile_segment,
        "Segment the EM",
        "A model finds one organelle in every EM slice, usually mitochondria.",
    ),
    (
        tile_blur,
        "Blur it like the confocal",
        "The mask is blurred with the confocal's PSF into synthetic fluorescence.",
    ),
    (
        tile_match,
        "Match real to synthetic",
        "Scan depth and z scale, then fit a 3D affine that moves the real channel onto it.",
    ),
    (
        tile_fused,
        "Fused volume",
        "The fluorescence sits on the EM. Export it as OME-Zarr, with the transform files.",
    ),
]


def build():
    W, H, mx = 1600, 730, 70
    tw, gap = 236, 70
    with open(
        os.path.join(REPO, "src", "pyclem3d", "assets", "logo", "pyclem3d_logo_on_light.svg"),
        encoding="utf-8",
    ) as fh:
        logo = fh.read()
    lw, lh = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', logo).groups())
    inner = logo[logo.index(">", logo.index("<svg")) + 1 : logo.rindex("</svg>")]
    inner = re.sub(r"<title>.*?</title>", "", inner)
    b = ['<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H)]
    k = 84 / lh
    b.append('<g transform="translate(%d 44) scale(%s)">%s</g>' % (mx, n(k), inner))
    b.append(heavy("How it works", W - mx, 92, 40, INK, "end"))
    b.append(
        '<text x="%d" y="122" text-anchor="end" font-family="%s" font-size="20" fill="%s">'
        "Organelles segmented in the EM carry the fluorescence onto it, in 3D.</text>"
        % (W - mx, BODY_FONT, MUTED)
    )
    b.append('<rect x="%d" y="160" width="%d" height="2" fill="%s"/>' % (mx, W - 2 * mx, RULE))
    ty = 206
    for i, (fn, title, text) in enumerate(STEPS):
        x = mx + i * (tw + gap)
        b.append('<g transform="translate(%d %d) scale(%s)">%s</g>' % (x, ty, n(tw / 120), fn()))
        if i:
            ax, ay = x - gap / 2, ty + tw / 2
            b.append(
                '<path d="M{} {}H{} M{} {}L{} {}L{} {}" fill="none" stroke="{}" stroke-width="4" '
                'stroke-linecap="round" stroke-linejoin="round"/>'.format(
                    n(ax - 16),
                    n(ay),
                    n(ax + 14),
                    n(ax + 4),
                    n(ay - 11),
                    n(ax + 15),
                    n(ay),
                    n(ax + 4),
                    n(ay + 11),
                    "#9aa3b2",
                )
            )
        cy = ty + tw + 46
        b.append(
            '<rect x="%d" y="%s" width="30" height="30" rx="8" fill="%s"/>'
            % (x, n(cy - 23), M.TILE)
        )
        b.append(heavy(str(i + 1), x + 15, cy - 1, 20, M.ACC, "middle"))
        b.append(heavy(title, x + 40, cy, 18, INK))
        b.append(body(text, x, cy + 36, 17, tw + 2))
    fy = H - 62
    b.append(
        '<rect x="%d" y="%d" width="%d" height="2" fill="%s"/>' % (mx, fy - 34, W - 2 * mx, RULE)
    )
    b.append(
        '<rect x="%d" y="%d" width="7" height="11" rx="2" fill="%s"/>' % (mx, fy - 10, M.TICK_LIGHT)
    )
    b.append(
        '<text x="%d" y="%d" font-family="%s" font-size="17" fill="%s">'
        "Step 4 starts from a rough alignment, such as a few landmarks picked in napari. "
        "The EM is never resampled: the fluorescence is moved onto it.</text>"
        % (mx + 18, fy, BODY_FONT, MUTED)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" '
        'aria-label="How pyCLEM3D works"><title>How pyCLEM3D works</title>%s</svg>'
        % (W, H, W, H, "".join(b))
    )
    os.makedirs(OUT, exist_ok=True)
    with open(
        os.path.join(OUT, "pyclem3d_how_it_works.svg"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write(svg)
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=W * 3 // 2, height=H * 3 // 2))
    with open(os.path.join(OUT, "pyclem3d_how_it_works.png"), "wb") as fh:
        fh.write(png)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
