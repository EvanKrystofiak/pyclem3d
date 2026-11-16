"""pyCLEM3D wordmark, in pyFIB's wordmark coordinate system (Archivo 800 at 0.12 unit/fu,
baseline y = 200; "py" is pyFIB's own JetBrains Mono outline, reused unchanged).

The "EM" carries the mark's idea: the teal fluorescence shape lies over it, inside the
amber landmark brackets.
"""

import re

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.svgLib.path import parse_path
from fontTools.ttLib import TTFont

SCALE, BASE, ORIGIN = 0.12, 200.0, 88.0  # pyFIB's "F" origin = our "C"
CAP_TOP = 117.56


def load(font_path, pyfib_wordmark_svg):
    f = TTFont(font_path)
    with open(pyfib_wordmark_svg, encoding="utf-8") as fh:
        py = re.findall(r'<path fill="[^"]+" d="([^"]+)"', fh.read())[0]
    return f, py


def path_bounds(d):
    bp = BoundsPen(None)
    parse_path(d, bp)
    return bp.bounds  # x0, y0, x1, y1 in SVG coordinates


def glyph(f, ch, ox):
    gs = f.getGlyphSet()
    g = f.getBestCmap()[ord(ch)]
    pen = SVGPathPen(gs)
    gs[g].draw(TransformPen(pen, (SCALE, 0, 0, -SCALE, ox, BASE)))
    bp = BoundsPen(gs)
    gs[g].draw(bp)
    return pen.getCommands(), gs[g].width * SCALE, bp.bounds


def build(f, py, *, accent, text, tick, fill_opacity, lit="EM", pad=8.0, margin=7.0, rx=14.0):
    """'py' + 'CL' + lit 'EM' + '3D'. Returns (body, left, top, right, bottom): the SVG
    body and its ink box, in wordmark coordinates."""
    name = "CLEM3D"
    i0 = name.index(lit)
    i1 = i0 + len(lit)
    x = ORIGIN
    letters, lit_box = [], None
    for i, ch in enumerate(name):
        if i == i0 or i == i1:
            x += pad
        d, adv, b = glyph(f, ch, x)
        letters.append(d)
        if i0 <= i < i1:
            box = (x + b[0] * SCALE, BASE - b[3] * SCALE, x + b[2] * SCALE, BASE - b[1] * SCALE)
            lit_box = (
                box
                if lit_box is None
                else (
                    min(lit_box[0], box[0]),
                    min(lit_box[1], box[1]),
                    max(lit_box[2], box[2]),
                    max(lit_box[3], box[3]),
                )
            )
        right = x + b[2] * SCALE
        x += adv
    sx0, sy0 = lit_box[0] - margin, lit_box[1] - margin
    sx1, sy1 = lit_box[2] + margin, lit_box[3] + margin
    sw = 2.47  # pyFIB's I-pattern stroke (3 x 0.8232)
    deco = (
        f'<rect x="{sx0 + sw / 2:.2f}" y="{sy0 + sw / 2:.2f}" width="{sx1 - sx0 - sw:.2f}" height="{sy1 - sy0 - sw:.2f}" rx="{rx:.2f}" fill="{accent}" '
        f'fill-opacity="{fill_opacity}" stroke="{accent}" stroke-width="{sw:.2f}"/>'
    )
    g, arm, bw = 4.5, 11.0, 3.4  # brackets, as pyMap's
    bx0, by0, bx1, by1 = sx0 - g, sy0 - g, sx1 + g, sy1 + g
    p = [
        f"M{cx:.2f} {cy + dy * arm:.2f}V{cy:.2f}H{cx + dx * arm:.2f}"
        for cx, cy, dx, dy in (
            (bx0, by0, 1, 1),
            (bx1, by0, -1, 1),
            (bx0, by1, 1, -1),
            (bx1, by1, -1, -1),
        )
    ]
    deco += (
        '<path d="{}" fill="none" stroke="{}" stroke-width="{:.2f}" stroke-linecap="round" '
        'stroke-linejoin="round"/>'.format(" ".join(p), tick, bw)
    )
    pb = path_bounds(py)
    body = (
        f'<path fill="{accent}" d="{py}"/>'
        + deco
        + '<path fill="{}" d="{}"/>'.format(text, " ".join(letters))
    )
    return (body, pb[0], min(CAP_TOP, by0 - bw / 2, pb[1]), right, max(BASE, by1 + bw / 2, pb[3]))
