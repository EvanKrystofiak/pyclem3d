"""Build the pyCLEM3D logo set (same files and layout as pyFIB's and pyMap's).

usage: python make_logo.py OUT_DIR ARCHIVO_800_WOFF PYFIB_WORDMARK_SVG

Needs fontTools and resvg-py (pip install fonttools resvg-py); resvg stands in for
rsvg-convert, which the pyMap script used.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitted as F
import marks as M
import resvg_py
import wordmark as W

OUT, FONT, PYFIB_WM = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
P = "pyclem3d"
n = M.n


def mark_body():
    return M.overlay(tilt=-24)


def wordmark(on_light):
    f, py = W.load(FONT, PYFIB_WM)
    return W.build(
        f,
        py,
        accent=M.ACC_LIGHT if on_light else M.ACC,
        text=M.TEXT_LIGHT_BG if on_light else M.TEXT_DARK_BG,
        tick=M.TICK_LIGHT if on_light else M.TICK,
        fill_opacity=0.28 if on_light else 0.32,
    )


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def build_all():
    sizes = {}
    write(P + "_mark.svg", M.svg(mark_body(), 120, 120))
    sizes[P + "_mark.svg"] = (120, 120)
    write(P + "_mark_16.svg", F.mark_16())
    write(P + "_mark_24.svg", F.mark_24())
    for tone in ("dark", "light"):
        body, x0, y0, x1, y1 = wordmark(tone == "light")
        w, h = x1 - x0, y1 - y0
        name = f"{P}_wordmark_on_{tone}.svg"
        write(name, M.svg(f'<g transform="translate({n(-x0)} {n(-y0)})">{body}</g>', w, h))
        sizes[name] = (w, h)
        # lockup as pyFIB's: mark at 1.715x (205.8 high), the capitals' middle on the
        # mark's middle, wordmark group 250.96 to the right of the mark's origin
        k = 1.715
        mh = 120 * k
        dy = mh / 2 - (W.CAP_TOP + W.BASE) / 2
        dx = 250.96
        top, bot = min(0.0, y0 + dy), max(mh, y1 + dy)
        inner = f'<g transform="scale({n(k)})">{mark_body()}</g><g transform="translate({n(dx)} {n(dy)})">{body}</g>'
        name = f"{P}_logo_on_{tone}.svg"
        write(name, M.svg(f'<g transform="translate(0 {n(-top)})">{inner}</g>', dx + x1, bot - top))
        sizes[name] = (dx + x1, bot - top)
    return sizes


def render(svg_name, png_name, w, h):
    with open(os.path.join(OUT, svg_name), encoding="utf-8") as fh:
        data = bytes(
            resvg_py.svg_to_bytes(svg_string=fh.read(), width=int(round(w)), height=int(round(h)))
        )
    with open(os.path.join(OUT, png_name), "wb") as fh:
        fh.write(data)


def write_ico(sizes):
    pngs = [open(os.path.join(OUT, "%s_mark_%d.png" % (P, s)), "rb").read() for s in sizes]
    head = struct.pack("<HHH", 0, 1, len(pngs))
    off = 6 + 16 * len(pngs)
    dirs, data = b"", b""
    for s, p in zip(sizes, pngs):
        dirs += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(p), off + len(data))
        data += p
    with open(os.path.join(OUT, P + ".ico"), "wb") as fh:
        fh.write(head + dirs + data)


if __name__ == "__main__":
    sizes = build_all()
    render(P + "_mark_16.svg", P + "_mark_16.png", 16, 16)
    render(P + "_mark_24.svg", P + "_mark_24.png", 24, 24)
    for s in (32, 48, 64, 128, 256, 512, 1024):
        render(P + "_mark.svg", "%s_mark_%d.png" % (P, s), s, s)
    for tone in ("dark", "light"):
        for kind, h in (("logo", 600), ("wordmark", 300)):
            name = f"{P}_{kind}_on_{tone}"
            w0, h0 = sizes[name + ".svg"]
            render(name + ".svg", name + ".png", w0 * h / h0, h)
    write_ico((16, 24, 32, 48, 64, 128, 256))
    print("\n".join(sorted(os.listdir(OUT))))
