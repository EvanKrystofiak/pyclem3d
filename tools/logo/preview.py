"""Family preview: pyCLEM3D beside pyMap and pyFIB, on dark and on light, plus the
mark at every small size.

usage: python preview.py PYCLEM3D_LOGO_DIR KIT_DIR OUT_PNG
"""

import io
import os
import re
import sys

import resvg_py
from PIL import Image

NEW, KIT, OUT = sys.argv[1:4]
SETS = [
    (NEW, "pyclem3d"),
    (os.path.join(KIT, "pyMap_logo"), "pymap"),
    (os.path.join(KIT, "pyFIB_logo_reference"), "pyfib"),
]


def png(path, h=None, size=None):
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    w0, h0 = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
    if size:
        w, hh = size, size
    else:
        w, hh = round(w0 * h / h0), h
    return Image.open(
        io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=s, width=int(w), height=int(hh))))
    ).convert("RGBA")


def build():
    pad, lh = 40, 150
    rows = []
    for tone in ("dark", "light"):
        rows.append([png(os.path.join(d, f"{p}_logo_on_{tone}.svg"), h=lh) for d, p in SETS])
    W = pad + sum(im.width + pad * 2 for im in rows[0])
    band = lh + pad * 2
    small = 150
    im = Image.new("RGBA", (W, band * 2 + small), "#1f232c")
    im.paste(Image.new("RGBA", (W, band), "#ffffff"), (0, band + small))
    for r, y in ((rows[0], pad), (rows[1], band + small + pad)):
        x = pad
        for a in r:
            im.alpha_composite(a, (x, y))
            x += a.width + pad * 2
    # small sizes, on dark
    x, y = pad, band + 10
    for d, p in SETS:
        for s in (128, 64, 48, 32, 24, 16):
            src = os.path.join(d, "%s_mark_%d.svg" % (p, s))
            if not os.path.exists(src):
                src = os.path.join(d, f"{p}_mark.svg")
            a = png(src, size=s)
            im.alpha_composite(a, (x, y + (128 - s) // 2))
            x += s + 16
        x += pad
    im.save(OUT)


if __name__ == "__main__":
    build()
    print("ok")
