"""Render the mark directions that were considered, at 256, 48, 24 and 16 px, on one
labelled contact sheet (24 and 16 also shown at 4x, nearest-neighbour).

usage: python explore.py OUT_DIR
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitted as F
import marks as M
import resvg_py
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

OPTIONS = [  # file name, label, body, pixel-fitted {size: svg} or None
    (
        "1_overlay_tilted",
        "1  Overlay, tilted (built)",
        M.overlay(tilt=-24),
        {24: F.mark_24(), 16: F.mark_16()},
    ),
    ("2_overlay_level", "2  Overlay, level", M.overlay(), None),
    ("3_swipe", "3  Swipe", M.swipe(), None),
    ("4_halo", "4  Halo", M.halo(), None),
]


def png(svg_text, size):
    data = bytes(resvg_py.svg_to_bytes(svg_string=svg_text, width=size, height=size))
    return Image.open(io.BytesIO(data)).convert("RGBA")


def sheet(path):
    pad, big, head = 28, 256, 44
    col_w = big + pad + 24 * 4 + pad + 16 * 4 + pad * 2
    W, H = pad + len(OPTIONS) * col_w, head + big + pad
    im = Image.new("RGBA", (W, H * 2), "#1f232c")
    im.paste(Image.new("RGBA", (W, H), "#ffffff"), (0, H))
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=20)
    for i, (name, label, body, fitted) in enumerate(OPTIONS):
        master = M.svg(body, 120, 120)
        with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(master)
        for row in (0, 1):
            x, y = pad + i * col_w, row * H + head
            draw.text((x, y - 32), label, font=font, fill="#e7ebf2" if row == 0 else "#1c212b")
            im.alpha_composite(png(master, big), (x, y))
            x += big + pad
            im.alpha_composite(png(master, 48), (x, y))
            for s in (24, 16):
                a = png(fitted[s] if fitted else master, s)
                im.alpha_composite(a, (x, y + 64))
                im.alpha_composite(a.resize((s * 4, s * 4), Image.NEAREST), (x, y + 104))
                x += s * 4 + pad
    im.save(path)


if __name__ == "__main__":
    sheet(os.path.join(OUT, "directions.png"))
    print("ok")
