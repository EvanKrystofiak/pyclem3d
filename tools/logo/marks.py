"""pyCLEM3D mark directions (120 x 120 masters), in the pyFIB family's parts and palette.

Each function returns the SVG body (no <svg> wrapper) for one direction.
"""

ACC, ACC_LIGHT = "#17c3b2", "#12776f"
TICK, TICK_LIGHT = "#f5b841", "#b07508"
TILE, EDGE = "#161a22", "#3a4150"
TEXT_DARK_BG, TEXT_LIGHT_BG = "#e7ebf2", "#1c212b"
GRID_FILL, GRID_EDGE, RIM = "#222936", "#3a4150", "#4a5366"


def n(v):
    return (f"{v:.2f}").rstrip("0").rstrip(".")


def svg(body, w, h, name="pyCLEM3D"):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{n(w)}" height="{n(h)}" viewBox="0 0 {n(w)} {n(h)}" '
        f'role="img" aria-label="{name}"><title>{name}</title>{body}</svg>'
    )


def tile():
    return (
        f'<rect x="2" y="2" width="116" height="116" rx="26" fill="{TILE}" stroke="{EDGE}" '
        'stroke-width="2"/>'
    )


def capsule(cx, cy, w, h, attrs):
    return f'<rect x="{n(cx - w / 2)}" y="{n(cy - h / 2)}" width="{n(w)}" height="{n(h)}" rx="{n(h / 2)}" {attrs}/>'


def bar(x, y, w, h, col, rx=1.5):
    return f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="{n(rx)}" fill="{col}"/>'


def cristae(cx, cy, h, xs, col, bw=4, frac=0.56, inset=0, first_top=True):
    """Short bars reaching in from alternate sides of a horizontal capsule (the
    mitochondrion's cristae). xs are bar centres; inset = gap from the membrane."""
    out = []
    ln = (h - 2 * inset) * frac
    top = first_top
    for x in xs:
        y = cy - h / 2 + inset if top else cy + h / 2 - inset - ln
        out.append(bar(x - bw / 2, y, bw, ln, col))
        top = not top
    return "".join(out)


def brackets(x0, y0, x1, y1, arm, sw, col):
    p = [
        f"M{n(cx)} {n(cy + dy * arm)}V{n(cy)}H{n(cx + dx * arm)}"
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1))
    ]
    return (
        '<path d="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" '
        'stroke-linejoin="round"/>'.format(" ".join(p), col, n(sw))
    )


# --------------------------------------------------------------------------------- #
# A. "Swipe": one mitochondrion under the swipe divider. Left of the divider it is
#    the EM picture (membrane and cristae on the grey slice); right of it, the same
#    organelle as its fluorescence (teal, no ultrastructure).
# --------------------------------------------------------------------------------- #
def swipe(em_line=TEXT_DARK_BG):
    cx, cy, w, h = 60, 60, 70, 36
    x0, y0, s = 18, 18, 84
    b = [
        "<defs>"
        '<clipPath id="clem-l"><rect x="0" y="0" width="60" height="120"/></clipPath>'
        '<clipPath id="clem-r"><rect x="60" y="0" width="60" height="120"/></clipPath>'
        '<clipPath id="clem-m">{}</clipPath>'
        "</defs>".format(capsule(cx, cy, w, h, "")),
        tile(),
    ]
    # EM half of the field: grey slice
    b.append(
        '<g clip-path="url(#clem-l)"><rect x="%d" y="%d" width="%d" height="%d" rx="3" '
        'fill="%s" stroke="%s" stroke-width="2"/></g>' % (x0, y0, s, s, GRID_FILL, GRID_EDGE)
    )
    # fluorescence half: dark field, same frame
    b.append(
        '<g clip-path="url(#clem-r)"><rect x="%d" y="%d" width="%d" height="%d" rx="3" '
        'fill="none" stroke="%s" stroke-width="2"/></g>' % (x0, y0, s, s, GRID_EDGE)
    )
    ctx = f'fill="none" stroke="{RIM}" stroke-width="2"'
    b.append(
        '<defs><clipPath id="clem-f"><rect x="19" y="19" width="41" height="82" rx="2"/>'
        f'</clipPath></defs><g clip-path="url(#clem-f)">{capsule(24, 31, 34, 13, ctx)}{capsule(50, 90, 34, 13, ctx)}</g>'
    )
    # EM half of the mitochondrion: membrane + cristae
    b.append(
        '<g clip-path="url(#clem-l)">{}<g clip-path="url(#clem-m)">{}</g></g>'.format(
            capsule(cx, cy, w - 3, h - 3, f'fill="{TILE}" stroke="{em_line}" stroke-width="3"'),
            cristae(cx, cy, h, (36, 48), em_line),
        )
    )
    # fluorescence half: the family's teal shape
    b.append(
        '<g clip-path="url(#clem-r)">{}</g>'.format(
            capsule(
                cx,
                cy,
                w - 3,
                h - 3,
                f'fill="{ACC}" fill-opacity="0.32" stroke="{ACC}" stroke-width="3"',
            )
        )
    )
    # the swipe divider, with its handle (pyFIB's tick)
    b.append(f'<rect x="58.5" y="14" width="3" height="92" rx="1.5" fill="{TICK}"/>')
    b.append(f'<rect x="56.5" y="88" width="7" height="11" rx="2" fill="{TICK}"/>')
    return "".join(b)


# --------------------------------------------------------------------------------- #
# B. "Overlay": an EM slice (top of a stack) with its mitochondria segmented; the
#    fluorescence lands on one of them, held in landmark brackets.
# --------------------------------------------------------------------------------- #
def overlay(tilt=0):
    b = [
        '<defs><clipPath id="clem-s"><rect x="18" y="28" width="74" height="74" rx="2"/></clipPath>'
        "</defs>",
        tile(),
    ]
    # the stack: two slices behind, then the top slice
    for d in (10, 5):
        b.append(
            '<rect x="%d" y="%d" width="76" height="76" rx="3" fill="%s" stroke="%s" '
            'stroke-width="2"/>' % (17 + d, 27 - d, TILE, GRID_EDGE)
        )
    b.append(
        f'<rect x="17" y="27" width="76" height="76" rx="3" fill="{GRID_FILL}" stroke="{GRID_EDGE}" '
        'stroke-width="2"/>'
    )
    # context organelles, cropped by the slice
    ctx = f'fill="none" stroke="{RIM}" stroke-width="2"'
    b.append('<g clip-path="url(#clem-s)">')
    b.append(capsule(30, 40, 36, 14, ctx))
    b.append(capsule(84, 92, 34, 14, ctx))
    b.append(f'<circle cx="92" cy="30" r="17" {ctx}/>')
    b.append("</g>")
    # the lit mitochondrion
    cx, cy, w, h = 55, 66, 46, 22
    b.append('<defs><clipPath id="clem-m2">{}</clipPath></defs>'.format(capsule(cx, cy, w, h, "")))
    b.append(f'<g transform="rotate({n(tilt)} {n(cx)} {n(cy)})">')
    b.append(
        capsule(cx, cy, w, h, f'fill="{ACC}" fill-opacity="0.32" stroke="{ACC}" stroke-width="3"')
    )
    b.append(
        f'<g clip-path="url(#clem-m2)">{cristae(cx, cy, h, (44, 55, 66), TEXT_DARK_BG, bw=3.5, frac=0.6)}</g>'
    )
    b.append("</g>")
    b.append(brackets(cx - w / 2 - 6, cy - h / 2 - 6, cx + w / 2 + 6, cy + h / 2 + 6, 8, 3, TICK))
    return "".join(b)


# --------------------------------------------------------------------------------- #
# C. "Checker": the checkerboard compare view. Four tiles alternate EM and
#    fluorescence over one mitochondrion that runs through all of them.
# --------------------------------------------------------------------------------- #
def checker():
    cx, cy, w, h = 60, 60, 70, 36
    lo, mid, hi, gap = 18, 60, 102, 3
    quads = {  # name: (x, y, w, h, is_em)
        "tl": (lo, lo, mid - lo - gap, mid - lo - gap, True),
        "tr": (mid + gap, lo, hi - mid - gap, mid - lo - gap, False),
        "bl": (lo, mid + gap, mid - lo - gap, hi - mid - gap, False),
        "br": (mid + gap, mid + gap, hi - mid - gap, hi - mid - gap, True),
    }
    defs = ['<clipPath id="clem-c">{}</clipPath>'.format(capsule(cx, cy, w, h, ""))]
    for k, (x, y, qw, qh, _) in quads.items():
        defs.append(
            f'<clipPath id="clem-{k}"><rect x="{n(x)}" y="{n(y)}" width="{n(qw)}" height="{n(qh)}" rx="3"/>'
            "</clipPath>"
        )
    b = ["<defs>{}</defs>".format("".join(defs)), tile()]
    for k, (x, y, qw, qh, em) in quads.items():
        b.append(
            f'<rect x="{n(x)}" y="{n(y)}" width="{n(qw)}" height="{n(qh)}" rx="3" fill="{GRID_FILL if em else TILE}" stroke="{GRID_EDGE}" '
            'stroke-width="2"/>'
        )
        if em:
            inner = (
                capsule(
                    cx, cy, w - 3, h - 3, f'fill="{TILE}" stroke="{TEXT_DARK_BG}" stroke-width="3"'
                )
                + f'<g clip-path="url(#clem-c)">{cristae(cx, cy, h, (36, 48, 72, 84), TEXT_DARK_BG, first_top=True)}</g>'
            )
        else:
            inner = capsule(
                cx,
                cy,
                w - 3,
                h - 3,
                f'fill="{ACC}" fill-opacity="0.32" stroke="{ACC}" ' 'stroke-width="3"',
            )
        b.append(f'<g clip-path="url(#clem-{k})">{inner}</g>')
    b.append(f'<rect x="56.5" y="54.5" width="7" height="11" rx="2" fill="{TICK}"/>')
    return "".join(b)


# --------------------------------------------------------------------------------- #
# D. "Halo": as B, but the organelle keeps its EM look (light membrane and cristae)
#    and the fluorescence is the wider, blurred teal shape lying over it.
# --------------------------------------------------------------------------------- #
def halo(tilt=0):
    b = [
        '<defs><clipPath id="clem-s"><rect x="18" y="28" width="74" height="74" rx="2"/></clipPath>'
        "</defs>",
        tile(),
    ]
    for d in (10, 5):
        b.append(
            '<rect x="%d" y="%d" width="76" height="76" rx="3" fill="%s" stroke="%s" '
            'stroke-width="2"/>' % (17 + d, 27 - d, TILE, GRID_EDGE)
        )
    b.append(
        f'<rect x="17" y="27" width="76" height="76" rx="3" fill="{GRID_FILL}" stroke="{GRID_EDGE}" '
        'stroke-width="2"/>'
    )
    ctx = f'fill="none" stroke="{RIM}" stroke-width="2"'
    b.append('<g clip-path="url(#clem-s)">')
    b.append(capsule(28, 38, 36, 13, ctx))
    b.append(capsule(86, 94, 34, 13, ctx))
    b.append(f'<circle cx="92" cy="30" r="17" {ctx}/>')
    b.append("</g>")
    cx, cy, w, h = 55, 66, 42, 20
    g = ['<defs><clipPath id="clem-m3">{}</clipPath></defs>'.format(capsule(cx, cy, w, h, ""))]
    g.append(
        capsule(
            cx,
            cy,
            w + 14,
            h + 14,
            f'fill="{ACC}" fill-opacity="0.32" stroke="{ACC}" ' 'stroke-width="3"',
        )
    )
    g.append(
        capsule(cx, cy, w - 2.5, h - 2.5, f'fill="none" stroke="{TEXT_DARK_BG}" stroke-width="2.5"')
    )
    g.append(
        f'<g clip-path="url(#clem-m3)">{cristae(cx, cy, h, (45, 55, 65), TEXT_DARK_BG, bw=3, frac=0.6)}</g>'
    )
    b.append('<g transform="rotate({} {} {})">{}</g>'.format(n(tilt), n(cx), n(cy), "".join(g)))
    return "".join(b)


def overlay_tilt():
    return overlay(tilt=-24)


DIRECTIONS = {
    "A_swipe": swipe,
    "B_overlay": overlay,
    "B2_overlay_tilt": overlay_tilt,
    "D_halo": halo,
}
