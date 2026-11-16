"""Pixel-fitted 16 and 24 px versions of the pyCLEM3D mark (the tilted, lit
mitochondrion in its landmark brackets). Everything sits on whole pixels: the tilted
capsule is drawn pixel by pixel, not anti-aliased."""

import math

import marks as M


def px(cells, col, opacity=None):
    d = "".join("M%d %dh1v1h-1z" % c for c in sorted(cells))
    op = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return f'<path d="{d}" fill="{col}"{op}/>' if cells else ""


def capsule_cells(size, cx, cy, half_len, r, deg, rim=1.0):
    """Pixels of a tilted capsule: (membrane, inside)."""
    a = math.radians(deg)
    ux, uy = math.cos(a), math.sin(a)
    edge, inside = set(), set()
    for y in range(size):
        for x in range(size):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            t = max(-half_len, min(half_len, dx * ux + dy * uy))
            d = math.hypot(dx - t * ux, dy - t * uy)
            if d <= r - rim:
                inside.add((x, y))
            elif d <= r:
                edge.add((x, y))
    return edge, inside


def corner_cells(x0, y0, x1, y1, arm):
    """Bracket pixels at the four corners of the box x0..x1, y0..y1 (inclusive)."""
    c = set()
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for i in range(arm):
            c.add((cx + sx * i, cy))
            c.add((cx, cy + sy * i))
    return c


def mark_24():
    b = [
        f'<rect width="24" height="24" rx="5" fill="{M.TILE}"/>',
        # one slice behind, then the top slice
        f'<path d="M4.5 5V3.5H20.5V19.5H19" fill="none" stroke="{M.GRID_EDGE}" stroke-width="1"/>',
        f'<rect x="2.5" y="5.5" width="16" height="16" fill="{M.GRID_FILL}" stroke="{M.GRID_EDGE}" stroke-width="1"/>',
    ]
    edge, inside = capsule_cells(24, 10.5, 13.5, 3.0, 2.9, -24)
    b.append(px(inside, M.ACC, 0.35))
    b.append(px(edge, M.ACC))
    # three cristae, reaching in from alternate sides of the membrane
    b.append(px({(8, 12), (8, 13), (10, 14), (10, 15), (13, 10), (13, 11)}, M.TEXT_DARK_BG))
    b.append(px(corner_cells(4, 7, 16, 19, 2), M.TICK))
    return M.svg("".join(b), 24, 24)


def mark_16():
    b = [
        f'<rect width="16" height="16" rx="3.5" fill="{M.TILE}"/>',
        f'<rect x="1" y="2" width="13" height="13" fill="{M.GRID_FILL}"/>',
        f'<path d="M3 1h12v12h-1v-11h-11z" fill="{M.GRID_EDGE}"/>',
    ]
    edge, inside = capsule_cells(16, 7.5, 8.5, 2.0, 2.4, -24)
    b.append(px(inside, M.ACC, 0.35))
    b.append(px(edge, M.ACC))
    b.append(px(corner_cells(2, 4, 12, 12, 2), M.TICK))
    return M.svg("".join(b), 16, 16)
