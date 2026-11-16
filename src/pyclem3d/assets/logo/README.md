# pyCLEM3D logo

Made 2026-11-16 to sit beside pyFIB's and pyMap's logos (see `family_preview.png`),
following the family's `LOGO_STYLE_GUIDE.md`. Chosen by Evan on 2026-11-16 from four
directions (direction 1, "overlay, tilted"; the others are in `options/`).

**The mark** is the top EM slice of a volume stack, seen from above, with its organelles
segmented. Three of them are plain grey outlines: two mitochondria and the arc of a
nucleus. The fourth mitochondrion carries the fluorescence: it is teal like pyFIB's
trenches and pyMap's target square, its cristae are light bars like the lamella, and it
sits inside amber landmark brackets. pyCLEM3D uses the segmented organelles to lay the
fluorescence over the EM volume, and this is the organelle where the two agree.

**In the wordmark**, the teal fluorescence shape lies over the "EM", inside the same
brackets. pyFIB does this with its "I" and pyMap with its "a".

## Files

| File | Use |
|---|---|
| `pyclem3d_mark.svg` | The mark (app icon), vector master, 120 × 120 |
| `pyclem3d_mark_16.svg`, `pyclem3d_mark_24.svg` | Pixel-fitted versions for 16 and 24 px |
| `pyclem3d_mark_<n>.png` | The mark at 16, 24, 32, 48, 64, 128, 256, 512 and 1024 px |
| `pyclem3d.ico` | Windows icon (16, 24, 32, 48, 64, 128, 256; PNG entries, as `pyfib.ico`) |
| `pyclem3d_logo_on_dark.svg/.png` | Mark + wordmark, for dark backgrounds (napari's default theme) |
| `pyclem3d_logo_on_light.svg/.png` | Mark + wordmark, for light backgrounds (slides, papers, the GitHub README) |
| `pyclem3d_wordmark_on_dark.svg/.png`, `pyclem3d_wordmark_on_light.svg/.png` | Wordmark alone |
| `family_preview.png` | pyCLEM3D beside pyMap and pyFIB |
| `tools/logo/options/` (repo root) | The four mark directions that were drawn; number 1 is the chosen one |

The PNGs have transparent backgrounds. The mark keeps its dark tile on any background.
The lockup follows pyFIB's exactly: the mark at 1.715×, the wordmark at the same offset,
and the capitals centred on the mark.

At 24 px the stack is one slice deep and the context organelles are dropped. At 16 px
the cristae go too. In both, the tilted mitochondrion is drawn pixel by pixel.

## Colours

The family palette, with no additions.

| Role | Hex |
|---|---|
| Accent: the lit mitochondrion, "py" on dark | `#17c3b2` |
| Accent on light backgrounds | `#12776f` |
| Brackets (tick) | `#f5b841` |
| Brackets on light backgrounds | `#b07508` |
| Tile / tile edge | `#161a22` / `#3a4150` |
| EM slice / slice edges / context organelles | `#222936` / `#3a4150` / `#4a5366` |
| Cristae and "CLEM3D" on dark | `#e7ebf2` |
| "CLEM3D" on light | `#1c212b` |

## Type

Same as pyFIB and pyMap: "py" is JetBrains Mono Medium (500), reused from pyFIB's own
outlines. "CLEM3D" is Archivo ExtraBold (800), at the same scale and baseline as pyFIB's
"F" and "B". Both fonts are SIL Open Font License 1.1. The letters are outlines, so no
font needs to be installed.

## Rebuilding

The generator lives outside the package, in `tools/logo/` at the repo root.
`make_logo.py` writes every logo file here:

    python tools/logo/make_logo.py src/pyclem3d/assets/logo archivo-latin-800-normal.woff pyfib_wordmark_on_dark.svg

It needs fontTools and resvg-py (`pip install fonttools resvg-py`). resvg stands in for
`rsvg-convert`, which the pyMap script used. The font and pyFIB's wordmark (which
supplies the "py") are in the family logo kit. `marks.py` holds all four directions; to
switch, change `mark_body()` in `make_logo.py` and redraw the two pixel-fitted icons in
`fitted.py`. `explore.py` redraws the options sheet and `preview.py` the family preview.

Keep clear space around the logo of at least the bracket arm's length. Don't recolour
the mark's parts separately, stretch it, or put the lockup on a busy image.
