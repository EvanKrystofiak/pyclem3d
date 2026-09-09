"""napari reader contribution: any supported file opens as lazy multiscale layers in nm."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..io.readers import open_volume
from ..io.volume import Volume

SUFFIXES = (".mrc", ".rec", ".st", ".ali", ".tif", ".tiff", ".zarr", ".lif", ".czi", ".nd2")


def napari_get_reader(path):
    p = path[0] if isinstance(path, (list, tuple)) else path
    s = str(p).lower()
    if os.path.isdir(p) or s.endswith(SUFFIXES):
        return reader_function
    return None


def _cmap_from_hex(color: str | None) -> Any:
    if not color or color.upper() in ("FFFFFF", "FFF"):
        return "gray"
    c = color.lstrip("#")
    rgb = tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    from vispy.color import Colormap

    return Colormap([(0, 0, 0, 1), (*rgb, 1)])


_UNITS: bool | None = None


def _supports_units() -> bool:
    global _UNITS
    if _UNITS is None:
        try:
            import inspect

            import napari

            _UNITS = "units" in inspect.signature(napari.Viewer.add_image).parameters
        except Exception:  # pragma: no cover
            _UNITS = False
    return _UNITS


def volume_layers(
    vol: Volume, prefix: str = "", transform: np.ndarray | None = None, visible: bool = True
) -> list[tuple]:
    """LayerData tuples (one image layer per channel), placed in world nm via the layer affine.

    ``transform`` (4x4, world -> world) is composed on top of the volume's voxel->world affine,
    which is how the LM is shown registered to the EM without resampling (plan §8).
    """
    A = vol.world_affine if transform is None else np.asarray(transform) @ vol.world_affine
    layers = []
    for c in vol.channels:
        levels = [vol.level_data(i)[c.index] for i in range(vol.n_levels())]
        data = levels if len(levels) > 1 else levels[0]
        meta: dict[str, Any] = {
            "name": f"{prefix}{c.name}",
            "affine": A,
            "blending": "additive" if vol.kind == "lm" else "translucent",
            "colormap": _cmap_from_hex(c.color) if vol.kind == "lm" else "gray",
            "visible": visible and c.visible,
            "metadata": {
                "pyclem3d": {
                    "kind": vol.kind,
                    "channel": c.index,
                    "source": vol.source,
                    "voxel_size_nm": list(vol.voxel_size_nm),
                }
            },
            "multiscale": len(levels) > 1,
        }
        if _supports_units():
            meta["units"] = ("nm", "nm", "nm")
        if c.display_min is not None and c.display_max is not None:
            meta["contrast_limits"] = (c.display_min, c.display_max)
        layers.append((data, meta, "image"))
    return layers


def reader_function(path):
    p = path[0] if isinstance(path, (list, tuple)) else path
    kind = "lm" if str(p).lower().endswith((".lif", ".czi", ".nd2")) else "em"
    vol = open_volume(p, kind=kind)
    return volume_layers(vol)
