"""Writers: OME-Zarr (NGFF 0.4, zarr v2 layout), OME-TIFF, MRC. All stream chunk-by-chunk.

The zarr v2 on-disk layout is used deliberately: it is what napari, Fiji (MoBIE),
neuroglancer and webKnossos all read today. ``open_group_v2`` / ``to_zarr_v2`` hide
the zarr 2 vs zarr 3 library API difference.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr

from .volume import Channel, default_chunks

log = logging.getLogger(__name__)

ZARR_V3_LIB = int(zarr.__version__.split(".")[0]) >= 3


def open_group_v2(path: str | os.PathLike, mode: str = "r"):
    """Open (or create) a zarr group on disk in the v2 layout, on any zarr library."""
    if ZARR_V3_LIB:
        if mode in ("w", "a", "w-"):
            return zarr.open_group(str(path), mode=mode, zarr_format=2)
        return zarr.open_group(str(path), mode=mode)
    return zarr.open_group(str(path), mode=mode)


def to_zarr_v2(
    arr: da.Array, path: str | os.PathLike, component: str, chunks: tuple[int, ...] | None = None
) -> None:
    """Write a dask array as ``path/component`` in v2 layout, streaming per chunk."""
    if chunks is not None:
        arr = arr.rechunk(chunks)
    kw: dict[str, Any] = {"overwrite": True}
    if ZARR_V3_LIB:
        kw["zarr_format"] = 2
    da.to_zarr(arr, str(path), component=component, **kw)


def _dtype_range(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def omero_channels(channels: Sequence[Channel], dtype: np.dtype) -> list[dict[str, Any]]:
    lo, hi = _dtype_range(dtype)
    out = []
    for ch in channels:
        imin = ch.intensity_min if ch.intensity_min is not None else lo
        imax = ch.intensity_max if ch.intensity_max is not None else hi
        start = ch.display_min if ch.display_min is not None else imin
        end = ch.display_max if ch.display_max is not None else imax
        out.append(
            {
                "label": ch.name,
                "color": (ch.color or "FFFFFF").upper(),
                "active": bool(ch.visible),
                "window": {"min": imin, "max": imax, "start": start, "end": end},
            }
        )
    return out


def write_ome_zarr(
    path: str | os.PathLike,
    levels: Sequence[da.Array],
    level_voxel_sizes_nm: Sequence[tuple[float, float, float]],
    channels: Sequence[Channel],
    name: str = "pyclem3d",
    translation_nm: tuple[float, float, float] | None = None,
    chunks: tuple[int, ...] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write a (C,Z,Y,X) multiscale image as OME-NGFF 0.4 in nanometre units.

    ``levels[i]`` is a dask array of level i; ``level_voxel_sizes_nm[i]`` its (dz,dy,dx).
    ``translation_nm`` is the world position of voxel (0,0,0) of level 0 (z,y,x);
    coarser levels get the mean-coarsening half-voxel offset added automatically.
    """
    path = Path(path)
    if len(levels) != len(level_voxel_sizes_nm):
        raise ValueError("levels and level_voxel_sizes_nm must have the same length")
    g = open_group_v2(path, mode="w")
    datasets = []
    base_vs = np.asarray(level_voxel_sizes_nm[0], dtype=float)
    t0 = np.asarray(translation_nm if translation_nm is not None else (0.0, 0.0, 0.0), dtype=float)
    for i, (lvl, vs) in enumerate(zip(levels, level_voxel_sizes_nm)):
        if lvl.ndim == 3:
            lvl = lvl[None]
        ch = chunks or default_chunks(lvl.shape, lvl.dtype.itemsize)
        ch = tuple(min(c, s) for c, s in zip(ch, lvl.shape))
        to_zarr_v2(lvl, path, str(i), chunks=ch)
        vs_arr = np.asarray(vs, dtype=float)
        f = vs_arr / base_vs
        trans = t0 + (f - 1.0) / 2.0 * base_vs
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, *[float(v) for v in vs_arr]]},
                    {"type": "translation", "translation": [0.0, *[float(v) for v in trans]]},
                ],
            }
        )
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": axes,
            "datasets": datasets,
            "type": "mean",
            "metadata": {"description": "written by pyclem3d", "method": "mean coarsening"},
        }
    ]
    g.attrs["omero"] = {
        "id": 1,
        "name": name,
        "version": "0.4",
        "channels": omero_channels(channels, levels[0].dtype),
        "rdefs": {"defaultZ": int(levels[0].shape[-3] // 2), "model": "color"},
    }
    if extra_attrs:
        for k, v in extra_attrs.items():
            g.attrs[k] = v
    log.info("wrote OME-Zarr %s (%d levels)", path, len(levels))
    return path


def _iter_planes(data: da.Array) -> Iterator[np.ndarray]:
    """Yield (Y, X) planes of a (C, Z, Y, X) dask array, computing one z-chunk at a time."""
    C, Z = data.shape[0], data.shape[1]
    zchunks = data.chunks[1]
    for c in range(C):
        z0 = 0
        for zc in zchunks:
            block = np.asarray(data[c, z0 : z0 + zc].compute())
            for k in range(block.shape[0]):
                yield block[k]
            z0 += zc
    assert z0 == Z


def write_ome_tiff(
    path: str | os.PathLike,
    data: da.Array,
    voxel_size_nm: tuple[float, float, float],
    channels: Sequence[Channel] | None = None,
    bigtiff: bool | None = None,
    compression: str | None = None,
) -> Path:
    """Stream a (C,Z,Y,X) dask array to OME-TIFF with physical sizes in micrometres."""
    import tifffile

    path = Path(path)
    if data.ndim == 3:
        data = data[None]
    C, Z, Y, X = (int(s) for s in data.shape)
    dz, dy, dx = (float(v) for v in voxel_size_nm)
    nbytes = C * Z * Y * X * data.dtype.itemsize
    if bigtiff is None:
        bigtiff = nbytes > (3.5 * 2**30)
    names = [c.name for c in channels] if channels else [f"ch{i}" for i in range(C)]
    meta: dict[str, Any] = {
        "axes": "CZYX",
        "PhysicalSizeX": dx / 1000.0,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": dy / 1000.0,
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": dz / 1000.0,
        "PhysicalSizeZUnit": "µm",
        "Channel": {"Name": names},
    }
    kw: dict[str, Any] = {}
    if compression:
        kw["compression"] = compression
    tifffile.imwrite(
        str(path),
        _iter_planes(data),
        shape=(C, Z, Y, X),
        dtype=data.dtype,
        ome=True,
        bigtiff=bigtiff,
        metadata=meta,
        photometric="minisblack",
        **kw,
    )
    log.info("wrote OME-TIFF %s", path)
    return path


_MRC_DTYPES = {
    np.dtype("int8"): np.dtype("int8"),
    np.dtype("uint8"): np.dtype("uint16"),  # lossless widening; MRC has no uint8 mode
    np.dtype("int16"): np.dtype("int16"),
    np.dtype("uint16"): np.dtype("uint16"),
    np.dtype("float32"): np.dtype("float32"),
    np.dtype("float16"): np.dtype("float16"),
}


def write_mrc(
    path: str | os.PathLike, data: da.Array, voxel_size_nm: tuple[float, float, float]
) -> Path:
    """Stream a (Z,Y,X) (or single-channel (1,Z,Y,X)) dask array to MRC, voxel size in the header."""
    import mrcfile

    path = Path(path)
    if data.ndim == 4:
        if data.shape[0] != 1:
            raise ValueError("MRC holds one channel; select a channel first")
        data = data[0]
    out_dtype = _MRC_DTYPES.get(data.dtype)
    if out_dtype is None:
        out_dtype = np.dtype("float32")
    Z, Y, X = (int(s) for s in data.shape)
    mode = {"int8": 0, "int16": 1, "float32": 2, "uint16": 6, "float16": 12}[out_dtype.name]
    with mrcfile.new_mmap(str(path), shape=(Z, Y, X), mrc_mode=mode, overwrite=True) as m:
        z0 = 0
        for zc in data.chunks[0]:
            block = np.asarray(data[z0 : z0 + zc].compute()).astype(out_dtype, copy=False)
            m.data[z0 : z0 + zc] = block
            z0 += zc
        dz, dy, dx = (float(v) for v in voxel_size_nm)
        m.voxel_size = (dx * 10.0, dy * 10.0, dz * 10.0)  # header wants Angstrom, (x, y, z)
        m.update_header_from_data()
        m.update_header_stats()
    log.info("wrote MRC %s", path)
    return path


def write_ome_zarr_from_level0(
    path: str | os.PathLike,
    level0: da.Array,
    voxel_size_nm: tuple[float, float, float],
    channels: Sequence[Channel],
    name: str = "pyclem3d",
    translation_nm: tuple[float, float, float] | None = None,
    max_levels: int = 8,
    min_size: int = 64,
    chunks: tuple[int, ...] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Stream level 0 to disk, then build the coarser levels *from the written level 0*.

    Use this when level 0 is an expensive lazy graph (a resampled/fused volume): every
    level is computed exactly once and no chunk of the graph is evaluated twice.
    """
    from .pyramid import coarsen, plan_levels

    path = Path(path)
    if level0.ndim == 3:
        level0 = level0[None]
    open_group_v2(path, mode="w")  # (re)create
    ch = chunks or default_chunks(level0.shape, level0.dtype.itemsize)
    ch = tuple(min(c, s) for c, s in zip(ch, level0.shape))
    to_zarr_v2(level0, path, "0", chunks=ch)
    prev = da.from_zarr(str(path), component="0")
    shape = (int(level0.shape[1]), int(level0.shape[2]), int(level0.shape[3]))
    factors = plan_levels(shape, voxel_size_nm, max_levels, min_size)
    levels = [prev]
    for i in range(1, len(factors)):
        step = tuple(factors[i][k] // factors[i - 1][k] for k in range(3))
        lvl = coarsen(prev, step).rechunk(default_chunks(prev.shape, prev.dtype.itemsize))  # type: ignore[arg-type]
        to_zarr_v2(lvl, path, str(i))
        prev = da.from_zarr(str(path), component=str(i))
        levels.append(prev)
    vs = [tuple(v * f for v, f in zip(voxel_size_nm, fac)) for fac in factors]
    _write_ome_attrs(path, levels, vs, channels, name, translation_nm, extra_attrs)
    log.info("wrote OME-Zarr %s (%d levels, streamed)", path, len(levels))
    return path


def _write_ome_attrs(
    path: Path,
    levels: Sequence[da.Array],
    level_voxel_sizes_nm: Sequence[tuple[float, float, float]],
    channels: Sequence[Channel],
    name: str,
    translation_nm: tuple[float, float, float] | None,
    extra_attrs: dict[str, Any] | None,
) -> None:
    g = open_group_v2(path, mode="a")
    base_vs = np.asarray(level_voxel_sizes_nm[0], dtype=float)
    t0 = np.asarray(translation_nm if translation_nm is not None else (0.0, 0.0, 0.0), dtype=float)
    datasets = []
    for i, vs in enumerate(level_voxel_sizes_nm):
        vs_arr = np.asarray(vs, dtype=float)
        f = vs_arr / base_vs
        trans = t0 + (f - 1.0) / 2.0 * base_vs
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, *[float(v) for v in vs_arr]]},
                    {"type": "translation", "translation": [0.0, *[float(v) for v in trans]]},
                ],
            }
        )
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": axes,
            "datasets": datasets,
            "type": "mean",
            "metadata": {"description": "written by pyclem3d", "method": "mean coarsening"},
        }
    ]
    g.attrs["omero"] = {
        "id": 1,
        "name": name,
        "version": "0.4",
        "channels": omero_channels(channels, levels[0].dtype),
        "rdefs": {"defaultZ": int(levels[0].shape[-3] // 2), "model": "color"},
    }
    if extra_attrs:
        for k, v in extra_attrs.items():
            g.attrs[k] = v


def write_volume_levels_ome_zarr(
    path: str | os.PathLike,
    levels: Sequence[da.Array],
    voxel_size_nm: tuple[float, float, float],
    factors: Sequence[tuple[int, int, int]],
    channels: Sequence[Channel],
    name: str,
    translation_nm: tuple[float, float, float] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    vs = [tuple(v * f for v, f in zip(voxel_size_nm, fac)) for fac in factors]
    return write_ome_zarr(
        path,
        levels,
        vs,
        channels,
        name=name,
        translation_nm=translation_nm,
        extra_attrs=extra_attrs,
    )
