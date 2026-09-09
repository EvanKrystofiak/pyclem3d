"""Anisotropy-aware multiscale pyramid, built lazily and cached on demand (plan §5)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import dask.array as da
import numpy as np

from .volume import Volume, default_chunks

log = logging.getLogger(__name__)


def next_factors(voxel_size_nm: tuple[float, float, float]) -> tuple[int, int, int]:
    """Which axes to halve at the next level: those within 2x of the finest spacing.

    An anisotropic confocal (300/100 nm) is downsampled in xy only until it is
    roughly isotropic, then in all axes; isotropic FIB-SEM halves all axes at once.
    """
    v = np.asarray(voxel_size_nm, dtype=float)
    m = v.min()
    return tuple(2 if vi < 2.0 * m else 1 for vi in v)  # type: ignore[return-value]


def plan_levels(
    shape_zyx: tuple[int, int, int],
    voxel_size_nm: tuple[float, float, float],
    max_levels: int = 8,
    min_size: int = 64,
) -> list[tuple[int, int, int]]:
    """Cumulative (fz, fy, fx) factors per level, level 0 = (1,1,1)."""
    factors: list[tuple[int, int, int]] = [(1, 1, 1)]
    shape = np.asarray(shape_zyx, dtype=int)
    vs = np.asarray(voxel_size_nm, dtype=float)
    cum = np.ones(3, dtype=int)
    for _ in range(max_levels - 1):
        f = np.asarray(next_factors(tuple(vs)), dtype=int)
        new_shape = shape // f
        if (new_shape[f == 2] < min_size).any() or (new_shape < 1).any():
            break
        shape = new_shape
        vs = vs * f
        cum = cum * f
        factors.append((int(cum[0]), int(cum[1]), int(cum[2])))
    return factors


def coarsen(arr: da.Array, factors: tuple[int, int, int]) -> da.Array:
    """Mean-downsample a (C,Z,Y,X) dask array by integer factors (trims the excess)."""
    if all(f == 1 for f in factors):
        return arr
    axes = {i + 1: int(f) for i, f in enumerate(factors) if f > 1}
    dtype = arr.dtype
    chunks = list(arr.chunks)
    rechunk = {}
    for ax, f in axes.items():
        if any(c % f for c in chunks[ax][:-1]):
            rechunk[ax] = max(f, (chunks[ax][0] // f) * f)
    if rechunk:
        arr = arr.rechunk(rechunk)
    out = da.coarsen(np.mean, arr, axes, trim_excess=True)
    if np.issubdtype(dtype, np.integer):
        out = da.round(out).astype(dtype)
    else:
        out = out.astype(dtype)
    return out


def build_pyramid(
    data: da.Array,
    voxel_size_nm: tuple[float, float, float],
    max_levels: int = 8,
    min_size: int = 64,
) -> tuple[list[da.Array], list[tuple[int, int, int]]]:
    """Lazy pyramid (list of dask arrays) and cumulative factors."""
    shape = (int(data.shape[1]), int(data.shape[2]), int(data.shape[3]))
    factors = plan_levels(shape, voxel_size_nm, max_levels, min_size)
    levels = [data]
    prev = data
    for i in range(1, len(factors)):
        step = tuple(factors[i][k] // factors[i - 1][k] for k in range(3))
        nxt = coarsen(prev, step)  # type: ignore[arg-type]
        nxt = nxt.rechunk(default_chunks(nxt.shape, nxt.dtype.itemsize))
        levels.append(nxt)
        prev = nxt
    return levels, factors


def _source_key(volume: Volume, factors: list[tuple[int, int, int]]) -> str:
    src = volume.source or ""
    stat = ""
    p = Path(src) if src else None
    if p is not None and p.exists():
        st = p.stat()
        stat = f"{st.st_size}:{st.st_mtime_ns}"
        if p.is_dir():
            try:
                stat += ":" + str(len(os.listdir(p)))
            except OSError:
                pass
    payload = json.dumps(
        {
            "src": src,
            "stat": stat,
            "shape": list(int(s) for s in volume.data.shape),
            "dtype": str(volume.dtype),
            "factors": factors,
            "vs": list(volume.voxel_size_nm),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def default_cache_dir() -> Path:
    env = os.environ.get("PYCLEM3D_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".pyclem3d_cache"


def ensure_pyramid(
    volume: Volume,
    cache_dir: str | os.PathLike | None = None,
    max_levels: int = 8,
    min_size: int = 64,
    eager: bool = False,
    persist_small: bool = True,
) -> Volume:
    """Attach a pyramid to ``volume``, using / filling the on-disk cache.

    * If the volume already carries a pyramid (e.g. OME-Zarr multiscales) it is kept.
    * Otherwise levels are computed lazily; ``eager=True`` (the CLI ``pyramid``
      command) materialises them into ``cache_dir`` as a v2 zarr group, after
      which later sessions read the cached levels instead of recomputing.
    * Levels smaller than ~64 MB are persisted in RAM regardless (they are what
      the viewer touches constantly).
    """
    if volume.pyramid is not None and len(volume.pyramid) > 1:
        return volume
    levels, factors = build_pyramid(volume.data, volume.voxel_size_nm, max_levels, min_size)
    if len(levels) == 1:
        return volume.with_(pyramid=levels, pyramid_factors=factors)

    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    key = _source_key(volume, factors)
    cache_path = cache_root / f"{key}.pyr.zarr"
    from .writers import open_group_v2, to_zarr_v2

    if cache_path.exists():
        try:
            open_group_v2(cache_path, mode="r")
            cached = [volume.data] + [
                da.from_zarr(str(cache_path), component=str(i)) for i in range(1, len(levels))
            ]
            log.info("pyramid: using cache %s", cache_path)
            levels = cached
        except Exception as e:  # pragma: no cover - corrupt cache
            log.warning("pyramid cache unreadable (%s); rebuilding", e)
            cache_path = cache_path.with_suffix(".rebuild.zarr")

    if eager and not cache_path.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        g = open_group_v2(cache_path, mode="w")
        g.attrs["source"] = volume.source
        g.attrs["factors"] = factors
        prev = volume.data
        stored = [volume.data]
        for i in range(1, len(levels)):
            step = tuple(factors[i][k] // factors[i - 1][k] for k in range(3))
            lvl = coarsen(prev, step)  # type: ignore[arg-type]
            lvl = lvl.rechunk(default_chunks(lvl.shape, lvl.dtype.itemsize))
            to_zarr_v2(lvl, cache_path, str(i))
            prev = da.from_zarr(str(cache_path), component=str(i))
            stored.append(prev)
        levels = stored
        log.info("pyramid: built %d levels into %s", len(levels), cache_path)

    if persist_small:
        out = []
        for lvl in levels[1:]:
            nbytes = int(np.prod(lvl.shape)) * lvl.dtype.itemsize
            out.append(lvl.persist() if nbytes <= (64 << 20) else lvl)
        levels = [levels[0]] + out
    return volume.with_(pyramid=levels, pyramid_factors=factors)
