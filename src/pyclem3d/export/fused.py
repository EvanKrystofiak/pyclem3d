"""Output 1 (plan §9): the fused OME-Zarr - EM + each LM channel warped into EM world space +
coverage mask, one shared frame, multiscale, written chunk-by-chunk."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

from ..io.volume import Channel, Volume, default_chunks
from ..io.writers import write_ome_zarr_from_level0
from ..register.transforms import Transform
from ..resample.grid import DisplacementGrid, Grid
from ..resample.lazy import resample_to_grid
from ..resample.slab import em_level_for_lm

log = logging.getLogger(__name__)


def _common_dtype(*dtypes: np.dtype) -> np.dtype:
    if any(np.issubdtype(d, np.floating) for d in dtypes):
        return np.dtype(np.float32)
    bits = max(np.dtype(d).itemsize for d in dtypes)
    signed = any(np.issubdtype(d, np.signedinteger) for d in dtypes)
    if signed:
        return np.dtype(f"int{max(bits, 2) * 8}")
    return np.dtype(f"uint{bits * 8}")


def fused_grid(
    em: Volume,
    lm: Volume,
    voxel_size_nm: float | None = None,
    em_level: int | None = None,
    roi_world: np.ndarray | None = None,
) -> tuple[int, Grid, tuple[slice, slice, slice]]:
    """Choose the EM level (default: nearest the LM xy resolution) and the voxel ROI on it."""
    if em_level is None:
        em_level = (
            em.level_for_voxel_size(float(voxel_size_nm))
            if voxel_size_nm
            else em_level_for_lm(em, lm)
        )
    full = Grid.from_volume(em, em_level)
    Z, Y, X = full.shape_zyx
    sl = (slice(0, Z), slice(0, Y), slice(0, X))
    if roi_world is not None:
        roi = np.asarray(roi_world, dtype=float)
        corners = np.array([[a, b, c] for a in roi[:, 0] for b in roi[:, 1] for c in roi[:, 2]])
        vox = full.world_to_voxel(corners)
        lo = np.clip(np.floor(vox.min(0)).astype(int), 0, [Z - 1, Y - 1, X - 1])
        hi = np.clip(np.ceil(vox.max(0)).astype(int) + 1, 1, [Z, Y, X])
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[assignment]
        T = np.eye(4)
        T[:3, 3] = lo
        grid = Grid(tuple(int(b - a) for a, b in zip(lo, hi)), full.affine @ T)  # type: ignore[arg-type]
        return em_level, grid, sl
    return em_level, full, sl


def export_fused_ome_zarr(
    em: Volume,
    lm: Volume,
    transform: Transform,
    out_path: str | os.PathLike,
    voxel_size_nm: float | None = None,
    em_level: int | None = None,
    roi_world: np.ndarray | None = None,
    lm_level: int = 0,
    lm_channels: list[int] | None = None,
    displacement: DisplacementGrid | None = None,
    include_coverage: bool = True,
    name: str = "fused",
    max_levels: int = 8,
    min_size: int = 64,
    extra_attrs: dict[str, Any] | None = None,
) -> Path:
    """Write EM + warped LM (+ coverage) as one OME-Zarr in EM world space.

    Default resolution is the EM pyramid level nearest the LM xy voxel; pass
    ``em_level=0`` (or a small ``voxel_size_nm``) with ``roi_world`` for full-resolution
    crops. Nothing is materialised for the whole volume: every chunk is resampled on demand.
    """
    lvl, grid, sl = fused_grid(em, lm, voxel_size_nm, em_level, roi_world)
    em_data = em.level_data(lvl)[:, sl[0], sl[1], sl[2]]
    chans = list(range(lm.n_channels)) if lm_channels is None else list(lm_channels)
    lm_data, cov = resample_to_grid(
        lm, transform, grid, level=lm_level, channels=chans, displacement=displacement
    )
    dtype = _common_dtype(em.dtype, lm.dtype)
    ch = default_chunks((1, *grid.shape_zyx), dtype.itemsize)
    parts = [
        em_data.astype(dtype).rechunk((1, ch[1], ch[2], ch[3])),
        lm_data.astype(dtype).rechunk((1, ch[1], ch[2], ch[3])),
    ]
    channels: list[Channel] = []
    for c in em.channels:
        channels.append(
            Channel(
                f"EM:{c.name}",
                len(channels),
                c.color or "FFFFFF",
                c.display_min,
                c.display_max,
                c.intensity_min,
                c.intensity_max,
            )
        )
    for i in chans:
        c = lm.channels[i]
        channels.append(
            Channel(
                f"LM:{c.name}",
                len(channels),
                c.color,
                c.display_min,
                c.display_max,
                c.intensity_min,
                c.intensity_max,
                emission_nm=c.emission_nm,
            )
        )
    if include_coverage:
        hi = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
        parts.append((cov.astype(dtype) * dtype.type(hi))[None].rechunk((1, ch[1], ch[2], ch[3])))
        channels.append(Channel("coverage", len(channels), "808080", 0, hi, 0, hi, visible=False))
    fused = da.concatenate(parts, axis=0)
    attrs = {
        "pyclem3d": {
            "kind": "fused",
            "em_source": em.source,
            "lm_source": lm.source,
            "em_level": int(lvl),
            "grid": grid.to_dict(),
            "transform": _transform_summary(transform),
            "roi_world_nm": None if roi_world is None else np.asarray(roi_world).tolist(),
        }
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    out = write_ome_zarr_from_level0(
        out_path,
        fused,
        grid.voxel_size_nm,
        channels,
        name=name,
        translation_nm=grid.origin_nm,
        max_levels=max_levels,
        min_size=min_size,
        extra_attrs=attrs,
    )
    log.info(
        "fused OME-Zarr: %s channels=%s grid=%s", out, [c.name for c in channels], grid.shape_zyx
    )
    return out


def _transform_summary(t: Transform) -> dict[str, Any]:
    d = t.to_dict()
    if d.get("type") == "tps":
        return {
            "type": "tps",
            "lam": d.get("lam"),
            "n_control": len(d.get("control_pts", [])),
            "affine_part": t.affine_part().matrix.tolist(),
        }  # type: ignore[attr-defined]
    return d
