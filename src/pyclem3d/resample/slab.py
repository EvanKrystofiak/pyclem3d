"""LM-driven view (plan §7c): for confocal slice k, the EM slab projection resampled into the LM grid.

The EM slices whose world z falls in the confocal slab (slice thickness or PSF FWHM) are
sampled along the LM z direction and projected (mean / min / max / Gaussian-weighted mean)
from the EM pyramid level that matches the LM xy resolution, so a 40-slice slab over a
6000^2 field is ~1e7 voxels, not 1e9. Results are cached per slice.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import dask
import dask.array as da
import numpy as np
from scipy.ndimage import map_coordinates

from ..io.metadata import psf_fallback
from ..io.volume import Volume, apply_affine
from ..register.transforms import Transform


@dataclass
class SlabParams:
    thickness: str | float = "slice"  # "slice" | "psf" | nm
    projection: str = "mean"  # mean | min | max | gaussian
    level: int | None = None  # EM pyramid level; None = match LM xy
    channel: int = 0
    max_samples: int = 64

    def key(self) -> tuple:
        return (self.thickness, self.projection, self.level, self.channel, self.max_samples)


def slab_thickness_nm(lm: Volume, thickness: str | float) -> float:
    if isinstance(thickness, (int, float)):
        return float(thickness)
    if thickness == "psf":
        return float(lm.psf_nm[0] if lm.psf_nm else psf_fallback(lm.voxel_size_nm)[0])
    return float(lm.voxel_size_nm[0])


def em_level_for_lm(em: Volume, lm: Volume, level: int | None = None) -> int:
    if level is not None:
        return int(level)
    return em.level_for_voxel_size(max(lm.voxel_size_nm[1], lm.voxel_size_nm[2]))


def _lm_z_direction(lm: Volume) -> np.ndarray:
    v = lm.world_affine[:3, 0]
    return v / np.linalg.norm(v)


def slab_offsets(
    thickness_nm: float, em_dz_nm: float, projection: str, max_samples: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Sample offsets (nm, along LM z) and weights across the slab."""
    n = int(np.clip(np.ceil(thickness_nm / max(em_dz_nm, 1e-6)) + 1, 1, max_samples))
    if n == 1:
        return np.zeros(1), np.ones(1)
    off = np.linspace(-thickness_nm / 2.0, thickness_nm / 2.0, n)
    if projection == "gaussian":
        sigma = thickness_nm / 2.355
        w = np.exp(-0.5 * (off / sigma) ** 2)
    else:
        w = np.ones(n)
    return off, w / w.sum()


def em_slab_for_lm_slice(
    em: Volume,
    lm: Volume,
    transform: Transform,
    k: int,
    params: SlabParams | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """EM projected into LM slice k's grid: (Y_lm, X_lm) float32, coverage (Y, X) uint8, info."""
    p = params or SlabParams()
    lvl = em_level_for_lm(em, lm, p.level)
    em_data = em.level_data(lvl)[p.channel]
    em_inv = np.linalg.inv(em.level_affine(lvl))
    em_dz = em.level_voxel_size_nm(lvl)[0]
    t_nm = slab_thickness_nm(lm, p.thickness)
    off, w = slab_offsets(t_nm, em_dz, p.projection, p.max_samples)
    zdir = _lm_z_direction(lm)
    Z, Y, X = lm.shape_zyx
    yy, xx = np.meshgrid(np.arange(Y, dtype=float), np.arange(X, dtype=float), indexing="ij")
    vox = np.stack([np.full(Y * X, float(k)), yy.ravel(), xx.ravel()], axis=1)
    base_world = apply_affine(lm.world_affine, vox)  # (N, 3) LM world
    EZ, EY, EX = (int(s) for s in em_data.shape)
    acc = None
    cov = np.zeros(Y * X, dtype=np.float32)
    wsum = np.zeros(Y * X, dtype=np.float32)
    # bounding box in EM voxel coords across all offsets (computed first, one read)
    all_vox = []
    for o in off:
        em_world = transform.apply(base_world + o * zdir)
        all_vox.append(em_world @ em_inv[:3, :3].T + em_inv[:3, 3])
    allv = np.concatenate(all_vox, axis=0)
    inside = (
        (allv[:, 0] >= -0.5)
        & (allv[:, 0] <= EZ - 0.5)
        & (allv[:, 1] >= -0.5)
        & (allv[:, 1] <= EY - 0.5)
        & (allv[:, 2] >= -0.5)
        & (allv[:, 2] <= EX - 0.5)
    )
    info = {
        "em_level": lvl,
        "thickness_nm": t_nm,
        "n_samples": len(off),
        "projection": p.projection,
    }
    if not inside.any():
        return np.zeros((Y, X), np.float32), np.zeros((Y, X), np.uint8), info
    lo = np.maximum(np.floor(allv[inside].min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(allv[inside].max(0)).astype(int) + 2, [EZ, EY, EX])
    sub = np.asarray(em_data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].compute()).astype(
        np.float32
    )
    info["em_z_range"] = [int(lo[0]), int(hi[0] - 1)]
    for vox_i, wi in zip(all_vox, w):
        vals = map_coordinates(sub, (vox_i - lo).T, order=1, mode="constant", cval=np.nan)
        ins = ~np.isnan(vals)
        cov += ins
        wsum += ins * wi
        if p.projection == "min":
            vals = np.where(ins, vals, np.inf)
            acc = vals if acc is None else np.minimum(acc, vals)
        elif p.projection == "max":
            vals = np.where(ins, vals, -np.inf)
            acc = vals if acc is None else np.maximum(acc, vals)
        else:
            vals = np.where(ins, vals, 0.0)
            acc = vals * wi if acc is None else acc + vals * wi
    assert acc is not None
    if p.projection in ("mean", "gaussian"):
        # renormalise where some samples fell outside the EM
        acc = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 0.0)
    coverage = (cov >= 0.5 * len(off)).astype(np.uint8)
    acc = np.where(np.isfinite(acc) & (cov > 0), acc, 0.0)
    return acc.reshape(Y, X).astype(np.float32), coverage.reshape(Y, X), info


class SlabCache:
    """LRU cache of EM slab projections keyed by (slice, params)."""

    def __init__(self, em: Volume, lm: Volume, transform: Transform, maxsize: int = 64):
        self.em, self.lm, self.transform = em, lm, transform
        self.maxsize = maxsize
        self._cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, dict]] = OrderedDict()

    def get(
        self, k: int, params: SlabParams | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        p = params or SlabParams()
        key = (int(k), p.key())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        val = em_slab_for_lm_slice(self.em, self.lm, self.transform, k, p)
        self._cache[key] = val
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return val

    def invalidate(self, transform: Transform | None = None) -> None:
        if transform is not None:
            self.transform = transform
        self._cache.clear()


def em_in_lm_stack(
    em: Volume, lm: Volume, transform: Transform, params: SlabParams | None = None
) -> tuple[da.Array, da.Array]:
    """Lazy (Z_lm, Y, X) EM-in-LM-space stack and coverage, one delayed task per LM slice (plan §9.3)."""
    p = params or SlabParams()
    Z, Y, X = lm.shape_zyx
    dtype = em.dtype

    @dask.delayed
    def _one(k: int):
        img, cov, _ = em_slab_for_lm_slice(em, lm, transform, k, p)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            img = np.clip(np.round(img), info.min, info.max)
        return img.astype(dtype)[None], cov[None]

    planes = [_one(k) for k in range(Z)]
    data = da.concatenate(
        [da.from_delayed(pl[0], shape=(1, Y, X), dtype=dtype) for pl in planes], axis=0
    )
    cov = da.concatenate(
        [da.from_delayed(pl[1], shape=(1, Y, X), dtype=np.uint8) for pl in planes], axis=0
    )
    return data, cov


def slab_brute_force(
    em: Volume, lm: Volume, transform: Transform, k: int, params: SlabParams | None = None
) -> np.ndarray:
    """Reference implementation for tests: same sampling, plain loops over samples (no bbox tricks)."""
    p = params or SlabParams()
    lvl = em_level_for_lm(em, lm, p.level)
    em_np = np.asarray(em.level_data(lvl)[p.channel].compute()).astype(np.float32)
    em_inv = np.linalg.inv(em.level_affine(lvl))
    t_nm = slab_thickness_nm(lm, p.thickness)
    off, w = slab_offsets(t_nm, em.level_voxel_size_nm(lvl)[0], p.projection, p.max_samples)
    zdir = _lm_z_direction(lm)
    Z, Y, X = lm.shape_zyx
    out = np.zeros((Y, X), np.float32)
    wsum = np.zeros((Y, X), np.float32)
    for y in range(Y):
        for x in range(X):
            wl = apply_affine(lm.world_affine, np.array([k, y, x], dtype=float))
            vals = []
            ws = []
            for o, wi in zip(off, w):
                v = apply_affine(em_inv, transform.apply(wl + o * zdir))
                val = map_coordinates(em_np, v[:, None], order=1, mode="constant", cval=np.nan)[0]
                if np.isfinite(val):
                    vals.append(val)
                    ws.append(wi)
            if not vals:
                continue
            vals = np.asarray(vals)
            if p.projection == "min":
                out[y, x] = vals.min()
            elif p.projection == "max":
                out[y, x] = vals.max()
            else:
                out[y, x] = float(np.sum(vals * np.asarray(ws)) / np.sum(ws))
            wsum[y, x] = 1
    return out
