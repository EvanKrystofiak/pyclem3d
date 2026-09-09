"""EM-driven view (plan §7c, default): scrolling EM z shows the confocal resampled at that exact
world plane, trilinear in z so it varies smoothly across the ~40 EM slices per confocal slice.
Optional PSF-aware weighting of LM samples along the LM z axis. Plus the z readout string."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..io.metadata import psf_fallback
from ..io.volume import Volume, apply_affine
from ..register.transforms import Transform
from .grid import DisplacementGrid, Grid, InverseMapper
from .lazy import resample_plane
from .slab import _lm_z_direction


def lm_plane_at_em_slice(
    lm: Volume,
    em: Volume,
    transform: Transform,
    j: int,
    em_level: int = 0,
    lm_level: int = 0,
    channels: list[int] | None = None,
    psf_aware: bool = False,
    n_psf_samples: int = 5,
    displacement: DisplacementGrid | None = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """LM channels resampled onto EM slice j (at ``em_level``): (C, Y, X) and coverage (Y, X)."""
    inv = InverseMapper(transform, displacement)
    grid = Grid.from_volume(em, em_level)
    if not psf_aware:
        return resample_plane(lm, inv, grid, j, level=lm_level, order=order, channels=channels)
    fwhm_z = lm.psf_nm[0] if lm.psf_nm else psf_fallback(lm.voxel_size_nm)[0]
    sigma = fwhm_z / 2.355
    offs = np.linspace(-sigma, sigma, n_psf_samples)
    w = np.exp(-0.5 * (offs / sigma) ** 2)
    w /= w.sum()
    zdir = _lm_z_direction(lm)
    acc = None
    cov = None
    for o, wi in zip(offs, w):

        def shifted_inv(world, o=o):
            return inv(world) + o * zdir

        data, c = resample_plane(
            lm, shifted_inv, grid, j, level=lm_level, order=order, channels=channels
        )
        acc = data.astype(np.float32) * wi if acc is None else acc + data.astype(np.float32) * wi
        cov = c if cov is None else np.maximum(cov, c)
    assert acc is not None and cov is not None
    if np.issubdtype(lm.dtype, np.integer):
        info = np.iinfo(lm.dtype)
        acc = np.clip(np.round(acc), info.min, info.max)
    return acc.astype(lm.dtype), cov


def lm_z_at_em_slice(
    lm: Volume,
    em: Volume,
    transform: Transform,
    j: int,
    em_level: int = 0,
    displacement: DisplacementGrid | None = None,
) -> dict[str, Any]:
    """Which LM z (fractional slice) the centre of EM slice j maps to."""
    inv = InverseMapper(transform, displacement)
    Z, Y, X = em.level_data(em_level).shape[1:]
    centre = em.voxel_to_world(np.array([j, (Y - 1) / 2.0, (X - 1) / 2.0]), level=em_level)
    lm_world = inv(centre[None])[0]
    lm_vox = lm.world_to_voxel(lm_world)
    return {
        "em_z": int(j),
        "em_nz": int(Z),
        "lm_z": float(lm_vox[0]),
        "lm_nz": int(lm.shape_zyx[0]),
        "lm_voxel": lm_vox.tolist(),
        "inside": bool(-0.5 <= lm_vox[0] <= lm.shape_zyx[0] - 0.5),
    }


def em_z_range_for_lm_slice(
    lm: Volume,
    em: Volume,
    transform: Transform,
    k: int,
    thickness_nm: float | None = None,
    em_level: int = 0,
) -> dict[str, Any]:
    """EM slice index range covered by LM slice k's slab (thickness default: LM dz)."""
    t = float(thickness_nm) if thickness_nm is not None else float(lm.voxel_size_nm[0])
    Z, Y, X = lm.shape_zyx
    zdir = _lm_z_direction(lm)
    corners = np.array(
        [[k, 0, 0], [k, 0, X - 1], [k, Y - 1, 0], [k, Y - 1, X - 1], [k, (Y - 1) / 2, (X - 1) / 2]],
        dtype=float,
    )
    world = apply_affine(lm.world_affine, corners)
    pts = np.concatenate([world - t / 2 * zdir, world + t / 2 * zdir, world])
    em_vox = em.world_to_voxel(transform.apply(pts), level=em_level)
    zc = em.world_to_voxel(transform.apply(world[-1:]), level=em_level)[0, 0]
    z0, z1 = float(em_vox[:, 0].min()), float(em_vox[:, 0].max())
    EZ = int(em.level_data(em_level).shape[1])
    return {
        "lm_z": int(k),
        "lm_nz": int(Z),
        "em_z0": int(np.floor(z0)),
        "em_z1": int(np.ceil(z1)),
        "em_center": float(zc),
        "n_em_slices": int(np.ceil(z1) - np.floor(z0) + 1),
        "em_nz": EZ,
        "thickness_nm": t,
        "oblique_spread_slices": float(em_vox[-5:, 0].max() - em_vox[-5:, 0].min()),
    }


def z_readout(
    lm: Volume,
    em: Volume,
    transform: Transform,
    lm_k: int | None = None,
    em_j: int | None = None,
    em_level: int = 0,
    displacement: DisplacementGrid | None = None,
) -> str:
    """Readout such as ``LM z 12/48 <-> EM z 480-521 (42 slices, centre 500)`` (plan §7c)."""
    if lm_k is not None:
        r = em_z_range_for_lm_slice(lm, em, transform, lm_k, em_level=em_level)
        s = f"LM z {r['lm_z']}/{r['lm_nz'] - 1} <-> EM z {r['em_z0']}-{r['em_z1']} ({r['n_em_slices']} slices, centre {r['em_center']:.0f})"
        if r["oblique_spread_slices"] > 1.5:
            s += f" [plane is oblique: {r['oblique_spread_slices']:.0f} EM slices across the field]"
        return s
    if em_j is not None:
        r = lm_z_at_em_slice(lm, em, transform, em_j, em_level, displacement)
        flag = "" if r["inside"] else " [outside LM stack]"
        return f"EM z {r['em_z']}/{r['em_nz'] - 1} <-> LM z {r['lm_z']:.2f}/{r['lm_nz'] - 1}{flag}"
    raise ValueError("give lm_k or em_j")


def z_profile(
    lm: Volume,
    em: Volume,
    transform: Transform,
    em_world_point: np.ndarray,
    channel: int = 0,
    half_range_nm: float | None = None,
    n: int = 41,
    displacement: DisplacementGrid | None = None,
) -> dict[str, Any]:
    """LM intensity along LM z through a mapped EM feature (plan §7d 'z-profile check').

    Returns the profile and where the EM feature lands on it; a peak that is off the mapped
    z reveals a z error at that location.
    """
    from scipy.ndimage import map_coordinates

    inv = InverseMapper(transform, displacement)
    p_lm = inv(np.asarray(em_world_point, dtype=float)[None])[0]
    dz = lm.voxel_size_nm[0]
    hr = float(half_range_nm) if half_range_nm is not None else 4.0 * dz
    zdir = _lm_z_direction(lm)
    offs = np.linspace(-hr, hr, n)
    pts = p_lm[None] + offs[:, None] * zdir
    vox = lm.world_to_voxel(pts)
    sub, lo = _lm_subblock(lm, vox, channel)
    vals = map_coordinates(
        sub.astype(np.float32), (vox - lo).T, order=1, mode="constant", cval=np.nan
    )
    ok = ~np.isnan(vals)
    peak = float(offs[ok][np.nanargmax(vals[ok])]) if ok.any() else float("nan")
    return {
        "offset_nm": offs.tolist(),
        "intensity": [None if not o else float(v) for o, v in zip(ok, vals)],
        "peak_offset_nm": peak,
        "lm_voxel": vox[n // 2].tolist(),
    }


def _lm_subblock(lm: Volume, vox: np.ndarray, channel: int):
    Z, Y, X = lm.shape_zyx
    lo = np.maximum(np.floor(vox.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(vox.max(0)).astype(int) + 2, [Z, Y, X])
    sub = np.asarray(lm.data[channel, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].compute())
    return sub, lo
