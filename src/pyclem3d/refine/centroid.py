"""Landmark refinement for endogenous features (plan §6): midpoint, local 3D centroid, paired.

Every refinement returns what it did (crop, mask, centroid, sigma) so the user can reject it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from ..io.volume import Volume


@dataclass
class CentroidResult:
    centroid_world_nm: np.ndarray  # (3,)
    centroid_voxel: np.ndarray  # (3,) level-0 voxel coords
    sigma_nm: np.ndarray  # (3,)
    bbox_voxel: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    crop: np.ndarray  # (z, y, x) float32 (possibly inverted)
    mask: np.ndarray  # (z, y, x) bool
    n_voxels: int
    threshold: float
    inverted: bool
    method: str = "centroid"
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid_world_nm": self.centroid_world_nm.tolist(),
            "centroid_voxel": self.centroid_voxel.tolist(),
            "sigma_nm": self.sigma_nm.tolist(),
            "bbox_voxel": [list(b) for b in self.bbox_voxel],
            "n_voxels": int(self.n_voxels),
            "threshold": float(self.threshold),
            "inverted": bool(self.inverted),
            "method": self.method,
            "info": self.info,
        }


def midpoint_z(z_first: float, z_last: float, dz_nm: float) -> tuple[float, float]:
    """Top/bottom midpoint (plan §6): centre slice and sigma_z (nm) from the visible extent.

    sigma_z combines the half-slice uncertainty of each end (dz / sqrt(2) in total) with a
    10 % of extent term for features whose ends fade rather than cut off.
    """
    zc = 0.5 * (float(z_first) + float(z_last))
    extent = abs(float(z_last) - float(z_first) + 1.0) * dz_nm
    sigma = float(np.hypot(dz_nm / np.sqrt(2.0), 0.1 * extent))
    return zc, sigma


def _box_voxels(vol: Volume, box_nm: float | tuple[float, float, float], level: int) -> np.ndarray:
    vs = np.asarray(vol.level_voxel_size_nm(level), dtype=float)
    b = np.broadcast_to(np.asarray(box_nm, dtype=float), (3,))
    return np.maximum(np.ceil(b / vs / 2.0).astype(int), 1)  # half-widths


def local_centroid(
    vol: Volume,
    center_voxel: np.ndarray,
    box_nm: float | tuple[float, float, float] = 6000.0,
    channel: int = 0,
    level: int = 0,
    invert: bool | str = "auto",
    threshold: float | str = "otsu",
    smooth_sigma_nm: float = 0.0,
    min_fraction: float = 0.002,
) -> CentroidResult:
    """Intensity-weighted centroid of the largest connected component near ``center_voxel``.

    ``center_voxel`` is in level-0 voxel coordinates. EM contrast is inverted when
    ``invert`` is True or ("auto") when the centre is darker than the crop mean (dark
    nucleus / organelle in EM). Works at any pyramid level; the crop is small so full
    resolution is cheap.
    """
    data = vol.level_data(level)[channel]
    Z, Y, X = (int(s) for s in data.shape)
    c0 = np.asarray(center_voxel, dtype=float)
    # to the requested level
    A0 = vol.level_affine(0)
    Al = vol.level_affine(level)
    c = np.linalg.inv(Al)[:3, :3] @ (A0[:3, :3] @ c0 + A0[:3, 3] - Al[:3, 3])
    half = _box_voxels(vol, box_nm, level)
    lo = np.maximum(np.round(c - half).astype(int), 0)
    hi = np.minimum(np.round(c + half).astype(int) + 1, [Z, Y, X])
    if (hi <= lo).any():
        raise ValueError("centroid box is outside the volume")
    crop = np.asarray(data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].compute()).astype(
        np.float32
    )
    if smooth_sigma_nm > 0:
        sig = smooth_sigma_nm / np.asarray(vol.level_voxel_size_nm(level))
        crop = ndi.gaussian_filter(crop, sig)
    cc = np.round(c - lo).astype(int)
    cc = np.clip(cc, 0, np.array(crop.shape) - 1)
    if invert == "auto":
        centre_val = float(
            np.mean(
                crop[
                    max(cc[0] - 1, 0) : cc[0] + 2,
                    max(cc[1] - 1, 0) : cc[1] + 2,
                    max(cc[2] - 1, 0) : cc[2] + 2,
                ]
            )
        )
        inverted = centre_val < float(crop.mean())
    else:
        inverted = bool(invert)
    work = crop.max() - crop if inverted else crop.copy()
    if threshold == "otsu":
        try:
            thr = float(threshold_otsu(work))
        except ValueError:
            thr = float(work.mean())
    else:
        thr = float(threshold)
    mask = work > thr
    if mask.sum() < min_fraction * mask.size:
        raise ValueError("segmentation found (almost) nothing above threshold in the box")
    lab, n = ndi.label(mask)
    if n == 0:
        raise ValueError("no connected component")
    target = lab[cc[0], cc[1], cc[2]]
    if target == 0:
        # nearest labelled voxel to the click
        dist = ndi.distance_transform_edt(lab == 0, return_indices=True)
        idx = dist[1]
        target = lab[
            idx[0][cc[0], cc[1], cc[2]], idx[1][cc[0], cc[1], cc[2]], idx[2][cc[0], cc[1], cc[2]]
        ]
    comp = lab == target
    w = np.where(comp, work - thr, 0.0).astype(np.float64)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = comp.astype(np.float64)
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in crop.shape], indexing="ij")
    W = w.sum()
    cz, cy, cx = (np.sum(w * zz) / W, np.sum(w * yy) / W, np.sum(w * xx) / W)
    cen_level = np.array([cz, cy, cx]) + lo
    # standard error of the weighted centroid per axis, floored at half a voxel
    var = (
        np.array(
            [np.sum(w * (zz - cz) ** 2), np.sum(w * (yy - cy) ** 2), np.sum(w * (xx - cx) ** 2)]
        )
        / W
    )
    n_eff = W**2 / np.sum(w**2)
    vs = np.asarray(vol.level_voxel_size_nm(level), dtype=float)
    sigma = np.maximum(np.sqrt(var / max(n_eff, 1.0)) * vs, 0.5 * vs)
    world = Al[:3, :3] @ cen_level + Al[:3, 3]
    cen_l0 = np.linalg.inv(A0)[:3, :3] @ (world - A0[:3, 3])
    return CentroidResult(
        centroid_world_nm=world,
        centroid_voxel=cen_l0,
        sigma_nm=sigma,
        bbox_voxel=((int(lo[0]), int(hi[0])), (int(lo[1]), int(hi[1])), (int(lo[2]), int(hi[2]))),
        crop=crop,
        mask=comp,
        n_voxels=int(comp.sum()),
        threshold=thr,
        inverted=inverted,
        info={
            "level": level,
            "channel": channel,
            "n_components": int(n),
            "extent_nm": (np.ptp(np.array(comp.nonzero()), axis=1) * vs).tolist()
            if comp.any()
            else None,
        },
    )


def paired_centroids(
    em: Volume,
    lm: Volume,
    em_center_voxel: np.ndarray,
    lm_center_voxel: np.ndarray,
    box_nm: float | tuple[float, float, float] = 6000.0,
    em_channel: int = 0,
    lm_channel: int = 0,
    em_level: int | None = None,
    lm_level: int = 0,
    em_invert: bool | str = "auto",
    lm_invert: bool = False,
    threshold: float | str = "otsu",
) -> tuple[CentroidResult, CentroidResult]:
    """Run the same segmentation-centroid on both sides in one action (plan §6).

    The same box size in nm and the same threshold rule are used on both sides so the
    definition of "centre" is consistent between EM and LM; the EM side is computed at
    the pyramid level nearest the LM xy resolution by default (a 5 um nucleus does not
    need 8 nm voxels to find its centre; the crop stays small).
    """
    if em_level is None:
        em_level = em.level_for_voxel_size(max(lm.voxel_size_nm[1:]) / 2.0)
    r_em = local_centroid(em, em_center_voxel, box_nm, em_channel, em_level, em_invert, threshold)
    r_lm = local_centroid(lm, lm_center_voxel, box_nm, lm_channel, lm_level, lm_invert, threshold)
    r_em.method = r_lm.method = "paired-centroid"
    return r_em, r_lm
