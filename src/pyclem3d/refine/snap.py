"""Local xy snap (plan §6): after a coarse fit, normalized cross-correlation of a small LM crop
against the EM slab projection (in the LM grid) nudges the EM side of a landmark pair in xy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.feature import match_template

from ..io.volume import Volume, apply_affine
from ..register.transforms import Transform
from ..resample.slab import SlabParams, em_slab_for_lm_slice


@dataclass
class SnapResult:
    em_world_nm_new: np.ndarray
    delta_lm_px: np.ndarray  # (dy, dx) in LM pixels
    score: float
    lm_crop: np.ndarray
    em_slab_crop: np.ndarray
    ncc_map: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "em_world_nm_new": self.em_world_nm_new.tolist(),
            "delta_lm_px": self.delta_lm_px.tolist(),
            "score": float(self.score),
            "info": self.info,
        }


def snap_xy(
    lm: Volume,
    em: Volume,
    transform: Transform,
    lm_world_nm: np.ndarray,
    crop_nm: float = 4000.0,
    search_nm: float = 3000.0,
    lm_channel: int = 0,
    em_channel: int = 0,
    invert_em: bool | str = "auto",
    slab: SlabParams | None = None,
) -> SnapResult:
    """Return the EM world position matching the LM feature at ``lm_world_nm`` via NCC in xy.

    The LM crop (centred on the landmark, at its LM slice) is matched against the EM slab
    projection of the same LM slice over a search window. The matched offset ``delta`` (LM
    px) means the structure lies at ``lm + delta`` in the LM grid, so the new EM position is
    ``T(lm + delta)``. The LM coordinate itself is unchanged (it is the feature's position).
    """
    lm_vox = lm.world_to_voxel(np.asarray(lm_world_nm, dtype=float))
    k = int(round(lm_vox[0]))
    Z, Y, X = lm.shape_zyx
    k = int(np.clip(k, 0, Z - 1))
    vs = np.asarray(lm.voxel_size_nm[1:], dtype=float)
    half = np.maximum(np.round(crop_nm / vs / 2).astype(int), 2)
    search = np.maximum(np.round(search_nm / vs).astype(int), 1)
    cy, cx = int(round(lm_vox[1])), int(round(lm_vox[2]))
    y0, y1 = max(cy - half[0], 0), min(cy + half[0] + 1, Y)
    x0, x1 = max(cx - half[1], 0), min(cx + half[1] + 1, X)
    lm_crop = np.asarray(lm.data[lm_channel, k, y0:y1, x0:x1].compute()).astype(np.float32)
    p = slab or SlabParams(channel=em_channel)
    p.channel = em_channel
    em_img, cov, info = em_slab_for_lm_slice(em, lm, transform, k, p)
    if invert_em == "auto":
        inv = True  # fluorescence is bright, EM features usually dark
    else:
        inv = bool(invert_em)
    covered = cov.astype(bool)
    if not covered.any():
        raise ValueError("snap: the EM does not cover this LM slice")
    fill = float(em_img[covered].mean())
    em_filled = np.where(covered, em_img, fill)
    em_work = (em_filled.max() - em_filled) if inv else em_filled
    sy0, sy1 = max(y0 - search[0], 0), min(y1 + search[0], Y)
    sx0, sx1 = max(x0 - search[1], 0), min(x1 + search[1], X)
    em_crop = em_work[sy0:sy1, sx0:sx1]
    if (
        em_crop.shape[0] < lm_crop.shape[0]
        or em_crop.shape[1] < lm_crop.shape[1]
        or lm_crop.std() == 0
        or em_crop.std() == 0
    ):
        raise ValueError("snap: crops too small or featureless")
    ncc = match_template(em_crop, lm_crop, pad_input=False)
    iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    # top-left of the best match in search-crop coords -> centre position
    match_y = sy0 + iy + (y1 - y0) / 2.0 - 0.5
    match_x = sx0 + ix + (x1 - x0) / 2.0 - 0.5
    lm_centre_y = y0 + (y1 - y0) / 2.0 - 0.5
    lm_centre_x = x0 + (x1 - x0) / 2.0 - 0.5
    delta = np.array([match_y - lm_centre_y, match_x - lm_centre_x])
    new_lm_vox = lm_vox + np.array([0.0, delta[0], delta[1]])
    new_em_world = transform.apply(apply_affine(lm.world_affine, new_lm_vox))
    return SnapResult(
        em_world_nm_new=np.asarray(new_em_world, dtype=float),
        delta_lm_px=delta,
        score=float(ncc[iy, ix]),
        lm_crop=lm_crop,
        em_slab_crop=em_crop,
        ncc_map=ncc,
        info={"lm_slice": k, "search_px": search.tolist(), "slab": info, "inverted_em": inv},
    )
