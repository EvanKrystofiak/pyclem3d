"""Turn an EM organelle mask into a synthetic fluorescence volume (plan Phase 6).

The mask is mean-downsampled to a voxel size near the LM's and blurred with the LM PSF, so
it looks like what the confocal would have recorded of those organelles. Registering this
against the real channel is a mono-modal intensity problem; segmentation errors below the
PSF scale wash out.
"""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np
from scipy.ndimage import gaussian_filter

from ..io.pyramid import coarsen
from ..io.volume import Channel, Volume

log = logging.getLogger(__name__)


def synthetic_fluorescence(
    mask: da.Array,
    em: Volume,
    target_voxel_nm: float = 32.0,
    psf_fwhm_nm: tuple[float, float] = (400.0, 160.0),
    z_range: tuple[int, int] | None = None,
    name: str = "synthetic",
) -> Volume:
    """(Z, Y, X) mask on the EM grid -> blurred density Volume in EM world coordinates.

    ``psf_fwhm_nm`` is (axial, lateral). ``z_range`` (EM slice indices) limits the output to
    the slab of interest; the world affine keeps the true position.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be (Z, Y, X)")
    vs = np.asarray(em.voxel_size_nm, dtype=float)
    factors = tuple(int(max(1, round(target_voxel_nm / v))) for v in vs)
    z0, z1 = (0, int(mask.shape[0])) if z_range is None else (int(z_range[0]), int(z_range[1]))
    z0 -= z0 % factors[0]
    sub = mask[z0:z1].astype(np.float32)
    coarse = coarsen(sub[None], factors)[0]  # (z', y', x') mean density in [0, 1]
    dens = np.asarray(coarse.compute(), dtype=np.float32)
    out_vs = vs * np.asarray(factors)
    sigma = np.array([psf_fwhm_nm[0], psf_fwhm_nm[1], psf_fwhm_nm[1]]) / 2.355 / out_vs
    blurred = gaussian_filter(dens, sigma).astype(np.float32)
    mx = float(blurred.max())
    if mx > 0:
        blurred /= mx
    # world affine: EM voxel (z0 + fz*k + (fz-1)/2, ...) is the centre of coarse voxel k
    L = np.eye(4)
    for i, f in enumerate(factors):
        L[i, i] = f
        L[i, 3] = (f - 1.0) / 2.0
    L[0, 3] += z0
    affine = em.world_affine @ L
    vol = Volume(
        data=da.from_array(blurred[None], chunks=(1, *blurred.shape)),
        voxel_size_nm=tuple(float(v) for v in out_vs),  # type: ignore[arg-type]
        world_affine=affine,
        kind="em",
        channels=[Channel(name, 0, "00FF00", 0.0, 1.0, 0.0, 1.0)],
        source=f"synthetic:{em.source}",
        metadata={"factors": list(factors), "psf_fwhm_nm": list(psf_fwhm_nm), "z_range": [z0, z1]},
        memory_strategy="ram",
    )
    log.info("synthetic fluorescence %s: shape %s voxel %s nm", name, blurred.shape, vol.voxel_size_nm)
    return vol
