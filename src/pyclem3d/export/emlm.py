"""Output 3 (plan §9): EM-in-LM-space - the EM slab-projected and resampled into the confocal grid,
together with the LM channels, as OME-TIFF or OME-Zarr (small; for LM-side analysis and figures)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import dask.array as da
import numpy as np

from ..io.volume import Channel, Volume
from ..io.writers import write_ome_tiff, write_ome_zarr_from_level0
from ..register.transforms import Transform
from ..resample.slab import SlabParams, em_in_lm_stack
from .fused import _common_dtype

log = logging.getLogger(__name__)


def export_em_in_lm(
    em: Volume,
    lm: Volume,
    transform: Transform,
    out_path: str | os.PathLike,
    fmt: str = "ome-tiff",
    slab: SlabParams | None = None,
    include_lm: bool = True,
    lm_channels: list[int] | None = None,
    include_coverage: bool = True,
) -> Path:
    p = slab or SlabParams()
    em_stack, cov = em_in_lm_stack(em, lm, transform, p)
    dtype = _common_dtype(em.dtype, lm.dtype) if include_lm else em.dtype
    Z, Y, X = lm.shape_zyx
    parts = [em_stack.astype(dtype)[None]]
    channels = [
        Channel(
            f"EM:{em.channels[p.channel].name} ({p.projection} slab {p.thickness})", 0, "FFFFFF"
        )
    ]
    if include_lm:
        chans = list(range(lm.n_channels)) if lm_channels is None else list(lm_channels)
        for c in chans:
            ch = lm.channels[c]
            parts.append(lm.data[c : c + 1].astype(dtype))
            channels.append(
                Channel(
                    f"LM:{ch.name}",
                    len(channels),
                    ch.color,
                    ch.display_min,
                    ch.display_max,
                    emission_nm=ch.emission_nm,
                )
            )
    if include_coverage:
        hi = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
        parts.append((cov.astype(dtype) * dtype.type(hi))[None])
        channels.append(Channel("EM coverage", len(channels), "808080", visible=False))
    parts = [q.rechunk((1, 1, Y, X)) for q in parts]
    stack = da.concatenate(parts, axis=0)
    out_path = Path(out_path)
    if fmt in ("ome-tiff", "tif", "tiff"):
        write_ome_tiff(out_path, stack, lm.voxel_size_nm, channels)
    elif fmt in ("ome-zarr", "zarr"):
        origin = tuple(float(v) for v in lm.world_affine[:3, 3])
        write_ome_zarr_from_level0(
            out_path, stack, lm.voxel_size_nm, channels, name="em_in_lm", translation_nm=origin
        )  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown format {fmt!r}")
    log.info("wrote EM-in-LM %s (%d channels)", out_path, len(channels))
    return out_path
