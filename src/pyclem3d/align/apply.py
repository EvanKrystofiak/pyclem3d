"""Apply per-slice transforms lazily (while reading) or bake an aligned copy (plan §4).

Integer shifts are pure slicing/padding (lossless). Sub-pixel shifts and per-slice
rigid/affine matrices are opt-in and interpolate (order 1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
from scipy import ndimage as ndi

from ..io.volume import Volume, default_chunks
from .sidecar import SliceTransforms

log = logging.getLogger(__name__)


def place_shifted(
    img: np.ndarray, dy: int, dx: int, bbox: tuple[int, int, int, int], fill: float
) -> np.ndarray:
    """Put ``img`` moved by integer (dy, dx) into a canvas covering reference rows/cols ``bbox``."""
    y0, y1, x0, x1 = bbox
    H, W = img.shape
    out = np.full((y1 - y0, x1 - x0), fill, dtype=img.dtype)
    uy0, uy1 = max(dy, y0), min(dy + H, y1)
    ux0, ux1 = max(dx, x0), min(dx + W, x1)
    if uy1 > uy0 and ux1 > ux0:
        out[uy0 - y0 : uy1 - y0, ux0 - x0 : ux1 - x0] = img[
            uy0 - dy : uy1 - dy, ux0 - dx : ux1 - dx
        ]
    return out


def transform_slice(
    img: np.ndarray,
    shift: tuple[float, float],
    bbox: tuple[int, int, int, int],
    fill: float,
    subpixel: bool = False,
    matrix: np.ndarray | None = None,
) -> np.ndarray:
    """One slice into the reference frame canvas (integer path, sub-pixel path, or matrix path)."""
    if matrix is not None:
        y0, y1, x0, x1 = bbox
        Minv = np.linalg.inv(matrix)
        # output canvas pixel (u, v) = reference (u + y0, v + x0); input = Minv @ reference
        off = Minv[:2, :2] @ np.array([y0, x0], dtype=float) + Minv[:2, 2]
        return ndi.affine_transform(
            img,
            Minv[:2, :2],
            offset=off,
            output_shape=(y1 - y0, x1 - x0),
            order=1,
            mode="constant",
            cval=fill,
        ).astype(img.dtype, copy=False)
    dy, dx = float(shift[0]), float(shift[1])
    if not subpixel:
        return place_shifted(img, int(round(dy)), int(round(dx)), bbox, fill)
    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    canvas = place_shifted(img, iy, ix, bbox, fill)
    if abs(fy) < 1e-6 and abs(fx) < 1e-6:
        return canvas
    out = ndi.shift(canvas.astype(np.float32), (fy, fx), order=1, mode="constant", cval=fill)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(img.dtype, copy=False)


def _fill_value(volume: Volume) -> float:
    if np.issubdtype(volume.dtype, np.integer):
        return 0.0
    return 0.0


def apply_lazy(
    volume: Volume,
    st: SliceTransforms,
    crop: str = "common",
    subpixel: bool | None = None,
    bad_slices: str = "keep",
    fill: float | None = None,
) -> Volume:
    """Return a Volume that applies ``st`` while reading. Nothing is written.

    ``crop``: "common" (area covered by every slice), "union", or "same" (original canvas).
    ``bad_slices``: "keep", "interpolate" (replace excluded slices by the mean of their
    neighbours), or "drop" (remove them from the stack).
    """
    C, Z, H, W = (int(s) for s in volume.data.shape)
    if st.n != Z:
        raise ValueError(f"sidecar has {st.n} slices, volume has {Z}")
    sub = st.subpixel if subpixel is None else bool(subpixel)
    if crop == "common":
        bbox = st.common_bbox((H, W))
    elif crop == "union":
        bbox = st.union_bbox((H, W))
    else:
        bbox = (0, H, 0, W)
    y0, y1, x0, x1 = bbox
    if y1 <= y0 or x1 <= x0:
        raise ValueError("no common area after alignment; try crop='union'")
    fillv = _fill_value(volume) if fill is None else float(fill)
    shifts = st.effective_shifts() if sub else st.integer_shifts().astype(float)
    mats = st.matrices
    data = volume.data.rechunk({2: -1, 3: -1})

    def _block(block, block_info=None):
        loc = block_info[None]["array-location"]
        z_start = loc[1][0]
        out = np.empty((block.shape[0], block.shape[1], y1 - y0, x1 - x0), dtype=block.dtype)
        for k in range(block.shape[1]):
            z = z_start + k
            M = None if mats is None else mats[z]
            for c in range(block.shape[0]):
                out[c, k] = transform_slice(block[c, k], tuple(shifts[z]), bbox, fillv, sub, M)
        return out

    chunks = (data.chunks[0], data.chunks[1], (y1 - y0,), (x1 - x0))
    out = da.map_blocks(_block, data, dtype=data.dtype, chunks=chunks)

    excluded = list(st.excluded)
    if excluded and bad_slices == "drop":
        keep = [z for z in range(Z) if z not in set(excluded)]
        out = out[:, keep]
    elif excluded and bad_slices == "interpolate":
        pieces = []
        for z in range(Z):
            if z in set(excluded):
                lo = max(z - 1, 0)
                hi = min(z + 1, Z - 1)
                while lo in set(excluded) and lo > 0:
                    lo -= 1
                while hi in set(excluded) and hi < Z - 1:
                    hi += 1
                rep = (
                    (
                        out[:, lo : lo + 1].astype(np.float32)
                        + out[:, hi : hi + 1].astype(np.float32)
                    )
                    / 2
                ).astype(out.dtype)
                pieces.append(rep)
            else:
                pieces.append(out[:, z : z + 1])
        out = da.concatenate(pieces, axis=1)

    T = np.eye(4)
    T[1, 3] = y0
    T[2, 3] = x0
    affine = volume.world_affine @ T
    meta = dict(volume.metadata)
    meta["alignment"] = {
        "crop": crop,
        "bbox_yx": [y0, y1, x0, x1],
        "subpixel": sub,
        "bad_slices": bad_slices,
        "n_excluded": len(excluded),
    }
    return volume.with_(
        data=out,
        world_affine=affine,
        per_slice_transforms=st,
        pyramid=None,
        pyramid_factors=None,
        metadata=meta,
        memory_strategy=volume.memory_strategy,
    )


def bake(
    volume: Volume,
    st: SliceTransforms,
    out_path: str | os.PathLike,
    fmt: str = "zarr",
    crop: str = "common",
    subpixel: bool | None = None,
    bad_slices: str = "keep",
    max_levels: int = 8,
    min_size: int = 64,
) -> Path:
    """Write the aligned stack (cropped to the common area) as OME-Zarr (with pyramid), OME-TIFF or MRC."""
    from ..io.pyramid import build_pyramid
    from ..io.writers import write_mrc, write_ome_tiff, write_ome_zarr

    aligned = apply_lazy(volume, st, crop=crop, subpixel=subpixel, bad_slices=bad_slices)
    out_path = Path(out_path)
    data = aligned.data.rechunk(default_chunks(aligned.data.shape, aligned.dtype.itemsize))
    if fmt in ("zarr", "ome-zarr"):
        levels, factors = build_pyramid(
            data, aligned.voxel_size_nm, max_levels=max_levels, min_size=min_size
        )
        vs = [tuple(v * f for v, f in zip(aligned.voxel_size_nm, fac)) for fac in factors]
        origin = tuple(float(v) for v in aligned.world_affine[:3, 3])
        write_ome_zarr(
            out_path,
            levels,
            vs,
            aligned.channels,
            name=Path(volume.source).name or "aligned",
            translation_nm=origin,  # type: ignore[arg-type]
            extra_attrs={"pyclem3d_align": st.to_dict()},
        )
    elif fmt in ("tif", "tiff", "ome-tiff"):
        write_ome_tiff(out_path, data, aligned.voxel_size_nm, aligned.channels)
    elif fmt == "mrc":
        write_mrc(out_path, data[0], aligned.voxel_size_nm)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    log.info("baked aligned stack to %s", out_path)
    return out_path


def reslice_views(volume: Volume, level: int | None = None, channel: int = 0) -> dict[str, Any]:
    """xz / yz orthogonal slices through the volume centre at a coarse level (QC, plan §4)."""
    lvl = level if level is not None else max(0, volume.n_levels() - 2)
    d = volume.level_data(lvl)[channel]
    Z, Y, X = d.shape
    xz = np.asarray(d[:, Y // 2, :].compute())
    yz = np.asarray(d[:, :, X // 2].compute())
    vs = volume.level_voxel_size_nm(lvl)
    return {
        "level": lvl,
        "xz": xz,
        "yz": yz,
        "voxel_size_nm": vs,
        "aspect_xz": vs[0] / vs[2],
        "aspect_yz": vs[0] / vs[1],
    }
