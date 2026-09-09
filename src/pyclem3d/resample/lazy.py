"""Lazy per-chunk resampling of a source Volume into any output Grid (plan §5, §7).

Nothing is materialised for the whole volume: each output chunk maps its voxel centres
through the inverse transform, reads only the source sub-block it needs, and runs
``scipy.ndimage.map_coordinates``. Coverage masks are computed the same way.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
from scipy.ndimage import map_coordinates

from ..io.volume import Volume, default_chunks
from ..register.transforms import Transform
from .grid import DisplacementGrid, Grid, InverseMapper


def _sample_block(
    src_data: da.Array,
    src_inv_affine: np.ndarray,
    src_world: np.ndarray,
    out_shape: tuple[int, int, int],
    order: int,
    dtype: np.dtype,
    channels: list[int],
    with_coverage: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample ``src_data`` (C,Z,Y,X) at world points -> (len(channels), *out_shape)."""
    vox = src_world @ src_inv_affine[:3, :3].T + src_inv_affine[:3, 3]  # (N, 3) src voxel coords
    Z, Y, X = (int(s) for s in src_data.shape[1:])
    out = np.zeros((len(channels), *out_shape), dtype=dtype)
    inside = (
        (vox[:, 0] >= -0.5)
        & (vox[:, 0] <= Z - 0.5)
        & (vox[:, 1] >= -0.5)
        & (vox[:, 1] <= Y - 0.5)
        & (vox[:, 2] >= -0.5)
        & (vox[:, 2] <= X - 0.5)
    )
    cov = inside.reshape(out_shape).astype(np.uint8) if with_coverage else None
    if not inside.any():
        return out, cov
    lo = np.maximum(np.floor(vox[inside].min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(vox[inside].max(0)).astype(int) + 2, [Z, Y, X])
    sub = np.asarray(src_data[channels, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].compute())
    coords = (vox - lo).T  # (3, N)
    for ci in range(len(channels)):
        vals = map_coordinates(
            sub[ci].astype(np.float32), coords, order=order, mode="constant", cval=0.0
        )
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            vals = np.clip(np.round(vals), info.min, info.max)
        out[ci] = vals.reshape(out_shape).astype(dtype)
    return out, cov


def resample_to_grid(
    src: Volume,
    transform: Transform,
    grid: Grid,
    level: int = 0,
    order: int = 1,
    channels: list[int] | None = None,
    chunks: tuple[int, int, int] | None = None,
    displacement: DisplacementGrid | None = None,
    dtype: np.dtype | None = None,
) -> tuple[da.Array, da.Array]:
    """Warp ``src`` (whose world -> ``grid`` world mapping is ``transform``) onto ``grid``.

    Returns (data (C, Z, Y, X), coverage (Z, Y, X) uint8), both lazy dask arrays.
    ``transform`` maps src world -> grid world (LM -> EM); the inverse is used per chunk.
    """
    inv = InverseMapper(transform, displacement)
    src_data = src.level_data(level)
    src_inv_affine = np.linalg.inv(src.level_affine(level))
    chans = list(range(src.n_channels)) if channels is None else list(channels)
    out_dtype = np.dtype(dtype) if dtype is not None else src_data.dtype
    shape = grid.shape_zyx
    if chunks is None:
        c = default_chunks((1, *shape), out_dtype.itemsize, target_bytes=8 << 20)
        chunks = (c[1], c[2], c[3])
    chunks = tuple(min(int(c), s) for c, s in zip(chunks, shape))
    template = da.zeros(shape, chunks=chunks, dtype=np.uint8)

    def _block(block, block_info=None):
        loc = block_info[0]["array-location"]  # the template (input) array: 3 axes
        (z0, z1), (y0, y1), (x0, x1) = loc
        world = grid.block_world_coords(z0, z1, y0, y1, x0, x1)
        src_world = inv(world)
        data, cov = _sample_block(
            src_data,
            src_inv_affine,
            src_world,
            (z1 - z0, y1 - y0, x1 - x0),
            order,
            out_dtype,
            chans,
            True,
        )
        return np.concatenate([data.astype(np.float32), cov[None].astype(np.float32)], axis=0)

    stacked = da.map_blocks(
        _block,
        template,
        dtype=np.float32,
        chunks=((len(chans) + 1,), template.chunks[0], template.chunks[1], template.chunks[2]),
        new_axis=0,
    )
    data = stacked[: len(chans)].astype(out_dtype)
    coverage = stacked[len(chans)].astype(np.uint8)
    return data, coverage


def resample_plane(
    src: Volume,
    inverse_map,
    grid: Grid,
    z_index: int,
    level: int = 0,
    order: int = 1,
    channels: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One output plane (C, Y, X) + coverage (Y, X) computed eagerly (viewer use)."""
    src_data = src.level_data(level)
    src_inv_affine = np.linalg.inv(src.level_affine(level))
    chans = list(range(src.n_channels)) if channels is None else list(channels)
    Z, Y, X = grid.shape_zyx
    world = grid.block_world_coords(z_index, z_index + 1, 0, Y, 0, X)
    src_world = inverse_map(world)
    data, cov = _sample_block(
        src_data, src_inv_affine, src_world, (1, Y, X), order, src_data.dtype, chans, True
    )
    return data[:, 0], cov[0]  # type: ignore[index]
