"""Large-data smoke test (plan §11): a lazy volume far larger than RAM; assert that view, slab
projection, refinement and a cropped export never trigger a full load (chunk loads are
counted and peak RSS is tracked)."""

from __future__ import annotations

import threading

import dask
import dask.array as da
import numpy as np
import psutil
import pytest

from pyclem3d.export import export_fused_ome_zarr
from pyclem3d.io import Volume, apply_memory_strategy, ensure_pyramid, make_world_affine
from pyclem3d.io.readers import volume_from_array
from pyclem3d.refine import local_centroid
from pyclem3d.register import AffineTransform
from pyclem3d.resample import SlabParams, em_slab_for_lm_slice, lm_plane_at_em_slice, z_readout


class ChunkCounter:
    def __init__(self):
        self.n = 0
        self.lock = threading.Lock()

    def hit(self):
        with self.lock:
            self.n += 1


def virtual_em(
    shape=(2000, 4096, 4096), chunk=(1, 4096, 4096), counter: ChunkCounter | None = None
) -> tuple[da.Array, ChunkCounter]:
    """A 67 GB uint16 'EM stack' whose chunks are synthesised on demand and counted."""
    counter = counter or ChunkCounter()

    def make(block_id=None, **kw):
        counter.hit()
        z = block_id[0] if block_id else 0
        y, x = np.ogrid[: chunk[1], : chunk[2]]
        return ((y * 3 + x * 5 + z * 7) % 251).astype(np.uint16)[None]

    arr = da.map_blocks(
        make,
        chunks=(tuple([chunk[0]] * (shape[0] // chunk[0])), (chunk[1],), (chunk[2],)),
        dtype=np.uint16,
        meta=np.array((), dtype=np.uint16),
    )
    assert arr.shape == shape
    return arr, counter


@pytest.mark.slow
def test_views_and_export_never_load_the_whole_volume(tmp_path):
    arr, counter = virtual_em()
    total_chunks = arr.npartitions
    rss0 = psutil.Process().memory_info().rss
    em = Volume(
        data=arr[None],
        voxel_size_nm=(8.0, 8.0, 8.0),
        world_affine=make_world_affine((8.0, 8.0, 8.0)),
        kind="em",
        source="virtual",
    )
    assert em.nbytes > 60e9
    # 1) memory strategy: this cannot fit -> lazy, and deciding it touches nothing
    em = apply_memory_strategy(em)
    assert em.memory_strategy == "lazy"
    assert counter.n == 0
    # 2) pyramid attached lazily: nothing computed
    em = ensure_pyramid(em, cache_dir=tmp_path / "cache", persist_small=False)
    assert em.n_levels() >= 4 and counter.n == 0
    # a small confocal sitting in the middle of the block
    lm_np = np.random.default_rng(0).integers(0, 4000, (2, 12, 64, 64)).astype(np.uint16)
    lm = volume_from_array(lm_np, (300.0, 100.0, 100.0), kind="lm", source="lm")
    lm = apply_memory_strategy(lm)
    assert lm.memory_strategy == "ram"
    M = np.eye(4)
    M[:3, 3] = [4000.0, 12000.0, 12000.0]  # LM origin inside the EM block
    T = AffineTransform(M)
    # 3) EM-driven view at one EM slice (full-res plane of the LM) reads nothing from the EM
    plane, cov = lm_plane_at_em_slice(lm, em, T, 600, em_level=0)
    assert plane.shape[1:] == em.shape_zyx[1:]
    assert counter.n == 0
    assert "EM z 600" in z_readout(lm, em, T, em_j=600)
    # 4) LM-driven slab for one LM slice reads only the EM chunks under the LM footprint
    before = counter.n
    img, c, info = em_slab_for_lm_slice(em, lm, T, 5, SlabParams(thickness="slice", level=0))
    used = counter.n - before
    assert img.shape == (64, 64)
    assert 0 < used <= 45  # ~300 nm / 8 nm slab + 2 guard slices, one chunk per slice
    # 5) centroid refinement at full resolution reads a handful of chunks
    before = counter.n
    local_centroid(
        em,
        np.array([700.0, 2000.0, 2000.0]),
        box_nm=(160.0, 800.0, 800.0),
        level=0,
        invert=False,
        threshold=100.0,
    )
    assert 0 < counter.n - before <= 25
    # 6) a cropped full-res fused export touches only the ROI's slices
    before = counter.n
    roi = np.array([[4800.0, 13000.0, 13000.0], [5000.0, 13800.0, 13800.0]])
    export_fused_ome_zarr(
        em,
        lm,
        T,
        tmp_path / "roi.zarr",
        em_level=0,
        roi_world=roi,
        include_coverage=False,
        min_size=8,
    )
    used = counter.n - before
    assert 0 < used <= 40
    assert counter.n < 0.1 * total_chunks  # whole-slice chunks: a slab must read its ~40 slices
    rss1 = psutil.Process().memory_info().rss
    assert (rss1 - rss0) < 2.0e9  # the 67 GB volume never came close to being materialised


def test_lazy_volume_repr_and_sizes():
    arr, counter = virtual_em(shape=(100, 1024, 1024), chunk=(1, 1024, 1024))
    em = Volume(
        data=arr[None],
        voxel_size_nm=(10.0, 10.0, 10.0),
        world_affine=make_world_affine((10.0, 10.0, 10.0)),
        kind="em",
    )
    d = em.describe()
    assert "lazy" not in d or "undecided" in d
    assert counter.n == 0
    with dask.config.set(scheduler="synchronous"):
        assert int(em.data[0, 3, 0, 0].compute()) == (3 * 7) % 251
    assert counter.n == 1
