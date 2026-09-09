"""Resampling / slab / z-view tests on the phantom with known truth (plan §11)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from pyclem3d.io import ensure_pyramid
from pyclem3d.phantom import make_phantom
from pyclem3d.register import AffineTransform, fit_tps, fit_transform
from pyclem3d.resample import (
    DisplacementGrid,
    Grid,
    SlabCache,
    SlabParams,
    em_in_lm_stack,
    em_slab_for_lm_slice,
    lm_plane_at_em_slice,
    resample_to_grid,
    slab_brute_force,
    z_profile,
    z_readout,
)


@pytest.fixture(scope="module")
def phantom():
    em, lm, truth = make_phantom(
        em_shape=(32, 128, 128),
        n_nuclei=5,
        n_organelles=20,
        seed=11,
        lm_voxel_nm=(240.0, 80.0, 80.0),
    )
    em = ensure_pyramid(em, cache_dir=None, min_size=16)
    return em, lm, truth


def _whole_array_resample(lm, T, grid, channel=0):
    """Reference: map every output voxel through the inverse with the LM fully in memory."""
    Z, Y, X = grid.shape_zyx
    world = grid.block_world_coords(0, Z, 0, Y, 0, X)
    lm_world = T.inverse().apply(world)
    vox = lm.world_to_voxel(lm_world)
    arr = np.asarray(lm.data[channel].compute()).astype(np.float32)
    vals = map_coordinates(arr, vox.T, order=1, mode="constant", cval=0.0)
    return np.round(vals).reshape(Z, Y, X)


def test_chunked_resample_equals_whole_array(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em, "affine")
    grid = Grid.from_volume(em, level=1)
    data, cov = resample_to_grid(lm, T, grid, chunks=(5, 33, 40))
    got = np.asarray(data[0].compute()).astype(np.float32)
    ref = _whole_array_resample(lm, T, grid, 0)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1.0
    c = np.asarray(cov.compute())
    assert c.dtype == np.uint8 and c.max() == 1
    # the EM block sits inside the LM field, so coverage is complete
    assert c.mean() > 0.99


def test_resampled_lm_nuclei_land_on_em_nuclei(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em, "affine")
    grid = Grid.from_volume(em, level=0)
    data, _ = resample_to_grid(lm, T, grid, channels=[0])
    warped = np.asarray(data[0].compute()).astype(np.float32)
    em_np = np.asarray(em.data[0].compute()).astype(np.float32)
    # nuclei are bright in warped LM and dark in EM: strong negative correlation
    r = np.corrcoef(warped.ravel(), em_np.ravel())[0, 1]
    assert r < -0.2
    # warped nuclei channel is much brighter inside the true nuclei than outside
    inside = np.zeros(em.shape_zyx, bool)
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in em.shape_zyx], indexing="ij")
    world = em.voxel_to_world(np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1))
    for c, rad in zip(truth.nuclei_em_world, truth.nuclei_radii_nm):
        inside |= (np.sum(((world - c) / rad) ** 2, axis=1) <= 1.0).reshape(em.shape_zyx)
    assert warped[inside].mean() > 3.0 * warped[~inside].mean()
    # and the warped intensity peaks at the true nucleus centres
    for c in truth.nuclei_em_world[:3]:
        v = np.round(em.world_to_voxel(c)).astype(int)
        v = np.clip(v, 0, np.array(em.shape_zyx) - 1)
        assert warped[v[0], v[1], v[2]] > 2.0 * np.median(warped)


def test_slab_projection_matches_brute_force(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em, "affine")
    Zl = lm.shape_zyx[0]
    k = Zl // 2
    for proj in ("mean", "min", "gaussian"):
        p = SlabParams(thickness="slice", projection=proj, level=0)
        fast, cov, info = em_slab_for_lm_slice(em, lm, T, k, p)
        slow = slab_brute_force(em, lm, T, k, p)
        m = cov.astype(bool)
        assert m.any()
        assert np.abs(fast[m] - slow[m]).max() < 1e-3, proj
    assert info["n_samples"] >= 2  # 240 nm slab over 20 nm EM slices
    cache = SlabCache(em, lm, T)
    a, _, _ = cache.get(k)
    b, _, _ = cache.get(k)
    assert a is b


def test_em_in_lm_stack_is_lazy_and_correct(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em, "affine")
    p = SlabParams(thickness="slice", projection="mean")
    data, cov = em_in_lm_stack(em, lm, T, p)
    assert data.shape == (lm.shape_zyx[0], lm.shape_zyx[1], lm.shape_zyx[2])
    k = lm.shape_zyx[0] // 2
    plane = np.asarray(data[k].compute()).astype(np.float32)
    ref, _, _ = em_slab_for_lm_slice(em, lm, T, k, p)
    assert np.abs(plane - np.round(ref)).max() <= 1.0


def test_em_driven_view_and_z_readout(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em, "affine")
    j = em.shape_zyx[0] // 2
    plane, cov = lm_plane_at_em_slice(lm, em, T, j, em_level=0, channels=[0])
    assert plane.shape == (1, em.shape_zyx[1], em.shape_zyx[2])
    assert cov.mean() > 0.99
    # PSF-aware weighting is a smoothed version: same mean-ish, less variance
    psf_plane, _ = lm_plane_at_em_slice(lm, em, T, j, em_level=0, channels=[0], psf_aware=True)
    assert psf_plane.shape == plane.shape
    assert psf_plane.astype(float).std() <= plane.astype(float).std() * 1.05
    s = z_readout(lm, em, T, em_j=j)
    assert s.startswith(f"EM z {j}/") and "LM z" in s
    s2 = z_readout(lm, em, T, lm_k=lm.shape_zyx[0] // 2)
    assert "slices, centre" in s2
    # a 240 nm confocal slab is ~10 EM slices of 20 nm after shrinkage, but the plane is
    # oblique (the phantom tilts by a few degrees), so the readout spans more and says so
    import re

    n = int(re.search(r"\((\d+) slices", s2).group(1))
    assert 6 <= n <= 80
    assert "oblique" in s2
    # z profile through a true nucleus centre peaks near offset 0
    prof = z_profile(lm, em, T, truth.nuclei_em_world[0], channel=0)
    assert abs(prof["peak_offset_nm"]) <= 1.5 * lm.voxel_size_nm[0]


def test_displacement_grid_inverse_and_deformable_resample(phantom):
    em, lm, truth = phantom
    src = truth.nuclei_lm_world
    dst = truth.nuclei_em_world
    rng = np.random.default_rng(0)
    extra_src = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (12, 3))
    extra_dst = AffineTransform(truth.lm_to_em).apply(extra_src) + rng.normal(0, 30, (12, 3))
    S = np.vstack([src, extra_src])
    D = np.vstack([dst, extra_dst])
    fr = fit_transform("tps", S, D, lam=1.0)
    t = fr.transform
    dg = DisplacementGrid.build(t, em.bbox_world(), spacing_nm=800.0, margin_nm=1000.0)
    pts = rng.uniform(em.bbox_world()[0], em.bbox_world()[1], (50, 3))
    rt = dg.round_trip(t, pts)
    assert rt["max_nm"] < 20.0  # grid interpolation + Newton refinement
    grid = Grid.from_volume(em, level=1)
    data, cov = resample_to_grid(lm, t, grid, displacement=dg, channels=[0])
    arr = np.asarray(data[0].compute())
    assert arr.shape == grid.shape_zyx and arr.max() > 0
    # close to the affine result since the extra points carry only 30 nm of noise
    data_aff, _ = resample_to_grid(lm, AffineTransform(truth.lm_to_em), grid, channels=[0])
    a = np.asarray(data_aff[0].compute()).astype(float)
    assert np.corrcoef(arr.ravel().astype(float), a.ravel())[0, 1] > 0.97
    # an un-fitted TPS cannot be inverted without a grid
    with pytest.raises(ValueError):
        resample_to_grid(lm, fit_tps(S, D, lam=1.0), grid)
