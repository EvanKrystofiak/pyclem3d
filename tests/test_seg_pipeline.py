"""Segmentation-to-image registration on the phantom (plan Phase 6): an organelle mask on the EM
grid -> synthetic fluorescence -> z scan recovers the depth and sign -> affine registration
recovers the true LM->EM transform. Needs SimpleITK (seg/itk extra)."""

from __future__ import annotations

import numpy as np
import pytest

from pyclem3d.io import ensure_pyramid
from pyclem3d.io.volume import apply_affine
from pyclem3d.phantom import make_phantom
from pyclem3d.register import AffineTransform
from pyclem3d.seg.synthetic import synthetic_fluorescence

sitk = pytest.importorskip("SimpleITK")
from pyclem3d.seg.intensity import ncc_on_synthetic, register_affine, z_scan  # noqa: E402


@pytest.fixture(scope="module")
def phantom():
    em, lm, truth = make_phantom(
        em_shape=(128, 192, 192),
        n_nuclei=0,
        n_organelles=160,
        seed=41,
        lm_voxel_nm=(150.0, 50.0, 50.0),
        psf_fwhm_nm=(400.0, 160.0),
        noise=0.02,
    )
    em = ensure_pyramid(em, cache_dir=None, min_size=16)
    # a perfect "segmentation" of the organelles on the EM grid
    Z, Y, X = em.shape_zyx
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in (Z, Y, X)], indexing="ij")
    world = apply_affine(em.world_affine, np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1))
    mask = np.zeros(Z * Y * X, bool)
    for c, r in zip(truth.organelles_em_world, truth.organelle_radii_nm):
        mask |= np.linalg.norm(world - c, axis=1) <= r
    import dask.array as da

    return em, lm, truth, da.from_array(mask.reshape(Z, Y, X).astype(np.uint8), chunks=(1, Y, X))


def test_synthetic_matches_true_channel(phantom):
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    assert syn.voxel_size_nm == (40.0, 40.0, 40.0)
    T = AffineTransform(truth.lm_to_em)
    good = ncc_on_synthetic(syn, lm, 1, T)  # organelle channel
    # a 600 nm z error or a wrong z sign must score clearly worse
    M = T.matrix.copy()
    M[0, 3] += 600.0
    worse = ncc_on_synthetic(syn, lm, 1, AffineTransform(M))
    Mf = T.matrix.copy()
    Mf[0, :3] *= -1
    Mf[0, 3] = 2 * (em.bbox_world()[0][0] + em.bbox_world()[1][0]) / 2 - Mf[0, 3]
    flipped = ncc_on_synthetic(syn, lm, 1, AffineTransform(Mf))
    assert good > 0.6
    assert good > worse + 0.15
    assert good > flipped + 0.15


def test_z_scan_and_affine_recover_truth(phantom):
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    T = AffineTransform(truth.lm_to_em)
    # start from a perturbed transform: 400 nm off in z, 150 nm in y, 3% scale error
    M = T.matrix.copy()
    M[0, 3] += 400.0
    M[1, 3] -= 150.0
    M[:3, :3] *= 1.03
    T0 = AffineTransform(M)
    scan = z_scan(syn, lm, 1, T0, np.arange(-800.0, 801.0, 100.0))
    best = max(scan, key=lambda r: r["ncc"])
    assert abs(best["offset_nm"] + 400.0) <= 150.0  # the scan finds the injected z error
    Mb = M.copy()
    Mb[0, 3] += best["offset_nm"]
    res = register_affine(
        syn,
        lm,
        1,
        AffineTransform(Mb),
        shrink=(2, 1),
        sigmas=(1.0, 0.0),
        iterations=200,
        learning_rate=2.0,
        sampling=0.5,
    )
    assert res.metric_after <= res.metric_before
    # compare mappings of test points against the truth, split into z and xy
    rng = np.random.default_rng(0)
    pts = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (200, 3))
    d0 = T0.apply(pts) - T.apply(pts)
    d1 = res.lm_to_em.apply(pts) - T.apply(pts)
    assert np.linalg.norm(d0, axis=1).mean() > 400.0
    assert np.linalg.norm(d1, axis=1).mean() < 100.0  # nm: about two LM xy voxels
    assert np.abs(d1[:, 0]).mean() < 100.0  # z is recovered, not just xy
    assert np.linalg.norm(d1[:, 1:], axis=1).mean() < 60.0
