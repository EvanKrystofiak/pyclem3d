"""Refinement tests (plan §11): centroid refinement beats integer clicks; paired centroids agree;
midpoint sigma; xy snap recovers a perturbed EM point."""

from __future__ import annotations

import numpy as np
import pytest

from pyclem3d.io import ensure_pyramid
from pyclem3d.phantom import make_phantom
from pyclem3d.refine import local_centroid, midpoint_z, paired_centroids, snap_xy
from pyclem3d.register import AffineTransform


@pytest.fixture(scope="module")
def phantom():
    em, lm, truth = make_phantom(
        em_shape=(40, 160, 160),
        n_nuclei=4,
        n_organelles=10,
        seed=21,
        lm_voxel_nm=(240.0, 80.0, 80.0),
        noise=0.03,
    )
    em = ensure_pyramid(em, cache_dir=None, min_size=16)
    return em, lm, truth


def test_centroid_refinement_beats_integer_clicks(phantom):
    em, lm, truth = phantom
    rng = np.random.default_rng(1)
    errs_click, errs_ref = [], []
    for c_world, r in zip(truth.nuclei_em_world, truth.nuclei_radii_nm):
        true_vox = em.world_to_voxel(c_world)
        # a "click" somewhere inside the nucleus, off-centre by up to 40% of the radius
        click = true_vox + rng.uniform(-0.4, 0.4, 3) * (r / np.asarray(em.voxel_size_nm))
        click = np.round(click)
        res = local_centroid(em, click, box_nm=tuple(3.0 * r), channel=0, level=0, invert="auto")
        assert res.inverted  # nuclei are dark in EM
        errs_click.append(np.linalg.norm(em.voxel_to_world(click) - c_world))
        errs_ref.append(np.linalg.norm(res.centroid_world_nm - c_world))
        assert np.all(res.sigma_nm > 0)
        assert res.mask.sum() == res.n_voxels
    assert np.mean(errs_ref) < 0.4 * np.mean(errs_click)
    assert max(errs_ref) < 2.0 * max(em.voxel_size_nm)


def test_paired_centroids_are_consistent_across_modalities(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em)
    rng = np.random.default_rng(2)
    for c_em, c_lm, r in zip(truth.nuclei_em_world, truth.nuclei_lm_world, truth.nuclei_radii_nm):
        em_click = np.round(
            em.world_to_voxel(c_em) + rng.uniform(-0.3, 0.3, 3) * (r / np.asarray(em.voxel_size_nm))
        )
        lm_click = np.round(
            lm.world_to_voxel(c_lm) + rng.uniform(-0.3, 0.3, 3) * (r / np.asarray(lm.voxel_size_nm))
        )
        r_em, r_lm = paired_centroids(
            em, lm, em_click, lm_click, box_nm=tuple(3.0 * r), lm_channel=0
        )
        # LM centroid mapped through the true transform lands on the EM centroid
        mapped = T.apply(r_lm.centroid_world_nm)
        assert np.linalg.norm(mapped - r_em.centroid_world_nm) < 1.2 * max(lm.voxel_size_nm)
        assert np.linalg.norm(r_em.centroid_world_nm - c_em) < 2.0 * max(em.voxel_size_nm)
        assert r_em.method == "paired-centroid"
        assert r_lm.sigma_nm[0] >= r_lm.sigma_nm[1]  # LM z is the least certain coordinate
        d = r_em.to_dict()
        assert d["n_voxels"] > 0 and d["inverted"]


def test_midpoint_z():
    zc, sig = midpoint_z(10, 20, 300.0)
    assert zc == 15.0
    assert sig >= 300.0 / np.sqrt(2)
    zc2, sig2 = midpoint_z(10, 12, 300.0)
    assert sig2 < sig  # shorter extent, less uncertainty from the fade term


def test_snap_xy_recovers_perturbed_em_point(phantom):
    em, lm, truth = phantom
    T = AffineTransform(truth.lm_to_em)
    # choose a nucleus, correct LM position; the snap must return the EM position of T(lm)
    c_lm = truth.nuclei_lm_world[0]
    res = snap_xy(lm, em, T, c_lm, crop_nm=4000.0, search_nm=2000.0, lm_channel=0)
    assert np.abs(res.delta_lm_px).max() <= 1.5  # transform is exact: no nudge needed
    assert res.score > 0.3
    assert np.linalg.norm(res.em_world_nm_new - T.apply(c_lm)) <= 1.5 * max(lm.voxel_size_nm)
    # with a perturbed transform (xy translation error), the snap moves the EM point back
    M = truth.lm_to_em.copy()
    M[1, 3] += 600.0  # 600 nm error in y
    M[2, 3] -= 400.0
    Tbad = AffineTransform(M)
    res2 = snap_xy(lm, em, Tbad, c_lm, crop_nm=4000.0, search_nm=2000.0, lm_channel=0)
    before = np.linalg.norm((Tbad.apply(c_lm) - T.apply(c_lm))[1:])
    after = np.linalg.norm((res2.em_world_nm_new - T.apply(c_lm))[1:])
    assert after < 0.5 * before
