"""Registration core tests on synthetic 3D point sets with known truth (plan §11)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyclem3d.io import volume_from_array
from pyclem3d.register import (
    AffineTransform,
    LandmarkSet,
    PreAlign3D,
    TPSTransform,
    default_sigma_nm,
    fit_landmarks,
    fit_tps,
    fit_transform,
    loo_summary,
    prealign_transform,
    remap_landmarks,
    transform_from_dict,
)


def _points(n=20, seed=0, box=(30_000.0, 150_000.0, 150_000.0)):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, 3)) * np.asarray(box)


def _truth(kind: str, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = Rotation.from_euler(
        "zyx", [rng.uniform(-20, 20), rng.uniform(-3, 3), rng.uniform(-3, 3)], degrees=True
    ).as_matrix()
    t = rng.uniform(-5000, 5000, 3)
    M = np.eye(4)
    if kind == "rigid":
        M[:3, :3] = R
    elif kind == "similarity":
        M[:3, :3] = 0.85 * R
    else:
        S = np.diag([0.75, 0.88, 0.9])  # anisotropic shrinkage + z scaling
        shear = np.eye(3)
        shear[1, 2] = 0.03
        M[:3, :3] = R @ S @ shear
    M[:3, 3] = t
    return M


def _apply(M, p):
    return p @ M[:3, :3].T + M[:3, 3]


@pytest.mark.parametrize("kind", ["rigid", "similarity", "affine"])
def test_linear_estimators_recover_truth(kind):
    src = _points(25, seed=1)
    M = _truth(kind, seed=2)
    dst = _apply(M, src)
    fr = fit_transform(kind, src, dst, lam=None, with_loo=False)
    assert isinstance(fr.transform, AffineTransform)
    assert np.allclose(fr.transform.matrix, M, atol=1e-6)
    assert fr.rms["rms_total"] < 1e-6


@pytest.mark.parametrize("kind", ["rigid", "similarity", "affine"])
def test_linear_estimators_with_anisotropic_noise(kind):
    rng = np.random.default_rng(3)
    src = _points(30, seed=4)
    M = _truth(kind, seed=5)
    sigma = np.tile([400.0, 50.0, 50.0], (len(src), 1))  # z is the least certain coordinate
    dst = _apply(M, src) + rng.normal(0, 1, src.shape) * sigma
    fr = fit_transform(kind, src, dst, sigma=sigma, lam=None)
    # recovered points are within the noise, LOO too; z error dominates
    assert fr.rms["rms_xy"] < 120
    assert fr.rms["rms_z"] < 700
    assert fr.loo["rms_total"] < 900
    assert fr.rms["rms_z"] > fr.rms["rms_xy"]


@pytest.mark.parametrize("kind", ["rigid", "similarity"])
def test_anisotropic_weighting_improves_linear_fits_on_average(kind):
    """Per-axis sigma weighting is a statistical claim: check it over many noise draws."""
    test = _points(200, seed=999)
    err_w, err_u = [], []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        src = _points(30, seed=seed)
        M = _truth(kind, seed=seed + 1)
        sigma = np.tile([400.0, 50.0, 50.0], (len(src), 1))
        dst = _apply(M, src) + rng.normal(0, 1, src.shape) * sigma
        Mw = fit_transform(kind, src, dst, sigma=sigma, lam=None, with_loo=False).transform.matrix
        Mu = fit_transform(kind, src, dst, sigma=None, lam=None, with_loo=False).transform.matrix
        err_w.append(np.linalg.norm(_apply(Mw, test) - _apply(M, test), axis=1).mean())
        err_u.append(np.linalg.norm(_apply(Mu, test) - _apply(M, test), axis=1).mean())
    assert np.mean(err_w) < 0.85 * np.mean(err_u)


def test_min_points_and_coplanar_checks():
    src = _points(3)
    with pytest.raises(ValueError):
        fit_transform("affine", src, src)
    flat = _points(8)
    flat[:, 0] = 5000.0  # coplanar in z
    with pytest.raises(ValueError, match="coplanar"):
        fit_transform("affine", flat, flat)
    fit_transform("rigid", src, src, with_loo=False)  # 3 is enough for rigid


def test_tps_interpolates_exactly_and_regularization_helps_under_z_noise():
    src = _points(24, seed=7)
    M = _truth("affine", seed=8)

    def strong_warp(p):
        return _apply(M, p) + np.stack(
            [
                1500 * np.sin(p[:, 1] / 40_000),
                800 * np.cos(p[:, 2] / 50_000),
                600 * np.sin(p[:, 0] / 15_000),
            ],
            axis=1,
        )

    t0 = fit_tps(src, strong_warp(src), lam=0.0)
    assert np.allclose(
        t0.apply(src), strong_warp(src), atol=1e-3
    )  # exact interpolation at lambda 0

    # gentle, smooth deformation + confocal-scale z noise: the regularized spline must not chase the noise
    def warp(p):
        return _apply(M, p) + np.stack(
            [
                400 * np.sin(p[:, 1] / 100_000),
                200 * np.cos(p[:, 2] / 120_000),
                200 * np.sin(p[:, 0] / 40_000),
            ],
            axis=1,
        )

    rng = np.random.default_rng(21)
    clean = warp(src)
    sigma = np.tile([500.0, 60.0, 60.0], (len(src), 1))
    noisy = clean + rng.normal(0, 1, src.shape) * sigma
    loo0 = loo_summary("tps", src, noisy, sigma, 0.0)["rms_total"]
    fr = fit_transform("tps", src, noisy, sigma=sigma, lam="auto")
    assert fr.lam is not None and fr.lam > 0 and fr.lambda_search is not None
    assert fr.loo["rms_total"] < loo0
    assert isinstance(fr.transform, TPSTransform)
    test = _points(200, seed=10)
    truth = warp(test)
    err_reg = np.linalg.norm(fr.transform.apply(test) - truth, axis=1).mean()
    err_0 = np.linalg.norm(fit_tps(src, noisy, sigma, 0.0).apply(test) - truth, axis=1).mean()
    assert err_reg < 0.8 * err_0
    # the lambda sweep is well behaved: finite everywhere (no singular system)
    assert all(np.isfinite(v) for _, v in fr.lambda_search)
    # inverse round trip is small relative to the box
    assert fr.round_trip is not None and fr.round_trip["rms_nm"] < 200


def test_transform_serialization_roundtrip():
    src = _points(12, seed=11)
    dst = _apply(_truth("affine", seed=12), src)
    fr = fit_transform("affine", src, dst, lam=None, with_loo=False)
    back = transform_from_dict(fr.transform.to_dict())
    assert np.allclose(back.apply(src), dst, atol=1e-6)
    t = fit_tps(src, dst + 100, lam=1.0, fit_inverse=True)
    t2 = transform_from_dict(t.to_dict())
    assert np.allclose(t2.apply(src), t.apply(src))
    assert np.allclose(t2.inverse().apply(dst), t.inverse().apply(dst))


def test_prealign_remap_pins_points():
    lm = volume_from_array(np.zeros((1, 10, 20, 30), np.uint8), (300.0, 100.0, 100.0), kind="lm")
    raw = _points(8, seed=13, box=(2700, 1900, 2900))
    old = PreAlign3D(permutation=(0, 2, 1), flips=(False, True, False), rotation_z_deg=15.0)
    new = PreAlign3D(rotation_z_deg=-30.0)
    T_old = prealign_transform(old, lm)
    T_new = prealign_transform(new, lm)
    ls = LandmarkSet()
    for p in raw:
        ls.add(em_world_nm=(0, 0, 0), lm_world_nm=tuple(T_old.apply(p)), sigma_nm=(1, 1, 1))
    remapped = remap_landmarks(ls, old, new, lm)
    got = np.array([l.lm_world_nm for l in remapped])
    assert np.allclose(got, T_new.apply(raw), atol=1e-6)
    assert np.allclose([l.lm_world_nm for l in ls], T_old.apply(raw))  # original untouched
    assert remap_landmarks(ls, old, old, lm).to_dict() == ls.to_dict()
    # a permutation is a proper axis relabelling: |det| = 1, and identity is identity
    assert abs(np.linalg.det(old.linear())) == pytest.approx(1.0)
    assert PreAlign3D().is_identity()


def test_landmark_set_and_default_sigma():
    lm = volume_from_array(
        np.zeros((1, 5, 8, 8), np.uint8), (300.0, 100.0, 100.0), kind="lm", psf_nm=(800.0, 250.0)
    )
    em = volume_from_array(np.zeros((1, 5, 8, 8), np.uint8), (8.0, 8.0, 8.0), kind="em")
    s = default_sigma_nm(lm, em)
    assert s[1] == pytest.approx(np.hypot(50.0, 4.0)) and s[0] == pytest.approx(
        np.hypot(400.0, 4.0)
    )
    ls = LandmarkSet()
    src = _points(6, seed=14)
    M = _truth("rigid", seed=15)
    for p in src:
        ls.add(tuple(_apply(M, p[None])[0]), tuple(p), s, feature="nucleus")
    ls.landmarks[0].enabled = False
    assert ls.n_enabled == 5
    fr = fit_landmarks(ls, "rigid", lam=None)
    assert np.allclose(fr.transform.matrix, M, atol=1e-6)
    assert list(fr.ids) == [2, 3, 4, 5, 6]
    d = ls.to_dict()
    assert LandmarkSet.from_dict(d).to_dict() == d
    rep = ls.spread_report()
    assert rep["n"] == 5 and not rep["coplanar"]
