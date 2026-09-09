"""Own numpy 3D estimators (plan §6): weighted Kabsch / Umeyama / affine WLS / regularized TPS.

All functions take ``src`` (N,3) LM world nm and ``dst`` (N,3) EM world nm, optional
``sigma`` (N,3) per-point per-axis uncertainty in nm, and return a 4x4 matrix
(or a :class:`TPSTransform`) mapping src -> dst.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .transforms import TPSTransform, tps_kernel

MIN_POINTS = {"rigid": 3, "similarity": 3, "affine": 4, "tps": 5}


def _check(src: np.ndarray, dst: np.ndarray, sigma: np.ndarray | None, kind: str):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be (N,3); got {src.shape} and {dst.shape}")
    n = len(src)
    if n < MIN_POINTS[kind]:
        raise ValueError(f"{kind} needs at least {MIN_POINTS[kind]} landmark pairs; got {n}")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape == (3,):
            sigma = np.tile(sigma, (n, 1))
        if sigma.shape != (n, 3):
            raise ValueError("sigma must be (N,3) or (3,)")
        if (sigma <= 0).any():
            raise ValueError("sigma must be positive")
    return src, dst, sigma


def scalar_weights(sigma: np.ndarray | None, n: int) -> np.ndarray:
    """Per-point scalar weights 1/mean(sigma^2) normalised to mean 1 (rigid/similarity)."""
    if sigma is None:
        return np.ones(n)
    w = 1.0 / np.mean(np.asarray(sigma, dtype=float) ** 2, axis=1)
    return w / w.mean()


def _umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray, with_scale: bool) -> np.ndarray:
    W = w / w.sum()
    cs = W @ src
    cd = W @ dst
    X = src - cs
    Y = dst - cd
    cov = (Y * W[:, None]).T @ X  # 3x3, sum_i W_i y_i x_i^T
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = U @ D @ Vt
    if with_scale:
        var_x = float(np.sum(W * np.sum(X * X, axis=1)))
        c = float(np.trace(D @ np.diag(S)) / var_x)
    else:
        c = 1.0
    M = np.eye(4)
    M[:3, :3] = c * R
    M[:3, 3] = cd - c * R @ cs
    return M


def _refine_anisotropic(
    src: np.ndarray, dst: np.ndarray, sigma: np.ndarray, M0: np.ndarray, with_scale: bool
) -> np.ndarray:
    """Gauss-Newton refinement with per-axis weights (rotation vector + translation [+ log scale])."""
    L = M0[:3, :3]
    s0 = float(np.cbrt(abs(np.linalg.det(L)))) if with_scale else 1.0
    R0 = L / s0
    rv0 = Rotation.from_matrix(R0).as_rotvec()
    t0 = M0[:3, 3]
    p0 = np.concatenate([rv0, t0, [np.log(s0)] if with_scale else []])
    inv_sigma = 1.0 / sigma

    def resid(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        s = np.exp(p[6]) if with_scale else 1.0
        pred = src @ (s * R).T + p[3:6]
        return ((pred - dst) * inv_sigma).ravel()

    res = least_squares(
        resid, p0, method="lm" if len(src) * 3 >= len(p0) else "trf", xtol=1e-12, ftol=1e-12
    )
    p = res.x
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    s = np.exp(p[6]) if with_scale else 1.0
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = p[3:6]
    return M


def fit_rigid(src, dst, sigma=None, anisotropic: bool = True) -> np.ndarray:
    """Weighted Kabsch (6 dof). With per-axis sigma, refined by weighted Gauss-Newton."""
    src, dst, sigma = _check(src, dst, sigma, "rigid")
    M = _umeyama(src, dst, scalar_weights(sigma, len(src)), with_scale=False)
    if sigma is not None and anisotropic and not np.allclose(sigma, sigma[:, :1]):
        M = _refine_anisotropic(src, dst, sigma, M, with_scale=False)
    return M


def fit_similarity(src, dst, sigma=None, anisotropic: bool = True) -> np.ndarray:
    """Weighted Umeyama (7 dof)."""
    src, dst, sigma = _check(src, dst, sigma, "similarity")
    M = _umeyama(src, dst, scalar_weights(sigma, len(src)), with_scale=True)
    if sigma is not None and anisotropic and not np.allclose(sigma, sigma[:, :1]):
        M = _refine_anisotropic(src, dst, sigma, M, with_scale=True)
    return M


def fit_affine(src, dst, sigma=None) -> np.ndarray:
    """Per-axis weighted least squares (12 dof); needs >= 4 non-coplanar points."""
    src, dst, sigma = _check(src, dst, sigma, "affine")
    n = len(src)
    X = np.hstack([src, np.ones((n, 1))])
    if np.linalg.matrix_rank(src - src.mean(0), tol=1e-6 * max(1.0, np.abs(src).max())) < 3:
        raise ValueError("affine needs non-coplanar landmarks (spread them in z)")
    M = np.eye(4)
    for a in range(3):
        w = np.ones(n) if sigma is None else 1.0 / sigma[:, a]
        beta, *_ = np.linalg.lstsq(X * w[:, None], dst[:, a] * w, rcond=None)
        M[a, :3] = beta[:3]
        M[a, 3] = beta[3]
    return M


def _tps_system(src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = tps_kernel(cdist(src, src))
    P = np.hstack([np.ones((len(src), 1)), src])
    return K, P


def fit_tps(src, dst, sigma=None, lam: float = 0.0, fit_inverse: bool = False) -> TPSTransform:
    """Regularized 3D TPS: solve [[K + diag(reg), P], [P^T, 0]] [w; a] = [dst_axis; 0] per axis.

    ``reg_i = lam * sigma_i^2 / sigma_ref`` (nm) with ``sigma_ref`` the median sigma, so
    lam is scale-free: lam = 1 relaxes a typically-uncertain point by about one sigma of
    bending; points with larger sigma (confocal z) are relaxed more (plan §6).
    Without sigma, ``reg_i = lam`` in nm.
    """
    src, dst, sigma = _check(src, dst, sigma, "tps")
    n = len(src)
    K, P = _tps_system(src)
    weights = np.zeros((n, 3))
    affine = np.zeros((4, 3))
    sig_ref = float(np.median(sigma)) if sigma is not None else 1.0
    for a in range(3):
        if sigma is not None:
            reg = lam * sigma[:, a] ** 2 / sig_ref
        else:
            reg = np.full(n, lam, dtype=float)
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = K + np.diag(reg)
        A[:n, n:] = P
        A[n:, :n] = P.T
        rhs = np.concatenate([dst[:, a], np.zeros(4)])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        weights[:, a] = sol[:n]
        affine[:, a] = sol[n:]
    t = TPSTransform(src, weights, affine, lam)
    if fit_inverse:
        t.inverse_model = fit_tps(dst, src, sigma, lam, fit_inverse=False)
    return t


def fit_linear(kind: str, src, dst, sigma=None) -> np.ndarray:
    if kind == "rigid":
        return fit_rigid(src, dst, sigma)
    if kind == "similarity":
        return fit_similarity(src, dst, sigma)
    if kind == "affine":
        return fit_affine(src, dst, sigma)
    raise ValueError(f"unknown linear kind {kind!r}")
