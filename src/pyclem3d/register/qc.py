"""Honest verification (plan §6, §7d): residuals, RMS split into xy / z, LOO, round trip, z pattern."""

from __future__ import annotations

from typing import Any

import numpy as np

from .estimators import MIN_POINTS, fit_linear, fit_tps
from .transforms import AffineTransform, Transform


def residual_vectors(t: Transform, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """dst - T(src): the arrow from where the model puts a landmark to where it was picked (EM nm)."""
    return np.asarray(dst, dtype=float) - t.apply(np.asarray(src, dtype=float))


def rms_split(res: np.ndarray) -> dict[str, Any]:
    res = np.asarray(res, dtype=float).reshape(-1, 3)
    if len(res) == 0:
        return {
            "n": 0,
            "rms_total": None,
            "rms_xy": None,
            "rms_z": None,
            "max": None,
            "per_point": [],
        }
    per = np.linalg.norm(res, axis=1)
    return {
        "n": int(len(res)),
        "rms_total": float(np.sqrt(np.mean(per**2))),
        "rms_xy": float(np.sqrt(np.mean(res[:, 1] ** 2 + res[:, 2] ** 2))),
        "rms_z": float(np.sqrt(np.mean(res[:, 0] ** 2))),
        "mean_abs_z": float(np.mean(np.abs(res[:, 0]))),
        "max": float(per.max()),
        "argmax": int(per.argmax()),
        "per_point": [float(p) for p in per],
    }


def _fit(kind: str, src, dst, sigma, lam: float) -> Transform:
    if kind == "tps":
        return fit_tps(src, dst, sigma, lam)
    return AffineTransform(fit_linear(kind, src, dst, sigma), kind)


def loo_residuals(
    kind: str, src: np.ndarray, dst: np.ndarray, sigma: np.ndarray | None = None, lam: float = 0.0
) -> np.ndarray:
    """Leave-one-out residual vector of every landmark (NaN rows where too few remain)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    out = np.full((n, 3), np.nan)
    if n - 1 < MIN_POINTS[kind]:
        return out
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        try:
            t = _fit(kind, src[m], dst[m], None if sigma is None else sigma[m], lam)
        except (ValueError, np.linalg.LinAlgError):
            continue
        out[i] = dst[i] - t.apply(src[i : i + 1])[0]
    return out


def loo_summary(kind, src, dst, sigma=None, lam=0.0) -> dict[str, Any]:
    res = loo_residuals(kind, src, dst, sigma, lam)
    ok = ~np.isnan(res).any(axis=1)
    d = rms_split(res[ok]) if ok.any() else rms_split(np.zeros((0, 3)))
    d["n_evaluated"] = int(ok.sum())
    d["per_point_full"] = [None if not o else float(np.linalg.norm(r)) for o, r in zip(ok, res)]
    return d


def round_trip_error(fwd: Transform, inv: Transform, pts: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(pts, dtype=float)
    back = inv.apply(fwd.apply(pts))
    err = np.linalg.norm(back - pts, axis=1)
    return {
        "rms_nm": float(np.sqrt(np.mean(err**2))),
        "max_nm": float(err.max()),
        "n": int(len(pts)),
    }


def systematic_z_check(res: np.ndarray, src: np.ndarray) -> dict[str, Any]:
    """Does the z residual grow with position? A slope means a wrong z scale / tilt factor (plan §7d)."""
    res = np.asarray(res, dtype=float)
    src = np.asarray(src, dtype=float)
    out: dict[str, Any] = {}
    if len(res) < 4:
        return {"available": False}
    for name, col in (("z", 0), ("y", 1), ("x", 2)):
        x = src[:, col]
        y = res[:, 0]
        if np.ptp(x) < 1e-9:
            continue
        A = np.vstack([x - x.mean(), np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        out[f"dz_residual_per_{name}"] = float(coef[0])
        out[f"r2_{name}"] = float(1 - ss_res / ss_tot)
    slope = out.get("dz_residual_per_z", 0.0)
    r2 = out.get("r2_z", 0.0)
    out["available"] = True
    out["suspect_z_scale"] = bool(abs(slope) > 0.02 and r2 > 0.5)
    return out


def qc_report(
    t: Transform,
    kind: str,
    src: np.ndarray,
    dst: np.ndarray,
    sigma: np.ndarray | None,
    lam: float = 0.0,
    with_loo: bool = True,
) -> dict[str, Any]:
    res = residual_vectors(t, src, dst)
    rep: dict[str, Any] = {
        "kind": kind,
        "n": int(len(src)),
        "residuals_nm": res.tolist(),
        "rms": rms_split(res),
        "z_check": systematic_z_check(res, src),
    }
    if with_loo:
        rep["loo"] = loo_summary(kind, src, dst, sigma, lam)
    if sigma is not None and len(src):
        chi = res / np.asarray(sigma, dtype=float)
        rep["normalized_rms"] = float(np.sqrt(np.mean(chi**2)))
    return rep
