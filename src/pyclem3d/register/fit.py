"""High-level fit (plan §6): kind + landmarks (+ sigma) -> transform, QC, and lambda selection by LOO."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .estimators import MIN_POINTS, fit_linear, fit_tps
from .landmarks import LandmarkSet
from .qc import loo_summary, qc_report, residual_vectors, rms_split, round_trip_error
from .transforms import ALL_KINDS, AffineTransform, TPSTransform, Transform

log = logging.getLogger(__name__)

LAMBDA_GRID = (0.0,) + tuple(float(x) for x in np.logspace(-2, 3, 11))


@dataclass
class FitResult:
    transform: Transform
    kind: str
    n: int
    lam: float | None = None
    residuals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rms: dict[str, Any] = field(default_factory=dict)
    loo: dict[str, Any] | None = None
    round_trip: dict[str, Any] | None = None
    lambda_search: list[tuple[float, float]] | None = None
    warnings: list[str] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)
    ids: np.ndarray | None = None

    def summary(self) -> str:
        r = self.rms
        s = f"{self.kind}: n={self.n} RMS total {r.get('rms_total', float('nan')):.1f} nm (xy {r.get('rms_xy', float('nan')):.1f}, z {r.get('rms_z', float('nan')):.1f})"
        if self.loo and self.loo.get("rms_total") is not None:
            s += f"; LOO {self.loo['rms_total']:.1f} nm (xy {self.loo['rms_xy']:.1f}, z {self.loo['rms_z']:.1f})"
        if self.lam is not None:
            s += f"; lambda={self.lam:g}"
        if self.round_trip:
            s += f"; round-trip {self.round_trip['rms_nm']:.2f} nm"
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n": self.n,
            "lam": self.lam,
            "transform": self.transform.to_dict(),
            "rms": self.rms,
            "loo": self.loo,
            "round_trip": self.round_trip,
            "lambda_search": self.lambda_search,
            "warnings": list(self.warnings),
            "qc": self.qc,
            "ids": None if self.ids is None else [int(i) for i in self.ids],
            "residuals_nm": self.residuals.tolist(),
        }


def select_lambda(
    src: np.ndarray,
    dst: np.ndarray,
    sigma: np.ndarray | None,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> tuple[float, list[tuple[float, float]]]:
    """Pick the TPS lambda with the smallest leave-one-out RMS (plan §6)."""
    search = []
    best = (None, np.inf)
    for lam in grid:
        s = loo_summary("tps", src, dst, sigma, lam)
        v = s["rms_total"] if s["rms_total"] is not None else np.inf
        search.append((float(lam), float(v)))
        if v < best[1]:
            best = (lam, v)
    if best[0] is None:
        return float(grid[len(grid) // 2]), search
    return float(best[0]), search


def fit_transform(
    kind: str,
    src: np.ndarray,
    dst: np.ndarray,
    sigma: np.ndarray | None = None,
    lam: float | str | None = "auto",
    with_loo: bool = True,
    ids: np.ndarray | None = None,
) -> FitResult:
    """Fit ``kind`` in ("rigid", "similarity", "affine", "tps") from LM->EM world-nm pairs."""
    if kind not in ALL_KINDS:
        raise ValueError(f"kind must be one of {ALL_KINDS}; got {kind!r}")
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    warnings: list[str] = []
    if n < MIN_POINTS[kind]:
        raise ValueError(f"{kind} needs >= {MIN_POINTS[kind]} pairs, got {n}")
    if kind == "tps" and n < 10:
        warnings.append(f"TPS with only {n} landmarks: use >= 10 well spread in 3D (plan §6)")
    if n < 10:
        warnings.append(
            f"only {n} landmarks; accuracy comes from 10-30 well-spread endogenous landmarks (plan §3)"
        )
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape == (3,):
            sigma = np.tile(sigma, (n, 1))

    lam_used: float | None = None
    search = None
    if kind == "tps":
        if lam == "auto" or lam is None:
            lam_used, search = select_lambda(src, dst, sigma)
        else:
            lam_used = float(lam)
        t: Transform = fit_tps(src, dst, sigma, lam_used, fit_inverse=True)
    else:
        t = AffineTransform(fit_linear(kind, src, dst, sigma), kind)

    res = residual_vectors(t, src, dst)
    fr = FitResult(t, kind, n, lam_used, res, rms_split(res), ids=ids, warnings=warnings)
    fr.qc = qc_report(t, kind, src, dst, sigma, lam_used or 0.0, with_loo=with_loo)
    fr.loo = fr.qc.get("loo")
    fr.lambda_search = search
    if isinstance(t, TPSTransform) and t.inverse_model is not None:
        fr.round_trip = round_trip_error(t, t.inverse_model, src)
        if fr.round_trip["rms_nm"] > 0.25 * max(fr.rms["rms_total"], 1.0):
            warnings.append(
                "TPS inverse round-trip error is large relative to the fit RMS; the deformation may be too strong"
            )
    if isinstance(t, AffineTransform):
        dec = t.decompose()
        fr.qc["decomposition"] = dec
        if dec["reflection"]:
            warnings.append("fitted transform contains a reflection: check pre-align flips")
        if kind == "affine" and dec["anisotropy"] > 1.6:
            warnings.append(
                f"affine scale anisotropy {dec['anisotropy']:.2f}: plausible for shrinkage/z-scaling but verify the z scale"
            )
    if fr.qc.get("z_check", {}).get("suspect_z_scale"):
        warnings.append("z residual grows with z: suspect wrong z scale or tilt factor (plan §7d)")
    log.info(fr.summary())
    return fr


def fit_landmarks(
    landmarks: LandmarkSet, kind: str, lam: float | str | None = "auto", with_loo: bool = True
) -> FitResult:
    src, dst, sig, ids = landmarks.arrays()
    return fit_transform(kind, src, dst, sig, lam, with_loo, ids=ids)
