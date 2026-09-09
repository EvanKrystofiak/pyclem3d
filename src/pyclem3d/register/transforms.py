"""Transform objects. All map **LM world nm -> EM world nm** (moving -> fixed), plan §6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from ..io.volume import apply_affine

LINEAR_KINDS = ("rigid", "similarity", "affine")
ALL_KINDS = LINEAR_KINDS + ("tps",)


class Transform:
    kind: str = "identity"

    def apply(self, pts: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def inverse(self) -> Transform:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def is_linear(self) -> bool:
        return isinstance(self, AffineTransform)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Transform:
        if d.get("type") == "tps":
            return TPSTransform.from_dict(d)
        return AffineTransform.from_dict(d)


@dataclass
class AffineTransform(Transform):
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    kind: str = "affine"  # descriptive: rigid | similarity | affine

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4")

    @classmethod
    def identity(cls, kind: str = "rigid") -> AffineTransform:
        return cls(np.eye(4), kind)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return apply_affine(self.matrix, pts)

    def inverse(self) -> AffineTransform:
        return AffineTransform(np.linalg.inv(self.matrix), self.kind)

    def compose(self, other: AffineTransform) -> AffineTransform:
        """self ∘ other: apply ``other`` first, then ``self``."""
        return AffineTransform(self.matrix @ other.matrix, "affine")

    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def decompose(self) -> dict[str, Any]:
        """Scale factors (singular values), rotation angle, shear measure - for the report."""
        L = self.linear
        U, S, Vt = np.linalg.svd(L)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        return {
            "scales": [float(s) for s in S],
            "mean_scale": float(np.cbrt(abs(np.linalg.det(L)))),
            "rotation_deg": angle,
            "anisotropy": float(S.max() / S.min()) if S.min() > 0 else float("inf"),
            "translation_nm": [float(t) for t in self.translation],
            "reflection": bool(np.linalg.det(L) < 0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"type": "affine", "kind": self.kind, "matrix": self.matrix.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AffineTransform:
        return cls(np.asarray(d["matrix"], dtype=float), d.get("kind", "affine"))


def tps_kernel(r: np.ndarray) -> np.ndarray:
    """3D thin-plate kernel, U(r) = -r.

    The plan writes U(r) = r; the sign is immaterial for interpolation (lambda = 0) but
    matters for regularization: a Euclidean distance matrix is conditionally *negative*
    definite, so with +r the system K + lambda*diag(reg) passes through singularity as
    lambda grows. With -r it is conditionally positive definite and K + lambda*diag(reg)
    stays non-singular for every lambda >= 0.
    """
    return -r


@dataclass
class TPSTransform(Transform):
    """Regularized 3D thin-plate spline: f(p) = [1, p] @ affine + sum_i w_i U(|p - c_i|)."""

    control_pts: np.ndarray  # (N, 3) source (LM world nm)
    weights: np.ndarray  # (N, 3) one column per output axis
    affine: np.ndarray  # (4, 3): rows = [const, z, y, x] coefficients
    lam: float = 0.0
    inverse_model: TPSTransform | None = None
    kind: str = "tps"

    def __post_init__(self) -> None:
        self.control_pts = np.asarray(self.control_pts, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.affine = np.asarray(self.affine, dtype=float)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        single = pts.ndim == 1
        P = np.atleast_2d(pts)
        U = tps_kernel(cdist(P, self.control_pts))
        out = U @ self.weights + np.hstack([np.ones((len(P), 1)), P]) @ self.affine
        return out[0] if single else out

    def affine_part(self) -> AffineTransform:
        M = np.eye(4)
        M[:3, :3] = self.affine[1:].T
        M[:3, 3] = self.affine[0]
        return AffineTransform(M, "affine")

    def displacement(self, pts: np.ndarray) -> np.ndarray:
        """Non-affine part of the mapping at ``pts`` (what a displacement field stores)."""
        return self.apply(pts) - self.affine_part().apply(pts)

    def inverse(self) -> TPSTransform:
        if self.inverse_model is None:
            raise ValueError(
                "TPS inverse not available: fit the reverse spline first (fit_transform does)"
            )
        return self.inverse_model

    def bending_energy(self) -> float:
        K = tps_kernel(cdist(self.control_pts, self.control_pts))
        return float(sum(self.weights[:, a] @ K @ self.weights[:, a] for a in range(3)))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": "tps",
            "kind": "tps",
            "lam": float(self.lam),
            "control_pts": self.control_pts.tolist(),
            "weights": self.weights.tolist(),
            "affine": self.affine.tolist(),
        }
        if self.inverse_model is not None:
            inv = self.inverse_model.to_dict()
            inv.pop("inverse_model", None)
            d["inverse_model"] = inv
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TPSTransform:
        inv = d.get("inverse_model")
        return cls(
            np.asarray(d["control_pts"], dtype=float),
            np.asarray(d["weights"], dtype=float),
            np.asarray(d["affine"], dtype=float),
            float(d.get("lam", 0.0)),
            cls.from_dict(inv) if inv else None,
        )


def transform_from_dict(d: dict[str, Any]) -> Transform:
    return Transform.from_dict(d)
