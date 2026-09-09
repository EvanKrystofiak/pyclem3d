"""Landmark model (plan §5): endogenous features as world-nm points with per-axis uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..io.metadata import psf_fallback
from ..io.volume import Volume

Vec3 = tuple[float, float, float]


def default_sigma_nm(lm: Volume, em: Volume | None = None) -> Vec3:
    """Default per-axis sigma: half the LM xy voxel; half the axial PSF FWHM for z (plan §6)."""
    dz, dy, dx = lm.voxel_size_nm
    sxy = 0.5 * max(dy, dx)
    fwhm_z = lm.psf_nm[0] if lm.psf_nm else psf_fallback(lm.voxel_size_nm)[0]
    sz = 0.5 * fwhm_z
    if em is not None:  # EM voxel adds (tiny) uncertainty in quadrature
        ez, ey, ex = em.voxel_size_nm
        sz = float(np.hypot(sz, 0.5 * ez))
        sxy = float(np.hypot(sxy, 0.5 * max(ey, ex)))
    return (float(sz), float(sxy), float(sxy))


@dataclass
class Landmark3D:
    id: int
    em_world_nm: Vec3
    lm_world_nm: Vec3  # post-pre-align LM world
    sigma_nm: Vec3
    em_voxel: Vec3 | None = None
    lm_voxel: Vec3 | None = None
    method: dict[str, str] = field(default_factory=lambda: {"em": "click", "lm": "click"})
    feature: str = ""
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "em_world_nm": [float(v) for v in self.em_world_nm],
            "lm_world_nm": [float(v) for v in self.lm_world_nm],
            "sigma_nm": [float(v) for v in self.sigma_nm],
            "em_voxel": None if self.em_voxel is None else [float(v) for v in self.em_voxel],
            "lm_voxel": None if self.lm_voxel is None else [float(v) for v in self.lm_voxel],
            "method": dict(self.method),
            "feature": self.feature,
            "enabled": bool(self.enabled),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Landmark3D:
        return cls(
            id=int(d["id"]),
            em_world_nm=tuple(d["em_world_nm"]),  # type: ignore[arg-type]
            lm_world_nm=tuple(d["lm_world_nm"]),  # type: ignore[arg-type]
            sigma_nm=tuple(d["sigma_nm"]),  # type: ignore[arg-type]
            em_voxel=None if d.get("em_voxel") is None else tuple(d["em_voxel"]),  # type: ignore[arg-type]
            lm_voxel=None if d.get("lm_voxel") is None else tuple(d["lm_voxel"]),  # type: ignore[arg-type]
            method=dict(d.get("method", {"em": "click", "lm": "click"})),
            feature=d.get("feature", ""),
            enabled=bool(d.get("enabled", True)),
            notes=d.get("notes", ""),
        )


@dataclass
class LandmarkSet:
    landmarks: list[Landmark3D] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self):
        return iter(self.landmarks)

    def next_id(self) -> int:
        return 1 + max((lm.id for lm in self.landmarks), default=0)

    def add(
        self,
        em_world_nm: Vec3,
        lm_world_nm: Vec3,
        sigma_nm: Vec3,
        **kw: Any,
    ) -> Landmark3D:
        lm = Landmark3D(
            self.next_id(), tuple(em_world_nm), tuple(lm_world_nm), tuple(sigma_nm), **kw
        )  # type: ignore[arg-type]
        self.landmarks.append(lm)
        return lm

    def get(self, id_: int) -> Landmark3D:
        for lm in self.landmarks:
            if lm.id == id_:
                return lm
        raise KeyError(id_)

    def remove(self, id_: int) -> None:
        self.landmarks = [lm for lm in self.landmarks if lm.id != id_]

    def enabled(self) -> list[Landmark3D]:
        return [lm for lm in self.landmarks if lm.enabled]

    @property
    def n_enabled(self) -> int:
        return len(self.enabled())

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(src=LM world, dst=EM world, sigma, ids) of the enabled landmarks."""
        en = self.enabled()
        if not en:
            z = np.zeros((0, 3))
            return z, z.copy(), z.copy(), np.zeros(0, dtype=int)
        src = np.array([lm.lm_world_nm for lm in en], dtype=float)
        dst = np.array([lm.em_world_nm for lm in en], dtype=float)
        sig = np.array([lm.sigma_nm for lm in en], dtype=float)
        ids = np.array([lm.id for lm in en], dtype=int)
        return src, dst, sig, ids

    def spread_report(self) -> dict[str, Any]:
        """How well the landmarks cover 3D (plan §3: aim for 10-30 spread in z)."""
        src, dst, _, _ = self.arrays()
        if len(dst) == 0:
            return {"n": 0}
        ext = dst.max(0) - dst.min(0)
        rank = int(np.linalg.matrix_rank(dst - dst.mean(0), tol=1e-3 * max(1.0, ext.max())))
        return {
            "n": int(len(dst)),
            "em_extent_nm": [float(e) for e in ext],
            "rank": rank,
            "coplanar": rank < 3,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"landmarks": [lm.to_dict() for lm in self.landmarks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LandmarkSet:
        return cls([Landmark3D.from_dict(x) for x in d.get("landmarks", [])])

    def copy(self) -> LandmarkSet:
        return LandmarkSet.from_dict(self.to_dict())
