"""Pre-align 3D (plan §6): axis permutation + flips + coarse rotation about z of the LM volume.

The pre-align is a world-space affine (LM world nm -> pre-aligned LM world nm) about
the LM volume centre. Landmarks store the *post*-pre-align LM coordinate; changing
the pre-align remaps them so they stay pinned to the same physical LM feature
(pyCLEM's ``prealign_map``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..io.volume import Volume
from .landmarks import LandmarkSet
from .transforms import AffineTransform


@dataclass
class PreAlign3D:
    permutation: tuple[int, int, int] = (0, 1, 2)  # output axis i takes input axis permutation[i]
    flips: tuple[bool, bool, bool] = (False, False, False)  # applied to (z, y, x) before permuting
    rotation_z_deg: float = 0.0  # rotation in the (y, x) plane, after permutation

    def is_identity(self) -> bool:
        return (
            tuple(self.permutation) == (0, 1, 2)
            and not any(self.flips)
            and abs(self.rotation_z_deg) < 1e-12
        )

    def linear(self) -> np.ndarray:
        F = np.diag([-1.0 if f else 1.0 for f in self.flips])
        P = np.zeros((3, 3))
        for i, j in enumerate(self.permutation):
            P[i, j] = 1.0
        th = np.radians(self.rotation_z_deg)
        c, s = np.cos(th), np.sin(th)
        # rotation about z in (z, y, x) coordinates: y' = c*y - s*x ; x' = s*y + c*x
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return R @ P @ F

    def matrix(self, center_nm: np.ndarray) -> np.ndarray:
        """4x4 world->world about ``center_nm`` (z, y, x)."""
        c = np.asarray(center_nm, dtype=float)
        L = self.linear()
        M = np.eye(4)
        M[:3, :3] = L
        M[:3, 3] = c - L @ c
        return M

    def to_dict(self) -> dict[str, Any]:
        return {
            "permutation": [int(p) for p in self.permutation],
            "flips": [bool(f) for f in self.flips],
            "rotation_z_deg": float(self.rotation_z_deg),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PreAlign3D:
        if not d:
            return cls()
        return cls(
            tuple(d.get("permutation", (0, 1, 2))),  # type: ignore[arg-type]
            tuple(d.get("flips", (False, False, False))),  # type: ignore[arg-type]
            float(d.get("rotation_z_deg", 0.0)),
        )


def volume_center_nm(vol: Volume) -> np.ndarray:
    bb = vol.bbox_world()
    return (bb[0] + bb[1]) / 2.0


def prealign_transform(pre: PreAlign3D, lm: Volume) -> AffineTransform:
    """LM world nm -> pre-aligned LM world nm."""
    return AffineTransform(pre.matrix(volume_center_nm(lm)), "prealign")


def remap_landmarks(
    landmarks: LandmarkSet, old: PreAlign3D, new: PreAlign3D, lm: Volume
) -> LandmarkSet:
    """Return a copy whose LM coordinates are re-expressed under ``new`` pre-align."""
    out = landmarks.copy()
    if old.to_dict() == new.to_dict():
        return out
    M_old = prealign_transform(old, lm)
    M_new = prealign_transform(new, lm)
    chain = M_new.compose(M_old.inverse())
    for lmk in out.landmarks:
        p = chain.apply(np.asarray(lmk.lm_world_nm, dtype=float))
        lmk.lm_world_nm = (float(p[0]), float(p[1]), float(p[2]))
    return out
