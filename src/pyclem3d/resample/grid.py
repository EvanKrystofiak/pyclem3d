"""Output grids and inverse mapping helpers (plan §7 'lazy resampling mechanics')."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from ..io.volume import Volume, apply_affine
from ..register.transforms import AffineTransform, TPSTransform, Transform


@dataclass
class Grid:
    """A regular voxel grid with a voxel(z,y,x,1) -> world nm affine."""

    shape_zyx: tuple[int, int, int]
    affine: np.ndarray

    def __post_init__(self) -> None:
        self.shape_zyx = tuple(int(s) for s in self.shape_zyx)  # type: ignore[assignment]
        self.affine = np.asarray(self.affine, dtype=float)

    @classmethod
    def from_volume(cls, vol: Volume, level: int = 0) -> Grid:
        return cls(tuple(int(s) for s in vol.level_data(level).shape[1:]), vol.level_affine(level))  # type: ignore[arg-type]

    @classmethod
    def from_bbox(cls, bbox_world: np.ndarray, voxel_size_nm: tuple[float, float, float]) -> Grid:
        """Axis-aligned grid covering ``bbox_world`` ((2,3) min/max) at the given voxel size."""
        lo = np.asarray(bbox_world[0], dtype=float)
        hi = np.asarray(bbox_world[1], dtype=float)
        vs = np.asarray(voxel_size_nm, dtype=float)
        shape = tuple(int(max(1, np.ceil((h - l) / v) + 1)) for l, h, v in zip(lo, hi, vs))
        A = np.eye(4)
        A[:3, :3] = np.diag(vs)
        A[:3, 3] = lo
        return cls(shape, A)  # type: ignore[arg-type]

    @property
    def voxel_size_nm(self) -> tuple[float, float, float]:
        return tuple(float(np.linalg.norm(self.affine[:3, i])) for i in range(3))  # type: ignore[return-value]

    @property
    def origin_nm(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.affine[:3, 3])  # type: ignore[return-value]

    def voxel_to_world(self, vox: np.ndarray) -> np.ndarray:
        return apply_affine(self.affine, vox)

    def world_to_voxel(self, world: np.ndarray) -> np.ndarray:
        return apply_affine(np.linalg.inv(self.affine), world)

    def block_world_coords(
        self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        """World coords ((z1-z0)*(y1-y0)*(x1-x0), 3) of a block of voxel centres."""
        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1, dtype=float),
            np.arange(y0, y1, dtype=float),
            np.arange(x0, x1, dtype=float),
            indexing="ij",
        )
        vox = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        return self.voxel_to_world(vox)

    def bbox_world(self) -> np.ndarray:
        n = np.asarray(self.shape_zyx, dtype=float) - 1
        corners = np.array([[a, b, c] for a in (0, n[0]) for b in (0, n[1]) for c in (0, n[2])])
        w = self.voxel_to_world(corners)
        return np.stack([w.min(0), w.max(0)])

    def to_dict(self) -> dict[str, Any]:
        return {"shape_zyx": list(self.shape_zyx), "affine": self.affine.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        return cls(tuple(d["shape_zyx"]), np.asarray(d["affine"]))  # type: ignore[arg-type]


def invert_numerically(t: Transform, dst_pts: np.ndarray, n_iter: int = 8) -> np.ndarray:
    """Solve T(x) = p for x by Newton-like iteration with the affine part's Jacobian.

    For a TPS whose non-affine part is a modest perturbation this converges in a few
    iterations; used to refine the reverse-spline inverse and to build displacement grids.
    """
    dst_pts = np.atleast_2d(np.asarray(dst_pts, dtype=float))
    if isinstance(t, AffineTransform):
        return t.inverse().apply(dst_pts)
    assert isinstance(t, TPSTransform)
    A = t.affine_part()
    Ainv_lin = np.linalg.inv(A.linear)
    x = (
        t.inverse_model.apply(dst_pts)
        if t.inverse_model is not None
        else A.inverse().apply(dst_pts)
    )
    for _ in range(n_iter):
        r = dst_pts - t.apply(x)
        if np.abs(r).max() < 1e-6:
            break
        x = x + r @ Ainv_lin.T
    return x


@dataclass
class DisplacementGrid:
    """Inverse mapping dst world -> src world sampled on a coarse world grid (plan §7).

    ``values[k, j, i] = inverse(p) - p`` at grid point p; trilinear interpolation gives the
    inverse anywhere. Built once per deformable fit and cached in the session.
    """

    origin_nm: np.ndarray  # (3,)
    spacing_nm: np.ndarray  # (3,)
    values: np.ndarray  # (nz, ny, nx, 3)
    fallback: AffineTransform  # used outside the grid (affine part inverse)

    @classmethod
    def build(
        cls,
        t: Transform,
        bbox_world: np.ndarray,
        spacing_nm: float | tuple[float, float, float] = 1000.0,
        margin_nm: float = 2000.0,
        refine: bool = True,
    ) -> DisplacementGrid:
        sp = np.broadcast_to(np.asarray(spacing_nm, dtype=float), (3,)).copy()
        lo = np.asarray(bbox_world[0], dtype=float) - margin_nm
        hi = np.asarray(bbox_world[1], dtype=float) + margin_nm
        n = np.maximum(2, np.ceil((hi - lo) / sp).astype(int) + 1)
        zz, yy, xx = np.meshgrid(
            *[lo[i] + sp[i] * np.arange(n[i]) for i in range(3)], indexing="ij"
        )
        pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        if isinstance(t, AffineTransform):
            inv = t.inverse().apply(pts)
            fallback = t.inverse()
        else:
            assert isinstance(t, TPSTransform)
            inv = invert_numerically(t, pts, n_iter=8 if refine else 0)
            fallback = t.affine_part().inverse()
        vals = (inv - pts).reshape(*n, 3).astype(np.float32)
        return cls(lo, sp, vals, fallback)

    def inverse(self, dst_pts: np.ndarray) -> np.ndarray:
        """dst world -> src world by trilinear interpolation of the displacement."""
        p = np.atleast_2d(np.asarray(dst_pts, dtype=float))
        coords = ((p - self.origin_nm) / self.spacing_nm).T  # (3, N)
        out = np.empty_like(p)
        for a in range(3):
            out[:, a] = map_coordinates(self.values[..., a], coords, order=1, mode="nearest")
        return p + out

    def round_trip(self, t: Transform, pts: np.ndarray) -> dict[str, float]:
        back = t.apply(self.inverse(pts))
        err = np.linalg.norm(back - np.atleast_2d(pts), axis=1)
        return {"rms_nm": float(np.sqrt(np.mean(err**2))), "max_nm": float(err.max())}

    def save(self, path: str) -> str:
        np.savez_compressed(
            path,
            origin_nm=self.origin_nm,
            spacing_nm=self.spacing_nm,
            values=self.values,
            fallback=self.fallback.matrix,
        )
        return path

    @classmethod
    def load(cls, path: str) -> DisplacementGrid:
        d = np.load(path)
        return cls(d["origin_nm"], d["spacing_nm"], d["values"], AffineTransform(d["fallback"]))


class InverseMapper:
    """dst world -> src world for any transform: exact for linear, grid-based for deformable."""

    def __init__(self, t: Transform, grid: DisplacementGrid | None = None):
        self.t = t
        self.grid = grid
        if isinstance(t, AffineTransform):
            self._inv = t.inverse()
        else:
            self._inv = None
            if grid is None:
                raise ValueError("deformable transforms need a DisplacementGrid for the inverse")

    def __call__(self, dst_pts: np.ndarray) -> np.ndarray:
        if self._inv is not None:
            return self._inv.apply(dst_pts)
        assert self.grid is not None
        return self.grid.inverse(dst_pts)

    @property
    def is_linear(self) -> bool:
        return self._inv is not None
