"""Volume model: lazy (C, Z, Y, X) dask data + physical (world) coordinate frame.

Conventions (see plan §5):
* canonical unit is **nanometres**;
* canonical axis order is **(z, y, x)** internally, (c, z, y, x) for the array;
* ``world_affine`` maps homogeneous voxel coordinates ``(z, y, x, 1)`` of level 0 to
  world nm. It carries voxel size, origin, and any tilt / flip corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import dask.array as da
import numpy as np

Kind = Literal["em", "lm"]


def make_world_affine(
    voxel_size_nm: tuple[float, float, float],
    origin_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    y_scale: float = 1.0,
    flips: tuple[bool, bool, bool] = (False, False, False),
    shape_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Build a 4x4 voxel(z,y,x,1) -> world nm affine.

    ``y_scale`` is the FIB-SEM tilt factor applied to y (plan §4: sign/value must be
    verified against a real capture, never derived). ``flips`` mirror an axis
    about the volume extent (needs ``shape_zyx``).
    """
    dz, dy, dx = (float(v) for v in voxel_size_nm)
    scale = np.array([dz, dy * float(y_scale), dx], dtype=float)
    A = np.eye(4)
    for i in range(3):
        A[i, i] = scale[i]
        A[i, 3] = float(origin_nm[i])
    if any(flips):
        if shape_zyx is None:
            raise ValueError("flips need shape_zyx")
        for i, f in enumerate(flips):
            if f:
                n = shape_zyx[i]
                A[i, i] = -scale[i]
                A[i, 3] += scale[i] * (n - 1)
    return A


def apply_affine(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to an (N, 3) (or (3,)) array of points."""
    pts = np.asarray(pts, dtype=float)
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    out = p @ A[:3, :3].T + A[:3, 3]
    return out[0] if single else out


@dataclass
class Channel:
    """pyCLEM's per-channel model: true intensity range vs display window."""

    name: str
    index: int
    color: str = "FFFFFF"  # hex RGB
    display_min: float | None = None
    display_max: float | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    visible: bool = True
    emission_nm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Channel:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


DEFAULT_COLORS = ["00FF00", "FF00FF", "00FFFF", "FF0000", "FFFF00", "0000FF"]


@dataclass
class Volume:
    """A lazily-accessed 4D (C, Z, Y, X) image with a physical frame."""

    data: da.Array
    voxel_size_nm: tuple[float, float, float]
    world_affine: np.ndarray
    kind: Kind
    channels: list[Channel] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    psf_nm: tuple[float, float] | None = None  # (fwhm_z, fwhm_xy)
    per_slice_transforms: Any = None  # pyclem3d.align.SliceTransforms | None
    pyramid: list[da.Array] | None = None
    pyramid_factors: list[tuple[int, int, int]] | None = None
    memory_strategy: str | None = None  # "ram" | "lazy" | None (undecided)

    # ------------------------------------------------------------------ basics
    def __post_init__(self) -> None:
        if not isinstance(self.data, da.Array):
            self.data = da.from_array(np.asarray(self.data), chunks="auto")
        if self.data.ndim == 3:
            self.data = self.data[None]
        if self.data.ndim != 4:
            raise ValueError(f"Volume.data must be (C,Z,Y,X) or (Z,Y,X); got {self.data.shape}")
        self.voxel_size_nm = tuple(float(v) for v in self.voxel_size_nm)  # type: ignore[assignment]
        self.world_affine = np.asarray(self.world_affine, dtype=float)
        if self.world_affine.shape != (4, 4):
            raise ValueError("world_affine must be 4x4")
        if not self.channels:
            self.channels = [
                Channel(name=f"ch{i}", index=i, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                for i in range(self.data.shape[0])
            ]
            if self.kind == "em":
                self.channels[0].color = "FFFFFF"

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape[1:])  # type: ignore[return-value]

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.data.shape)) * self.data.dtype.itemsize

    def channel(self, i: int) -> da.Array:
        return self.data[i]

    # ------------------------------------------------------------- coordinates
    def voxel_to_world(self, vox: np.ndarray, level: int = 0) -> np.ndarray:
        return apply_affine(self.level_affine(level), vox)

    def world_to_voxel(self, world: np.ndarray, level: int = 0) -> np.ndarray:
        return apply_affine(np.linalg.inv(self.level_affine(level)), world)

    def level_affine(self, level: int = 0) -> np.ndarray:
        """World affine of a pyramid level (voxel centres of the coarsened grid)."""
        if level == 0 or not self.pyramid_factors:
            return self.world_affine
        f = np.array(self.pyramid_factors[level], dtype=float)
        L = np.eye(4)
        for i in range(3):
            L[i, i] = f[i]
            L[i, 3] = (f[i] - 1.0) / 2.0  # mean-coarsening: centre of the block
        return self.world_affine @ L

    def level_voxel_size_nm(self, level: int = 0) -> tuple[float, float, float]:
        if level == 0 or not self.pyramid_factors:
            return self.voxel_size_nm
        f = self.pyramid_factors[level]
        return tuple(v * fi for v, fi in zip(self.voxel_size_nm, f))  # type: ignore[return-value]

    def level_data(self, level: int = 0) -> da.Array:
        if level == 0 or self.pyramid is None:
            return self.data
        return self.pyramid[level]

    def n_levels(self) -> int:
        return 1 if self.pyramid is None else len(self.pyramid)

    def level_for_voxel_size(self, target_xy_nm: float) -> int:
        """Pyramid level whose xy voxel is the largest one <= target (plan §5)."""
        best = 0
        for lvl in range(self.n_levels()):
            _, dy, dx = self.level_voxel_size_nm(lvl)
            if max(dy, dx) <= target_xy_nm * 1.001:
                best = lvl
        return best

    def bbox_world(self) -> np.ndarray:
        """(2, 3) min/max world nm corners of the voxel-centre extent."""
        n = np.array(self.shape_zyx, dtype=float) - 1
        corners = np.array([[a, b, c] for a in (0, n[0]) for b in (0, n[1]) for c in (0, n[2])])
        w = self.voxel_to_world(corners)
        return np.stack([w.min(0), w.max(0)])

    # ---------------------------------------------------------------- helpers
    def with_(self, **changes: Any) -> Volume:
        return replace(self, **changes)

    def describe(self) -> str:
        c, z, y, x = self.data.shape
        gb = self.nbytes / 1e9
        strat = self.memory_strategy or "undecided"
        vs = "x".join(f"{v:g}" for v in self.voxel_size_nm)
        lv = self.n_levels()
        return (
            f"{self.kind.upper()} {self.source or '<array>'}: C={c} Z={z} Y={y} X={x} "
            f"{self.dtype} ({gb:.2f} GB) voxel {vs} nm (z,y,x), {lv} level(s), memory={strat}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Volume({self.describe()})"


def is_isotropic(voxel_size_nm: tuple[float, float, float], tol: float = 1.2) -> bool:
    v = np.asarray(voxel_size_nm, dtype=float)
    return bool(v.max() / v.min() <= tol)


def nice_gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"


def default_chunks(
    shape_czyx: tuple[int, ...], itemsize: int, target_bytes: int = 32 << 20
) -> tuple[int, ...]:
    """Chunking that keeps whole xy planes (or large tiles) and ~target bytes per chunk."""
    c, z, y, x = shape_czyx
    plane = y * x * itemsize
    if plane <= target_bytes:
        nz = max(1, min(z, target_bytes // max(plane, 1)))
        return (1, int(nz), int(y), int(x))
    side = int(math.sqrt(target_bytes / itemsize))
    side = max(256, side - side % 64)
    return (1, 1, int(min(y, side)), int(min(x, side)))
