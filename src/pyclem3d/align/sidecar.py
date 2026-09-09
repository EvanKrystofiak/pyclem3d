"""Per-slice 2D transforms stored as a sidecar (``align.json``) next to the source (plan §4)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SliceTransforms:
    """Correction to apply to each slice so the stack is aligned.

    ``shifts[i] = (dy, dx)`` in level-0 pixels: moving slice i by this vector puts it in
    the reference frame (slice 0). Integer shifts are lossless; ``subpixel`` records whether
    the user opted into fractional application. ``matrices`` (optional) hold per-slice 3x3
    homogeneous (y, x) transforms for rigid/affine corrections; when present they take
    precedence over ``shifts`` at bake time.
    """

    shifts: np.ndarray  # (n, 2) float
    confidence: np.ndarray | None = None  # (n-1,) pairwise NCC after alignment
    flagged: list[int] = field(default_factory=list)
    matrices: np.ndarray | None = None  # (n, 3, 3) or None
    subpixel: bool = False
    method: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    excluded: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shifts = np.asarray(self.shifts, dtype=float).reshape(-1, 2)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float)
        if self.matrices is not None:
            self.matrices = np.asarray(self.matrices, dtype=float)

    @property
    def n(self) -> int:
        return int(len(self.shifts))

    def integer_shifts(self) -> np.ndarray:
        return np.round(self.shifts).astype(int)

    def effective_shifts(self) -> np.ndarray:
        return self.shifts if self.subpixel else self.integer_shifts().astype(float)

    def is_identity(self, tol: float = 1e-9) -> bool:
        return bool(np.all(np.abs(self.effective_shifts()) < tol)) and self.matrices is None

    def common_bbox(self, shape_yx: tuple[int, int]) -> tuple[int, int, int, int]:
        """(y0, y1, x0, x1) of the region covered by *every* slice after shifting (in the
        output canvas, which is the reference frame of slice 0)."""
        H, W = shape_yx
        s = self.effective_shifts()
        s = np.delete(s, self.excluded, axis=0) if self.excluded else s
        y0 = int(np.ceil(max(0.0, s[:, 0].max())))
        y1 = int(np.floor(min(float(H), H + s[:, 0].min())))
        x0 = int(np.ceil(max(0.0, s[:, 1].max())))
        x1 = int(np.floor(min(float(W), W + s[:, 1].min())))
        return y0, max(y1, y0), x0, max(x1, x0)

    def union_bbox(self, shape_yx: tuple[int, int]) -> tuple[int, int, int, int]:
        H, W = shape_yx
        s = self.effective_shifts()
        return (
            int(np.floor(min(0.0, s[:, 0].min()))),
            int(np.ceil(max(float(H), H + s[:, 0].max()))),
            int(np.floor(min(0.0, s[:, 1].min()))),
            int(np.ceil(max(float(W), W + s[:, 1].max()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "tool": "pyclem3d.align",
            "source": self.source,
            "n": self.n,
            "subpixel": bool(self.subpixel),
            "shifts_yx": self.shifts.tolist(),
            "confidence": None if self.confidence is None else self.confidence.tolist(),
            "flagged": [int(i) for i in self.flagged],
            "excluded": [int(i) for i in self.excluded],
            "matrices": None if self.matrices is None else self.matrices.tolist(),
            "method": self.method,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SliceTransforms:
        return cls(
            shifts=np.asarray(d["shifts_yx"], dtype=float),
            confidence=None
            if d.get("confidence") is None
            else np.asarray(d["confidence"], dtype=float),
            flagged=[int(i) for i in d.get("flagged", [])],
            matrices=None if d.get("matrices") is None else np.asarray(d["matrices"], dtype=float),
            subpixel=bool(d.get("subpixel", False)),
            method=dict(d.get("method", {})),
            source=d.get("source", ""),
            stats=dict(d.get("stats", {})),
            excluded=[int(i) for i in d.get("excluded", [])],
        )

    def save(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> SliceTransforms:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def identity(cls, n: int) -> SliceTransforms:
        return cls(np.zeros((n, 2)))


def sidecar_path(source: str | os.PathLike) -> Path:
    p = Path(source)
    if p.is_dir():
        return p / "align.json"
    return p.with_name(p.name + ".align.json")
