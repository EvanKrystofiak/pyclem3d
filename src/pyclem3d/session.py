"""Session v2 (plan §5): everything needed to reproduce a correlation headlessly, bit-for-bit.

A session holds the volume specs (paths, voxel sizes, tilt factor, alignment sidecar), the
3D pre-align, the landmarks (LM side in post-pre-align world nm), the model kind and lambda,
the fitted transform, display settings and the outputs written so far.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io.pyramid import ensure_pyramid
from .io.readers import open_volume
from .io.volume import Volume
from .register.fit import FitResult, fit_landmarks
from .register.landmarks import LandmarkSet, default_sigma_nm
from .register.prealign import PreAlign3D, prealign_transform
from .register.transforms import TPSTransform, Transform, transform_from_dict
from .resample.grid import DisplacementGrid

log = logging.getLogger(__name__)

SESSION_VERSION = 2


@dataclass
class VolumeSpec:
    path: str
    kind: str
    voxel_size_nm: tuple[float, float, float] | None = None
    y_scale: float = 1.0
    origin_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channel_names: list[str] | None = None
    psf_nm: tuple[float, float] | None = None
    align_sidecar: str | None = None
    align_crop: str = "common"
    align_bad_slices: str = "keep"
    memory: str | None = None
    reader_kw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VolumeSpec:
        return cls(
            path=d["path"],
            kind=d["kind"],
            voxel_size_nm=None if d.get("voxel_size_nm") is None else tuple(d["voxel_size_nm"]),
            y_scale=float(d.get("y_scale", 1.0)),
            origin_nm=tuple(d.get("origin_nm", (0.0, 0.0, 0.0))),
            channel_names=d.get("channel_names"),
            psf_nm=None if d.get("psf_nm") is None else tuple(d["psf_nm"]),
            align_sidecar=d.get("align_sidecar"),
            align_crop=d.get("align_crop", "common"),
            align_bad_slices=d.get("align_bad_slices", "keep"),
            memory=d.get("memory"),
            reader_kw=dict(d.get("reader_kw", {})),
        )

    def resolve(self, base: Path | None) -> Path:
        p = Path(self.path)
        if not p.is_absolute() and base is not None:
            p = base / p
        return p

    def open(
        self,
        base: Path | None = None,
        cache_dir: str | os.PathLike | None = None,
        memory: str | None = None,
        pyramid: str = "lazy",
    ) -> Volume:
        vol = open_volume(
            self.resolve(base),
            kind=self.kind,
            voxel_size_nm=self.voxel_size_nm,
            y_scale=self.y_scale,
            origin_nm=self.origin_nm,
            channel_names=self.channel_names,
            psf_nm=self.psf_nm,
            memory=memory if memory is not None else self.memory,
            pyramid=pyramid if self.align_sidecar is None else "none",
            cache_dir=cache_dir,
            **self.reader_kw,
        )
        if self.align_sidecar:
            from .align.apply import apply_lazy
            from .align.sidecar import SliceTransforms

            sc = Path(self.align_sidecar)
            if not sc.is_absolute() and base is not None:
                sc = base / sc
            st = SliceTransforms.load(sc)
            vol = apply_lazy(vol, st, crop=self.align_crop, bad_slices=self.align_bad_slices)
            if pyramid != "none":
                vol = ensure_pyramid(vol, cache_dir=cache_dir, eager=(pyramid == "eager"))
        return vol


def apply_prealign(lm: Volume, pre: PreAlign3D) -> Volume:
    """Compose the pre-align into the LM world affine (LM world coords become pre-aligned coords)."""
    if pre.is_identity():
        return lm
    T = prealign_transform(pre, lm)
    meta = dict(lm.metadata)
    meta["prealign"] = pre.to_dict()
    return lm.with_(world_affine=T.matrix @ lm.world_affine, metadata=meta)


@dataclass
class Session:
    em: VolumeSpec
    lm: VolumeSpec
    prealign: PreAlign3D = field(default_factory=PreAlign3D)
    landmarks: LandmarkSet = field(default_factory=LandmarkSet)
    kind: str = "affine"
    lam: float | str | None = "auto"
    transform: dict[str, Any] | None = None
    fit: dict[str, Any] | None = None
    display: dict[str, Any] = field(default_factory=dict)
    displacement: dict[str, Any] | None = None  # {"path": ..., "spacing_nm": ...}
    outputs: dict[str, Any] = field(default_factory=dict)
    locate: dict[str, Any] | None = None  # coarse similarity from 2-3 rough pairs
    notes: str = ""
    created: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    modified: str = ""
    path: str | None = None
    version: int = SESSION_VERSION

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tool": "pyclem3d",
            "created": self.created,
            "modified": self.modified,
            "em": self.em.to_dict(),
            "lm": self.lm.to_dict(),
            "prealign": self.prealign.to_dict(),
            "landmarks": self.landmarks.to_dict(),
            "kind": self.kind,
            "lambda": self.lam,
            "transform": self.transform,
            "fit": self.fit,
            "display": self.display,
            "displacement": self.displacement,
            "outputs": self.outputs,
            "locate": self.locate,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: str | None = None) -> Session:
        if int(d.get("version", 2)) != SESSION_VERSION:
            log.warning("session version %s; expected %s", d.get("version"), SESSION_VERSION)
        return cls(
            em=VolumeSpec.from_dict(d["em"]),
            lm=VolumeSpec.from_dict(d["lm"]),
            prealign=PreAlign3D.from_dict(d.get("prealign")),
            landmarks=LandmarkSet.from_dict(d.get("landmarks", {})),
            kind=d.get("kind", "affine"),
            lam=d.get("lambda", "auto"),
            transform=d.get("transform"),
            fit=d.get("fit"),
            display=dict(d.get("display", {})),
            displacement=d.get("displacement"),
            outputs=dict(d.get("outputs", {})),
            locate=d.get("locate"),
            notes=d.get("notes", ""),
            created=d.get("created", ""),
            modified=d.get("modified", ""),
            path=path,
        )

    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path is not None else Path(self.path or "session.json")
        self.path = str(p)
        self.modified = _dt.datetime.now().isoformat(timespec="seconds")
        p.write_text(json.dumps(self.to_dict(), indent=1, default=_json_default), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> Session:
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), path=str(p))

    @property
    def base_dir(self) -> Path | None:
        return Path(self.path).parent if self.path else None

    # ------------------------------------------------------------------ volumes
    def open_volumes(
        self,
        cache_dir: str | os.PathLike | None = None,
        memory: str | None = None,
        pyramid: str = "lazy",
    ) -> tuple[Volume, Volume]:
        em = self.em.open(self.base_dir, cache_dir, memory, pyramid)
        lm = self.lm.open(self.base_dir, cache_dir, memory, pyramid)
        return em, apply_prealign(lm, self.prealign)

    def set_prealign(self, new: PreAlign3D, lm_raw: Volume) -> None:
        """Change the pre-align and remap the landmarks so they stay pinned (plan §6)."""
        from .register.prealign import remap_landmarks

        self.landmarks = remap_landmarks(self.landmarks, self.prealign, new, lm_raw)
        self.prealign = new
        self.transform = None
        self.fit = None

    # ---------------------------------------------------------------- fitting
    def transform_obj(self) -> Transform | None:
        return None if self.transform is None else transform_from_dict(self.transform)

    def do_fit(self, with_loo: bool = True) -> FitResult:
        fr = fit_landmarks(self.landmarks, self.kind, self.lam, with_loo=with_loo)
        self.transform = fr.transform.to_dict()
        d = fr.to_dict()
        d.pop("transform", None)
        self.fit = d
        self.displacement = None
        return fr

    def ensure_displacement(
        self, em: Volume, spacing_nm: float | None = None, path: str | os.PathLike | None = None
    ) -> DisplacementGrid | None:
        """Build (or load) the coarse inverse displacement grid for a deformable fit (plan §7)."""
        t = self.transform_obj()
        if t is None or not isinstance(t, TPSTransform):
            return None
        sp = (
            float(spacing_nm)
            if spacing_nm is not None
            else float(self.display.get("displacement_spacing_nm", 1000.0))
        )
        if self.displacement and self.displacement.get("spacing_nm") == sp:
            p = Path(self.displacement["path"])
            if not p.is_absolute() and self.base_dir is not None:
                p = self.base_dir / p
            if p.exists():
                return DisplacementGrid.load(str(p))
        dg = DisplacementGrid.build(t, em.bbox_world(), spacing_nm=sp)
        if path is None:
            if self.path:
                path = Path(self.path).with_suffix(".disp.npz")
            else:  # unsaved session: keep the cache out of the working directory
                from .io.pyramid import default_cache_dir

                d = default_cache_dir()
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"session-{id(self):x}.disp.npz"
        dg.save(str(path))
        self.displacement = {
            "path": str(path),
            "spacing_nm": sp,
            "round_trip": dg.round_trip(t, em.bbox_world()),
        }
        return dg

    def default_sigma(self, lm: Volume, em: Volume | None = None) -> tuple[float, float, float]:
        return default_sigma_nm(lm, em)

    # ------------------------------------------------------------------- run
    def run(
        self,
        cache_dir: str | os.PathLike | None = None,
        report_path: str | os.PathLike | None = None,
        with_loo: bool = True,
        memory: str | None = None,
    ) -> dict[str, Any]:
        """Reproduce the fit headlessly and write the report (plan §5, §9)."""
        from .export.report import build_report, write_report

        em, lm = self.open_volumes(cache_dir=cache_dir, memory=memory)
        fr = self.do_fit(with_loo=with_loo)
        rep = build_report(
            em,
            lm,
            self.landmarks,
            fr,
            session={
                "path": self.path,
                "kind": self.kind,
                "lambda": self.lam,
                "prealign": self.prealign.to_dict(),
            },
            outputs=self.outputs,
        )
        if report_path is not None:
            write_report(rep, report_path)
            self.outputs["report"] = str(report_path)
        return rep


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)
