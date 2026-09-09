"""Output 5 (plan §9): report.json and figure slices (PNG overlays with scale bars)."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..io.volume import Channel, Volume
from ..register.fit import FitResult
from ..register.landmarks import LandmarkSet
from ..register.transforms import Transform
from ..resample.grid import DisplacementGrid
from ..resample.slab import SlabCache, SlabParams, em_level_for_lm
from ..resample.zview import lm_plane_at_em_slice, z_readout

log = logging.getLogger(__name__)


def _volume_summary(v: Volume) -> dict[str, Any]:
    return {
        "source": v.source,
        "kind": v.kind,
        "shape_czyx": [int(s) for s in v.data.shape],
        "dtype": str(v.dtype),
        "voxel_size_nm": list(v.voxel_size_nm),
        "voxel_size_source": v.metadata.get("voxel_size_source"),
        "world_affine": v.world_affine.tolist(),
        "n_levels": v.n_levels(),
        "memory_strategy": v.memory_strategy,
        "psf_nm": None if v.psf_nm is None else list(v.psf_nm),
        "channels": [c.name for c in v.channels],
        "warnings": list(v.metadata.get("warnings", [])),
        "recommend_zarr": v.metadata.get("recommend_zarr"),
        "alignment": v.metadata.get("alignment"),
    }


def build_report(
    em: Volume | None,
    lm: Volume | None,
    landmarks: LandmarkSet | None,
    fit: FitResult | None,
    align_stats: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    rep: dict[str, Any] = {
        "tool": "pyclem3d",
        "version": _version(),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "units": "nm",
    }
    if em is not None:
        rep["em"] = _volume_summary(em)
        warnings += [f"EM: {w}" for w in rep["em"]["warnings"]]
    if lm is not None:
        rep["lm"] = _volume_summary(lm)
        warnings += [f"LM: {w}" for w in rep["lm"]["warnings"]]
    if landmarks is not None:
        rep["landmarks"] = {
            "n": len(landmarks),
            "n_enabled": landmarks.n_enabled,
            "spread": landmarks.spread_report(),
            "items": landmarks.to_dict()["landmarks"],
        }
        if landmarks.spread_report().get("coplanar"):
            warnings.append("landmarks are coplanar: the fit is unconstrained in one direction")
    if fit is not None:
        f = fit.to_dict()
        rep["fit"] = {
            "kind": f["kind"],
            "n": f["n"],
            "lambda": f["lam"],
            "rms_nm": {
                k: f["rms"].get(k) for k in ("rms_total", "rms_xy", "rms_z", "max", "mean_abs_z")
            },
            "loo_rms_nm": None
            if not f["loo"]
            else {
                k: f["loo"].get(k) for k in ("rms_total", "rms_xy", "rms_z", "max", "n_evaluated")
            },
            "per_point": [
                {
                    "id": (None if f["ids"] is None else f["ids"][i]),
                    "residual_nm": f["residuals_nm"][i],
                    "norm_nm": f["rms"]["per_point"][i],
                    "loo_nm": (None if not f["loo"] else f["loo"]["per_point_full"][i]),
                }
                for i in range(f["n"])
            ],
            "round_trip": f["round_trip"],
            "lambda_search": f["lambda_search"],
            "z_check": f["qc"].get("z_check"),
            "decomposition": f["qc"].get("decomposition"),
            "normalized_rms": f["qc"].get("normalized_rms"),
            "transform": f["transform"],
            "warnings": f["warnings"],
        }
        warnings += list(f["warnings"])
    if align_stats:
        rep["alignment"] = {
            k: v
            for k, v in align_stats.items()
            if k not in ("trajectory_before_px", "trajectory_after_px", "pair_conf", "bad_info")
        }
        if align_stats.get("flagged"):
            warnings.append(f"EM alignment flagged slices {align_stats['flagged']}")
    if session:
        rep["session"] = session
    if outputs:
        rep["outputs"] = outputs
    if extra:
        rep.update(extra)
    rep["warnings"] = warnings
    return rep


def write_report(report: dict[str, Any], path: str | os.PathLike) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=1, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


# ------------------------------------------------------------------ figures
def hex_to_rgb(color: str | None) -> np.ndarray:
    c = (color or "FFFFFF").lstrip("#")
    return np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], dtype=np.float32) / 255.0


def auto_window(plane: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    p = np.asarray(plane, dtype=np.float32)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(finite, [lo, hi])
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


def _norm(plane: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    lo, hi = window if window is not None else auto_window(plane)
    return np.clip((np.asarray(plane, dtype=np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)


def composite_rgb(
    em_plane: np.ndarray | None,
    lm_planes: list[np.ndarray],
    lm_channels: list[Channel],
    em_window: tuple[float, float] | None = None,
    lm_windows: list[tuple[float, float] | None] | None = None,
    mode: str = "blend",
    alpha: float = 0.6,
    checker_px: int = 64,
    swipe_frac: float = 0.5,
) -> np.ndarray:
    """RGB uint8 composite: EM in gray + additive LM colours. Modes: blend | checker | swipe | em | lm."""
    shape = em_plane.shape if em_plane is not None else lm_planes[0].shape
    em_rgb = np.zeros((*shape, 3), np.float32)
    if em_plane is not None:
        g = _norm(em_plane, em_window)
        em_rgb[:] = g[..., None]
    lm_rgb = np.zeros((*shape, 3), np.float32)
    for i, (pl, ch) in enumerate(zip(lm_planes, lm_channels)):
        if not ch.visible:
            continue
        w = None if lm_windows is None else lm_windows[i]
        if w is None and ch.display_min is not None and ch.display_max is not None:
            w = (ch.display_min, ch.display_max)
        lm_rgb += _norm(pl, w)[..., None] * hex_to_rgb(ch.color)[None, None, :]
    lm_rgb = np.clip(lm_rgb, 0, 1)
    if mode == "em":
        out = em_rgb
    elif mode == "lm":
        out = lm_rgb
    elif mode == "checker":
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        mask = ((yy // checker_px) + (xx // checker_px)) % 2 == 0
        out = np.where(mask[..., None], em_rgb, lm_rgb)
    elif mode == "swipe":
        cut = int(shape[1] * swipe_frac)
        out = em_rgb.copy()
        out[:, cut:] = lm_rgb[:, cut:]
    else:  # blend: screen LM over EM
        out = np.clip(em_rgb * (1 - alpha) + em_rgb * alpha * (1 - lm_rgb) + lm_rgb * alpha, 0, 1)
        out = np.maximum(out, lm_rgb * alpha)
    return (out * 255).astype(np.uint8)


def nice_scale_bar_nm(width_nm: float) -> float:
    target = width_nm / 5.0
    for cand in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000):
        if cand >= target:
            return float(cand)
    return 100000.0


def draw_scale_bar(
    rgb: np.ndarray,
    voxel_x_nm: float,
    length_nm: float | None = None,
    margin: int = 12,
    thickness: int | None = None,
) -> tuple[np.ndarray, float]:
    H, W, _ = rgb.shape
    L = length_nm or nice_scale_bar_nm(W * voxel_x_nm)
    px = int(round(L / voxel_x_nm))
    px = max(2, min(px, W - 2 * margin))
    t = thickness or max(2, H // 100)
    out = rgb.copy()
    y1 = H - margin
    y0 = max(y1 - t, 0)
    x0 = W - margin - px
    out[y0:y1, x0 : x0 + px] = 255
    out[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x0 + px + 1][:, :, :] = np.maximum(
        out[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x0 + px + 1], 0
    )
    return out, L


def figure_em_slice(
    em: Volume,
    lm: Volume,
    transform: Transform,
    j: int,
    out_png: str | os.PathLike,
    em_level: int | None = None,
    mode: str = "blend",
    lm_channels: list[int] | None = None,
    displacement: DisplacementGrid | None = None,
    scale_bar_nm: float | None = None,
    psf_aware: bool = False,
) -> dict[str, Any]:
    """EM-driven figure: EM slice ``j`` (a level-0 index) drawn at ``em_level`` with the LM
    resampled at that plane."""
    import imageio.v3 as iio

    lvl = em_level if em_level is not None else em_level_for_lm(em, lm)
    chans = list(range(lm.n_channels)) if lm_channels is None else list(lm_channels)
    fz = em.pyramid_factors[lvl][0] if em.pyramid_factors else 1
    j = int(np.clip(round(int(j) / fz), 0, em.level_data(lvl).shape[1] - 1))
    em_plane = np.asarray(em.level_data(lvl)[0, j].compute())
    planes, cov = lm_plane_at_em_slice(
        lm,
        em,
        transform,
        j,
        em_level=lvl,
        channels=chans,
        psf_aware=psf_aware,
        displacement=displacement,
    )
    rgb = composite_rgb(
        em_plane, [planes[i] for i in range(len(chans))], [lm.channels[c] for c in chans], mode=mode
    )
    rgb, L = draw_scale_bar(rgb, em.level_voxel_size_nm(lvl)[2], scale_bar_nm)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_png, rgb)
    return {
        "path": str(out_png),
        "em_z": int(j),
        "em_level": int(lvl),
        "scale_bar_nm": L,
        "readout": z_readout(lm, em, transform, em_j=j, em_level=lvl, displacement=displacement),
        "coverage": float(cov.mean()),
    }


def figure_lm_slice(
    em: Volume,
    lm: Volume,
    transform: Transform,
    k: int,
    out_png: str | os.PathLike,
    slab: SlabParams | None = None,
    mode: str = "blend",
    lm_channels: list[int] | None = None,
    scale_bar_nm: float | None = None,
    cache: SlabCache | None = None,
) -> dict[str, Any]:
    """LM-driven figure: LM slice k with the EM slab projection in the LM grid."""
    import imageio.v3 as iio

    chans = list(range(lm.n_channels)) if lm_channels is None else list(lm_channels)
    p = slab or SlabParams()
    if cache is not None:
        em_img, cov, info = cache.get(k, p)
    else:
        from ..resample.slab import em_slab_for_lm_slice

        em_img, cov, info = em_slab_for_lm_slice(em, lm, transform, k, p)
    lm_planes = [np.asarray(lm.data[c, k].compute()) for c in chans]
    rgb = composite_rgb(em_img, lm_planes, [lm.channels[c] for c in chans], mode=mode)
    rgb, L = draw_scale_bar(rgb, lm.voxel_size_nm[2], scale_bar_nm)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_png, rgb)
    return {
        "path": str(out_png),
        "lm_z": int(k),
        "scale_bar_nm": L,
        "readout": z_readout(lm, em, transform, lm_k=k),
        "slab": info,
        "coverage": float(cov.mean()),
    }
