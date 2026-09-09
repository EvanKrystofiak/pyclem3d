"""EM stack alignment pipeline (plan §4): coarse-to-fine, translation first, drift options,
bad-slice handling, QC stats. Independent of the registration core; usable from the CLI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..io.volume import Volume
from .badslices import detect_bad_slices, drop_measurements_touching, interpolate_over
from .pairwise import pairwise_measurements, phase_shift, rigid_pair
from .sidecar import SliceTransforms
from .trajectory import (
    chain,
    highpass,
    running_mean_corrections,
    solve_multi_neighbour,
    trajectory_stats,
)

log = logging.getLogger(__name__)


@dataclass
class AlignResult:
    transforms: SliceTransforms
    measurements: list[dict[str, Any]]
    level: int
    factor: tuple[float, float]
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            f"align: n={self.transforms.n} level={self.level} method={self.transforms.method.get('method')} "
            f"drift={self.transforms.method.get('drift')} jitter RMS {s.get('jitter_rms_px_before', 0):.2f} -> "
            f"{s.get('jitter_rms_px_after', 0):.2f} px, median pair NCC {s.get('median_conf', 0):.2f}, "
            f"flagged {self.transforms.flagged}"
        )


def choose_level(volume: Volume, target_px: int = 1536) -> int:
    """Pyramid level where a slice is ~1-2k px on its long side (plan §4 step 1)."""
    best = 0
    for lvl in range(volume.n_levels()):
        _, Y, X = volume.level_data(lvl).shape[1:]
        if max(Y, X) >= target_px:
            best = lvl
        else:
            break
    return best


def align_stack(
    volume: Volume,
    method: str = "multi",
    offsets: tuple[int, ...] = (1, 2, 4),
    drift: str = "keep",
    highpass_sigma: float = 5.0,
    running_k: int = 5,
    upsample: int = 10,
    level: int | None = None,
    subpixel: bool = False,
    gpu: bool = False,
    per_pair: str = "translation",
    bad: str = "flag",
    channel: int = 0,
    refine_fine: bool = False,
    conf_threshold: float | None = None,
    margin: int = 0,
    progress=None,
) -> AlignResult:
    """Estimate per-slice corrections for an EM stack.

    method: "multi" (multi-neighbour consensus, default), "chain" (pairwise chaining) or
            "running-mean" (register to the mean of the previous k aligned slices).
    drift:  "keep" (high-pass the trajectory so slow drift survives as real geometry and only
            jitter is corrected) or "remove" (straighten everything to slice 0's frame).
    bad:    "flag" (interpolate the transform across flagged slices, list them) or
            "exclude" (also mark them for replacement/dropping at apply time).
    per_pair: "translation" (default) or "rigid"/"affine" (ORB+RANSAC, opt-in, stored as matrices).
    margin: border (level px) to ignore when measuring; use it for stacks with padded/blank edges.
    """
    lvl = choose_level(volume) if level is None else int(level)
    data = volume.level_data(lvl)[channel]
    n = int(data.shape[0])
    f = volume.level_voxel_size_nm(lvl)
    base = volume.voxel_size_nm
    factor = (f[1] / base[1], f[2] / base[2])

    def get_slice(i: int) -> np.ndarray:
        s = np.asarray(data[i].compute())
        if margin > 0:
            s = s[margin:-margin, margin:-margin]
        return s

    if method == "running-mean":
        offsets = (1, 2)
    measurements = pairwise_measurements(
        get_slice, n, offsets=offsets, upsample=upsample, gpu=gpu, progress=progress
    )
    flagged, bad_info = detect_bad_slices(measurements, n, conf_threshold=conf_threshold)
    # what the stack currently does (level px), bridged across flagged slices so their
    # bogus pair shifts do not enter the before/after jitter statistics
    # (the chained *corrections* are the negative of the content's motion)
    raw_traj = -_fix_chain_gaps(
        chain(drop_measurements_touching(measurements, flagged), n), measurements, flagged, n
    )

    if method == "running-mean":
        clean = drop_measurements_touching(measurements, flagged)
        c0 = _fix_chain_gaps(chain(clean, n), measurements, flagged, n)
        c, conf_rm = running_mean_corrections(
            get_slice, n, k=running_k, upsample=upsample, gpu=gpu, initial=c0, skip=flagged
        )
        c = interpolate_over(c, flagged)
    elif method == "chain":
        clean = drop_measurements_touching(measurements, flagged)
        c = chain(clean, n)
        # chain() has a zero step where a k=1 pair was dropped; bridge with k=2/4 measurements
        c = _fix_chain_gaps(c, measurements, flagged, n)
    elif method == "multi":
        clean = drop_measurements_touching(measurements, flagged)
        c = solve_multi_neighbour(clean, n)
        c = interpolate_over(c, flagged)
    else:
        raise ValueError(f"unknown method {method!r}")

    if drift == "keep":
        c = highpass(c, highpass_sigma)
    elif drift != "remove":
        raise ValueError("drift must be 'keep' or 'remove'")

    if refine_fine and lvl > 0:
        c = _refine_fine(volume, lvl, channel, c, upsample, gpu)

    c_level0 = c * np.asarray(factor)[None, :]
    conf = np.full(max(n - 1, 0), np.nan)
    for m in measurements:
        if m["k"] == 1:
            conf[m["j"] - 1] = m["conf"]
    matrices = None
    if per_pair in ("rigid", "affine"):
        matrices = _per_pair_matrices(get_slice, n, per_pair, c, factor, flagged)

    after = raw_traj + c
    stats = trajectory_stats(raw_traj * np.asarray(factor), after * np.asarray(factor))
    stats.update(
        {
            "median_conf": float(np.nanmedian(conf)) if len(conf) else 1.0,
            "min_conf": float(np.nanmin(conf)) if len(conf) else 1.0,
            "pair_conf": [None if np.isnan(v) else float(v) for v in conf],
            "flagged": [int(i) for i in flagged],
            "bad_info": bad_info,
            "level": lvl,
            "level_factor": [float(factor[0]), float(factor[1])],
        }
    )
    st = SliceTransforms(
        shifts=c_level0,
        confidence=conf,
        flagged=[int(i) for i in flagged],
        matrices=matrices,
        subpixel=subpixel,
        method={
            "method": method,
            "offsets": list(offsets),
            "drift": drift,
            "highpass_sigma": highpass_sigma,
            "running_k": running_k,
            "upsample": upsample,
            "per_pair": per_pair,
            "level": lvl,
            "refine_fine": refine_fine,
            "margin": margin,
        },
        source=volume.source,
        stats={
            k: v
            for k, v in stats.items()
            if k not in ("trajectory_before_px", "trajectory_after_px")
        },
        excluded=[int(i) for i in flagged] if bad == "exclude" else [],
    )
    res = AlignResult(st, measurements, lvl, factor, stats)
    log.info(res.summary())
    return res


def _fix_chain_gaps(
    c: np.ndarray, measurements: list[dict[str, Any]], flagged: list[int], n: int
) -> np.ndarray:
    """After dropping pairs touching flagged slices, chain() has a zero step there; bridge the
    gap with the k=2 (or k=4) measurement that jumps over the flagged slice when available."""
    if not flagged:
        return c
    bad = set(flagged)
    by_pair = {(m["i"], m["j"]): np.asarray(m["shift"], dtype=float) for m in measurements}
    out = c.copy()
    for j in range(1, n):
        if j in bad or (j - 1) in bad:
            # step j-1 -> j was dropped; find a bridge from the last good slice before the run
            i = j - 1
            while i in bad and i > 0:
                i -= 1
            if j in bad:
                continue
            bridge = by_pair.get((i, j))
            if bridge is not None:
                out[j:] += (out[i] + bridge - out[j])[None, :]
    return interpolate_over(out, flagged)


def _refine_fine(
    volume: Volume, lvl: int, channel: int, c: np.ndarray, upsample: int, gpu: bool
) -> np.ndarray:
    """Re-measure k=1 residuals at the next finer level with the coarse shift pre-applied on a
    central tile (plan §4 step 5)."""
    from ..phantom.generate import shift_int

    fine = lvl - 1
    data = volume.level_data(fine)[channel]
    n = int(data.shape[0])
    fc = volume.level_voxel_size_nm(lvl)
    ff = volume.level_voxel_size_nm(fine)
    ratio = np.array([fc[1] / ff[1], fc[2] / ff[2]])
    Y, X = data.shape[1:]
    ty, tx = min(Y, 1024), min(X, 1024)
    y0, x0 = (Y - ty) // 2, (X - tx) // 2
    prev = None
    out = c.copy()
    for j in range(n):
        s = np.asarray(data[j].compute())
        if prev is not None:
            d = (out[j] - out[j - 1]) * ratio  # relative coarse correction, fine px
            pre = shift_int(s, int(round(d[0])), int(round(d[1])), fill=float(s.mean()))
            resid, conf = phase_shift(
                prev[y0 : y0 + ty, x0 : x0 + tx], pre[y0 : y0 + ty, x0 : x0 + tx], upsample, gpu=gpu
            )
            if conf > 0.2 and np.abs(resid).max() < 4:
                out[j:] += (resid / ratio)[None, :]
        prev = s
    return out


def _per_pair_matrices(
    get_slice, n: int, model: str, c: np.ndarray, factor, flagged
) -> np.ndarray | None:
    """Chain per-pair rigid/affine matrices (level px -> level-0 px). Experimental (plan §4 step 2)."""
    mats = np.tile(np.eye(3), (n, 1, 1))
    prev = None
    S = np.diag([factor[0], factor[1], 1.0])
    Sinv = np.linalg.inv(S)
    bad = set(flagged)
    for j in range(n):
        s = get_slice(j)
        if prev is not None and j not in bad and (j - 1) not in bad:
            M, n_in = rigid_pair(prev, s, model="euclidean" if model == "rigid" else "affine")
            if M is None:
                M = np.eye(3)
                M[:2, 2] = c[j] - c[j - 1]
            mats[j] = mats[j - 1] @ M
        elif prev is not None:
            M = np.eye(3)
            M[:2, 2] = c[j] - c[j - 1]
            mats[j] = mats[j - 1] @ M
        prev = s
    return np.einsum("ij,njk,kl->nil", S, mats, Sinv)


def verify_stack(volume: Volume, **kw: Any) -> AlignResult:
    """Check an 'already aligned' stack: same measurements, drift removed, nothing applied."""
    kw.setdefault("method", "multi")
    kw.setdefault("drift", "keep")
    res = align_stack(volume, **kw)
    s = res.stats
    verdict = (
        "aligned"
        if s["jitter_rms_px_before"] < 1.0 and not res.transforms.flagged
        else "needs alignment"
    )
    s["verdict"] = verdict
    log.info(
        "verify: %s (jitter RMS %.2f px, flagged %s)",
        verdict,
        s["jitter_rms_px_before"],
        res.transforms.flagged,
    )
    return res


def plot_alignment(result: AlignResult, path: str) -> str | None:
    """Trajectory (before/after) and pair-confidence plots; needs matplotlib (optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional
        return None
    s = result.stats
    before = np.asarray(s["trajectory_before_px"])
    after = np.asarray(s["trajectory_after_px"])
    conf = [np.nan if v is None else v for v in s["pair_conf"]]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for a, name in enumerate(("dy", "dx")):
        ax[a].plot(before[:, a], label="before (measured)")
        ax[a].plot(after[:, a], label="after correction")
        ax[a].set_ylabel(f"{name} (px)")
        ax[a].legend(loc="best")
    ax[2].plot(np.arange(1, len(conf) + 1), conf, ".-")
    for fz in result.transforms.flagged:
        ax[2].axvline(fz, color="r", alpha=0.4)
    ax[2].set_ylabel("pair NCC")
    ax[2].set_xlabel("slice")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
