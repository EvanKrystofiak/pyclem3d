"""Bad-slice detection and repair (plan §4 step 4): low correlation or outlier shift
(knife chatter, charging, focus loss) -> flag, interpolate the transform across, report."""

from __future__ import annotations

from typing import Any

import numpy as np


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def detect_bad_slices(
    measurements: list[dict[str, Any]],
    n: int,
    conf_threshold: float | None = None,
    shift_k: float = 5.0,
    min_abs_conf: float = 0.15,
) -> tuple[list[int], dict[str, Any]]:
    """Flag slices whose k=1 pairs are outliers in confidence or shift.

    A slice is flagged when *both* of its neighbouring pairs are bad (the slice itself is
    the problem) or when it is the later slice of a single bad pair at the stack edge.
    """
    d1 = sorted((m for m in measurements if m["k"] == 1), key=lambda m: m["j"])
    if not d1:
        return [], {"conf_threshold": None}
    conf = np.array([m["conf"] for m in d1])
    shifts = np.array([m["shift"] for m in d1], dtype=float)
    mag = np.linalg.norm(shifts, axis=1)
    med_conf = float(np.median(conf))
    if conf_threshold is None:
        # 3 MAD below the median, but at least a 25% drop: on a clean stack the spread of
        # pair confidences is tiny and a pure MAD rule flags ordinary slices
        conf_threshold = max(med_conf - max(3.0 * _mad(conf), 0.25 * med_conf), min_abs_conf)
        conf_threshold = min(conf_threshold, 0.9)
    mad_shift = _mad(mag)
    shift_threshold = float(np.median(mag) + shift_k * max(mad_shift, 2.0))
    # a shift outlier alone can be a real stage jump; it counts only with a confidence drop
    bad_pair = (conf < conf_threshold) | ((mag > shift_threshold) & (conf < med_conf))
    flagged: set[int] = set()
    for idx, m in enumerate(d1):
        if not bad_pair[idx]:
            continue
        j = m["j"]
        prev_bad = idx > 0 and bad_pair[idx - 1]
        next_bad = idx + 1 < len(d1) and bad_pair[idx + 1]
        if prev_bad or next_bad:
            # runs of bad pairs: the slices between them are bad
            if prev_bad:
                flagged.add(m["i"])
            if next_bad:
                flagged.add(j)
        else:
            # isolated bad pair: one of i, j is bad; the later one unless it is the last slice
            flagged.add(j if j < n - 1 else m["i"])
    info = {
        "conf_threshold": float(conf_threshold),
        "shift_threshold_px": shift_threshold,
        "bad_pairs": [int(d1[i]["j"]) for i in np.where(bad_pair)[0]],
        "pair_conf": conf.tolist(),
        "pair_shift_mag": mag.tolist(),
    }
    return sorted(flagged), info


def interpolate_over(traj: np.ndarray, flagged: list[int]) -> np.ndarray:
    """Linearly interpolate the trajectory across flagged slices from their good neighbours."""
    if not flagged:
        return traj.copy()
    n = len(traj)
    good = np.ones(n, dtype=bool)
    good[[f for f in flagged if 0 <= f < n]] = False
    if good.sum() < 2:
        return traj.copy()
    idx = np.arange(n)
    out = traj.copy()
    for a in range(traj.shape[1]):
        out[~good, a] = np.interp(idx[~good], idx[good], traj[good, a])
    return out


def drop_measurements_touching(
    measurements: list[dict[str, Any]], flagged: list[int]
) -> list[dict[str, Any]]:
    bad = set(flagged)
    return [m for m in measurements if m["i"] not in bad and m["j"] not in bad]
