"""Turning pairwise shifts into a per-slice trajectory (plan §4 step 3, the drift problem).

Conventions: a measurement ``(i, j, d)`` says "moving slice j by ``d`` puts it onto slice i".
With ``c[i]`` the correction applied to slice i (so all slices land in slice 0's frame),
``d = c[j] - c[i]``.

Three strategies are offered:
  (a) running mean: sequential registration to the mean of the previous k aligned slices
  (b) chain + high-pass: chain the k=1 shifts, then keep only the fast (jitter) part of the
      cumulative trajectory so slow drift survives as real geometry
  (c) multi-neighbour consensus: least-squares solve for c from all (i, i+k) measurements,
      weighted by confidence; robust to a single bad slice
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def chain(measurements: list[dict[str, Any]], n: int) -> np.ndarray:
    """Cumulative trajectory from the k=1 measurements only: c[0] = 0, c[j] = c[i] + d_ij."""
    c = np.zeros((n, 2))
    d1 = {m["j"]: np.asarray(m["shift"], dtype=float) for m in measurements if m["k"] == 1}
    for j in range(1, n):
        c[j] = c[j - 1] + d1.get(j, np.zeros(2))
    return c


def highpass(traj: np.ndarray, sigma_slices: float) -> np.ndarray:
    """Jitter-only correction: trajectory minus its slow (Gaussian-smoothed) component."""
    if sigma_slices <= 0:
        return traj.copy()
    slow = gaussian_filter1d(traj, sigma_slices, axis=0, mode="nearest")
    return traj - slow


def solve_multi_neighbour(
    measurements: list[dict[str, Any]],
    n: int,
    weights: np.ndarray | None = None,
    min_conf: float = 0.0,
) -> np.ndarray:
    """Weighted least squares for c (n, 2) with gauge c[0] = 0 from all measurements."""
    rows, cols, vals, rhs, w = [], [], [], [], []
    r = 0
    for m in measurements:
        conf = float(m.get("conf", 1.0))
        if conf < min_conf:
            continue
        i, j = int(m["i"]), int(m["j"])
        wt = max(conf, 1e-3) if weights is None else float(weights[r])
        rows += [r, r]
        cols += [j, i]
        vals += [1.0, -1.0]
        rhs.append(np.asarray(m["shift"], dtype=float))
        w.append(wt)
        r += 1
    if r == 0:
        return np.zeros((n, 2))
    # gauge: c[0] = 0 (strong weight)
    rows.append(r)
    cols.append(0)
    vals.append(1.0)
    rhs.append(np.zeros(2))
    w.append(1e3)
    r += 1
    W = np.asarray(w)
    A = coo_matrix((np.asarray(vals) * W[np.asarray(rows)], (rows, cols)), shape=(r, n)).tocsr()
    B = np.asarray(rhs) * W[:, None]
    c = np.zeros((n, 2))
    for a in range(2):
        c[:, a] = lsqr(A, B[:, a], atol=1e-10, btol=1e-10, iter_lim=10000)[0]
    c -= c[0]
    return c


def running_mean_corrections(
    get_slice,
    n: int,
    k: int = 5,
    upsample: int = 10,
    highpass_sigma: float = 8.0,
    gpu: bool = False,
    initial: np.ndarray | None = None,
    skip: list[int] | None = None,
    min_conf: float = 0.3,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential registration of slice j to the mean of the previous k *aligned* slices.

    Slices in ``skip`` (flagged bad) and slices whose residual confidence is below
    ``min_conf`` keep their initial estimate and never enter the reference mean.

    Aligned slices live on a padded canvas and the reference is the mean over valid
    (content-covered) pixels only, so padding never enters the correlation. Each slice is
    first placed at its chained estimate (``initial``) and only the residual is measured on
    the overlap, which keeps the estimate on the right correlation peak under large drift.
    """
    from .pairwise import phase_shift

    c = np.zeros((n, 2))
    conf = np.zeros(max(n - 1, 0))
    first = np.asarray(get_slice(0), dtype=np.float32)
    H, W = first.shape
    guess = np.zeros((n, 2)) if initial is None else np.asarray(initial, dtype=float)
    m = int(np.ceil(np.abs(guess).max())) + 4 if n > 1 else 4
    m = max(m, 4)
    CH, CW = H + 2 * m, W + 2 * m
    canvases: list[np.ndarray] = []  # NaN outside content

    def place(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.full((CH, CW), np.nan, dtype=np.float32)
        y0, x0 = m + dy, m + dx
        y0c, x0c = max(y0, 0), max(x0, 0)
        y1c, x1c = min(y0 + H, CH), min(x0 + W, CW)
        if y1c > y0c and x1c > x0c:
            out[y0c:y1c, x0c:x1c] = img[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0]
        return out

    canvases.append(place(first, 0, 0))
    bad = set(skip or [])
    for j in range(1, n):
        s = np.asarray(get_slice(j), dtype=np.float32)
        if j in bad:
            c[j] = guess[j]
            continue
        stack = np.stack(canvases[-k:])
        with np.errstate(invalid="ignore"):
            ref = np.nanmean(stack, axis=0)
        gy, gx = int(round(guess[j, 0])), int(round(guess[j, 1]))
        placed = place(s, gy, gx)
        valid = ~np.isnan(ref) & ~np.isnan(placed)
        if valid.sum() < 256:
            c[j] = guess[j]
            canvases.append(place(s, gy, gx))
            continue
        ys, xs = np.where(valid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rc = ref[y0:y1, x0:x1]
        pc = placed[y0:y1, x0:x1]
        # fill the few remaining NaNs (irregular overlap corners) with the mean
        rc = np.where(np.isnan(rc), np.nanmean(rc), rc)
        pc = np.where(np.isnan(pc), np.nanmean(pc), pc)
        d, cf = phase_shift(rc, pc, upsample, highpass_sigma, gpu)
        conf[j - 1] = cf
        if cf < min_conf or np.abs(d).max() > max(4.0, 0.2 * min(H, W)):
            c[j] = guess[j]
            continue
        c[j] = np.array([gy, gx], dtype=float) + d
        canvases.append(place(s, int(round(c[j, 0])), int(round(c[j, 1]))))
        if len(canvases) > k:
            canvases.pop(0)
        if progress is not None:
            progress(j + 1, n)
    return c, conf


def trajectory_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """Numbers for the report and the trajectory plot."""
    d_before = np.diff(before, axis=0) if len(before) > 1 else np.zeros((0, 2))
    d_after = np.diff(after, axis=0) if len(after) > 1 else np.zeros((0, 2))
    return {
        "n": int(len(before)),
        "jitter_rms_px_before": float(np.sqrt(np.mean(d_before**2))) if len(d_before) else 0.0,
        "jitter_rms_px_after": float(np.sqrt(np.mean(d_after**2))) if len(d_after) else 0.0,
        "max_abs_correction_px": float(np.abs(after).max()) if len(after) else 0.0,
        "trajectory_before_px": before.tolist(),
        "trajectory_after_px": after.tolist(),
    }
