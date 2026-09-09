"""Pairwise slice registration (plan §4 step 1-2): phase correlation with a confidence score,
optional per-pair rigid/affine via ORB + RANSAC. CPU by default; GPU (cupy) drop-in for the FFT."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

log = logging.getLogger(__name__)

try:  # optional GPU path
    import cupy as _cp  # type: ignore

    HAVE_CUPY = True
except Exception:  # pragma: no cover
    _cp = None
    HAVE_CUPY = False


def preprocess(img: np.ndarray, highpass_sigma: float = 8.0, window: bool = True) -> np.ndarray:
    """float32, high-passed (removes illumination/charging gradients), Hann-windowed."""
    f = np.asarray(img, dtype=np.float32)
    if highpass_sigma and highpass_sigma > 0:
        f = f - gaussian_filter(f, highpass_sigma)
    f -= f.mean()
    s = f.std()
    if s > 0:
        f /= s
    if window:
        wy = np.hanning(f.shape[0]).astype(np.float32)
        wx = np.hanning(f.shape[1]).astype(np.float32)
        f = f * wy[:, None] * wx[None, :]
    return f


def ncc_after_shift(ref: np.ndarray, mov: np.ndarray, dy: float, dx: float) -> float:
    """Normalized cross-correlation of the overlap after moving ``mov`` by (dy, dx) (integer)."""
    dy_i, dx_i = int(round(dy)), int(round(dx))
    H, W = ref.shape
    ys = slice(max(dy_i, 0), H + min(dy_i, 0))
    xs = slice(max(dx_i, 0), W + min(dx_i, 0))
    ys_m = slice(max(-dy_i, 0), H + min(-dy_i, 0))
    xs_m = slice(max(-dx_i, 0), W + min(-dx_i, 0))
    a = ref[ys, xs].astype(np.float64)
    b = mov[ys_m, xs_m].astype(np.float64)
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _phase_shift_gpu(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    cp = _cp
    R = cp.fft.fft2(cp.asarray(ref))
    M = cp.fft.fft2(cp.asarray(mov))
    cross = R * cp.conj(M)
    cross /= cp.maximum(cp.abs(cross), 1e-12)
    corr = cp.real(cp.fft.ifft2(cross))
    idx = int(cp.argmax(corr))
    py, px = np.unravel_index(idx, corr.shape)
    c = corr

    def parabolic(i, n, get):
        m = get(i)
        lo, hi = get((i - 1) % n), get((i + 1) % n)
        den = lo - 2 * m + hi
        return 0.0 if den == 0 else 0.5 * (lo - hi) / den

    dy = py + parabolic(py, corr.shape[0], lambda i: float(c[i % corr.shape[0], px]))
    dx = px + parabolic(px, corr.shape[1], lambda i: float(c[py, i % corr.shape[1]]))
    if dy > corr.shape[0] / 2:
        dy -= corr.shape[0]
    if dx > corr.shape[1] / 2:
        dx -= corr.shape[1]
    return float(dy), float(dx)


def _overlap_crops(
    ref: np.ndarray, mov: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crops of ref and mov covering the same content once mov is moved by (dy, dx)."""
    H, W = ref.shape
    ys = slice(max(dy, 0), H + min(dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    ys_m = slice(max(-dy, 0), H + min(-dy, 0))
    xs_m = slice(max(-dx, 0), W + min(-dx, 0))
    return ref[ys, xs], mov[ys_m, xs_m]


def _estimate(r: np.ndarray, m: np.ndarray, upsample: int, gpu: bool) -> np.ndarray:
    if gpu and HAVE_CUPY:
        dy, dx = _phase_shift_gpu(r, m)
        return np.array([dy, dx])
    from skimage.registration import phase_cross_correlation

    shift, _err, _ph = phase_cross_correlation(
        r, m, upsample_factor=max(1, int(upsample)), normalization="phase"
    )
    return np.asarray(shift, dtype=float)


def phase_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    upsample: int = 10,
    highpass_sigma: float = 8.0,
    gpu: bool = False,
    max_shift: float | None = None,
    two_pass: bool = True,
) -> tuple[np.ndarray, float]:
    """Shift (dy, dx) to apply to ``mov`` to register it onto ``ref``, and an NCC confidence.

    Two passes: a first estimate on the whole (windowed) frames, then, when the shift is
    at least a pixel, a second estimate of the residual on the overlapping crops after the
    integer shift. The second pass removes the bias that windowing introduces for shifts
    that are a noticeable fraction of the frame.
    """
    ref = np.asarray(ref, dtype=np.float32)
    mov = np.asarray(mov, dtype=np.float32)
    r = preprocess(ref, highpass_sigma)
    m = preprocess(mov, highpass_sigma)
    shift = _estimate(r, m, upsample, gpu)
    if two_pass and np.abs(shift).max() >= 1.0:
        dy, dx = int(round(shift[0])), int(round(shift[1]))
        rc, mc = _overlap_crops(ref, mov, dy, dx)
        if min(rc.shape) >= 16:
            resid = _estimate(
                preprocess(rc, highpass_sigma), preprocess(mc, highpass_sigma), upsample, gpu
            )
            if np.abs(resid).max() <= 2.0:
                shift = np.array([dy, dx], dtype=float) + resid
    r_hp = preprocess(ref, highpass_sigma, window=False)
    m_hp = preprocess(mov, highpass_sigma, window=False)
    if max_shift is not None and np.abs(shift).max() > max_shift:
        conf = 0.0
    else:
        conf = ncc_after_shift(r_hp, m_hp, shift[0], shift[1])
    return shift, conf


def rigid_pair(
    ref: np.ndarray,
    mov: np.ndarray,
    model: str = "euclidean",
    n_keypoints: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray | None, int]:
    """Per-pair rigid ('euclidean'), 'similarity' or 'affine' 3x3 (y, x) via ORB + RANSAC.

    Returns (matrix mapping mov -> ref, n_inliers); (None, 0) when matching fails.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, EuclideanTransform, SimilarityTransform

    r = preprocess(ref, window=False)
    m = preprocess(mov, window=False)
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)
    try:
        orb.detect_and_extract(r)
        kr, dr = orb.keypoints, orb.descriptors
        orb.detect_and_extract(m)
        km, dm = orb.keypoints, orb.descriptors
    except (RuntimeError, ValueError):
        return None, 0
    if len(kr) < 8 or len(km) < 8:
        return None, 0
    matches = match_descriptors(dr, dm, cross_check=True, max_ratio=0.85)
    if len(matches) < 6:
        return None, 0
    src = km[matches[:, 1]][:, ::-1]  # skimage transforms want (x, y)
    dst = kr[matches[:, 0]][:, ::-1]
    cls = {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
    }[model]
    try:
        tf, inliers = ransac(
            (src, dst), cls, min_samples=3, residual_threshold=2.0, max_trials=500, rng=seed
        )
    except Exception:
        return None, 0
    if tf is None or inliers is None or inliers.sum() < 6:
        return None, 0
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # (x,y) <-> (y,x)
    M_yx = P @ np.asarray(tf.params) @ P
    return M_yx, int(inliers.sum())


def matrix_to_shift(M: np.ndarray) -> np.ndarray:
    return np.array([M[0, 2], M[1, 2]], dtype=float)


def pairwise_measurements(
    get_slice,
    n: int,
    offsets: tuple[int, ...] = (1,),
    upsample: int = 10,
    highpass_sigma: float = 8.0,
    gpu: bool = False,
    max_shift: float | None = None,
    progress=None,
) -> list[dict[str, Any]]:
    """Measure shifts between slice i and i+k for k in offsets, reading each slice once.

    ``get_slice(i)`` returns a 2D array (at the chosen pyramid level). Returns a list of
    ``{"i": i, "j": j, "shift": (dy, dx), "conf": c}`` where ``shift`` moves slice j onto slice i.
    """
    maxk = max(offsets)
    buf: dict[int, np.ndarray] = {}
    out: list[dict[str, Any]] = []
    for j in range(n):
        buf[j] = np.asarray(get_slice(j))
        for k in offsets:
            i = j - k
            if i < 0:
                continue
            shift, conf = phase_shift(buf[i], buf[j], upsample, highpass_sigma, gpu, max_shift)
            out.append({"i": i, "j": j, "k": k, "shift": shift, "conf": conf})
        for old in [q for q in buf if q < j - maxk]:
            del buf[old]
        if progress is not None:
            progress(j + 1, n)
    return out
