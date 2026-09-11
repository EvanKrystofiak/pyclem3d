"""Intensity registration of a synthetic (segmentation-derived) EM volume against a real
fluorescence channel with SimpleITK (plan Phase 6 'intensity-based refinement').

Convention: the EM-side synthetic volume is the ITK *fixed* image and the LM channel the
*moving* image, so the optimised ITK transform maps EM world -> LM world, which is exactly
the inverse map pyclem3d's resampler needs. The returned :class:`AffineTransform` is the
LM -> EM forward map (its inverse), matching the rest of the package; a B-spline stage is
returned as a :class:`DisplacementGrid` (EM -> LM) ready for the deformable overlay/export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..export.transforms import zyx_to_xyz
from ..io.volume import Volume
from ..register.transforms import AffineTransform
from ..resample.grid import DisplacementGrid, Grid
from ..resample.lazy import resample_to_grid

log = logging.getLogger(__name__)


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError(
            "intensity registration needs SimpleITK: uv sync --extra itk (or --extra seg)"
        ) from e
    return sitk


def _to_sitk(vol: Volume, channel: int = 0, dtype=np.float32):
    """Axis-aligned Volume -> SimpleITK image with physical coordinates = world nm (xyz)."""
    sitk = _sitk()
    A = vol.world_affine
    off = A[:3, :3] - np.diag(np.diag(A[:3, :3]))
    if np.abs(off).max() > 1e-6 * max(1.0, np.abs(A).max()):
        raise ValueError(
            "intensity registration needs an axis-aligned volume (apply the pre-align first)"
        )
    arr = np.asarray(vol.data[channel].compute(), dtype=dtype)
    img = sitk.GetImageFromArray(arr)
    d = np.diag(A)[:3]
    img.SetSpacing(tuple(float(abs(v)) for v in d[::-1]))
    img.SetOrigin(tuple(float(v) for v in A[:3, 3][::-1]))
    sign = np.sign(d)[::-1]
    img.SetDirection(tuple(float(v) for v in np.diag(sign).ravel()))
    return img


def _affine_to_sitk(t: AffineTransform):
    """4x4 zyx (world -> world) -> sitk.AffineTransform (xyz)."""
    sitk = _sitk()
    M = zyx_to_xyz(t.matrix)
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(tuple(float(v) for v in M[:3, :3].ravel()))
    tx.SetTranslation(tuple(float(v) for v in M[:3, 3]))
    tx.SetCenter((0.0, 0.0, 0.0))
    return tx


def _sitk_to_affine(tx) -> AffineTransform:
    """sitk affine (xyz, with centre) -> 4x4 zyx AffineTransform."""
    M3 = np.asarray(tx.GetMatrix(), dtype=float).reshape(3, 3)
    c = np.asarray(tx.GetCenter(), dtype=float)
    t = np.asarray(tx.GetTranslation(), dtype=float)
    M = np.eye(4)
    M[:3, :3] = M3
    M[:3, 3] = t + c - M3 @ c
    return AffineTransform(zyx_to_xyz(M), "affine")


@dataclass
class IntensityResult:
    lm_to_em: AffineTransform
    metric_before: float
    metric_after: float
    iterations: int
    stop: str
    displacement: DisplacementGrid | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        d = self.lm_to_em.decompose()
        s = f"intensity affine: metric {self.metric_before:.4f} -> {self.metric_after:.4f} ({self.iterations} it, {self.stop}); scales {np.round(d['scales'], 3).tolist()}, rotation {d['rotation_deg']:.2f} deg"
        if self.displacement is not None:
            s += f"; bspline displacement grid {self.displacement.values.shape[:3]}"
        return s


def fixed_mask_from_footprint(
    fixed: Volume, moving: Volume, lm_to_em: AffineTransform, margin_nm: float = 500.0
):
    """uint8 sitk mask on the fixed grid: voxels whose EM world position maps inside the LM."""
    sitk = _sitk()
    grid = Grid.from_volume(fixed, 0)
    Z, Y, X = grid.shape_zyx
    world = grid.block_world_coords(0, Z, 0, Y, 0, X)
    lm_world = lm_to_em.inverse().apply(world)
    bb = moving.bbox_world()
    inside = np.all((lm_world >= bb[0] - margin_nm) & (lm_world <= bb[1] + margin_nm), axis=1)
    m = inside.reshape(Z, Y, X).astype(np.uint8)
    img = sitk.GetImageFromArray(m)
    ref = _to_sitk(fixed)
    img.CopyInformation(ref)
    return img, float(m.mean())


def coverage_on_synthetic(
    fixed: Volume, moving: Volume, channel: int, lm_to_em: AffineTransform
) -> np.ndarray:
    """Boolean (Z, Y, X) array of synthetic voxels the LM covers under ``lm_to_em``."""
    grid = Grid.from_volume(fixed, 0)
    _, cov = resample_to_grid(moving, lm_to_em, grid, channels=[channel], chunks=grid.shape_zyx)
    return np.asarray(cov.compute()) > 0


def ncc_on_synthetic(
    fixed: Volume,
    moving: Volume,
    channel: int,
    lm_to_em: AffineTransform,
    region: np.ndarray | None = None,
) -> float:
    """Normalized correlation between the synthetic volume and the LM channel resampled onto it.

    By default the correlation runs over the voxels the LM covers under this transform. That
    rewards shrinking the LM slab (its dim edge planes drop out of the sum), so comparisons
    between transforms of different extent should pass a reference ``region`` (a boolean array
    on the synthetic grid, e.g. :func:`coverage_on_synthetic` at a reference transform): the
    sum then runs over the union of that region and the current coverage, so voxels the slab
    no longer covers count as zero signal and voxels it newly covers must earn their keep.
    """
    grid = Grid.from_volume(fixed, 0)
    data, cov = resample_to_grid(moving, lm_to_em, grid, channels=[channel], chunks=grid.shape_zyx)
    covered = np.asarray(cov.compute()) > 0
    lm = np.where(covered, np.asarray(data[0].compute(), dtype=np.float32), np.float32(0))
    c = covered if region is None else (np.asarray(region, bool) | covered)
    if c.sum() < 100 or (c & covered).sum() < 100:
        return float("nan")
    a = np.asarray(fixed.data[0].compute(), dtype=np.float32)[c]
    b = lm[c]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def z_scan(
    fixed: Volume,
    moving: Volume,
    channel: int,
    lm_to_em: AffineTransform,
    offsets_nm: np.ndarray,
    z_scales: tuple[float, ...] = (1.0,),
) -> list[dict[str, float]]:
    """NCC as a function of an extra EM-z offset (and z scale) applied to ``lm_to_em``.

    This is the cheapest check of whether z is recoverable: a peaked curve means the
    organelle pattern pins the confocal slab in depth; a flat one means it does not.
    At each offset every z scale is scored over the union of the unscaled slab's voxels and
    its own, so a smaller scale is not rewarded merely for dropping the slab's edge planes
    and a larger one has to match where it grows.
    """
    out = []
    M0 = lm_to_em.matrix
    for off in offsets_nm:
        M1 = M0.copy()
        M1[0, 3] = M0[0, 3] + float(off)
        region = None
        if len(z_scales) > 1:
            region = coverage_on_synthetic(fixed, moving, channel, AffineTransform(M1))
        for zs in z_scales:
            M = M1.copy()
            M[0, :3] *= zs
            out.append(
                {
                    "offset_nm": float(off),
                    "z_scale": float(zs),
                    "ncc": ncc_on_synthetic(fixed, moving, channel, AffineTransform(M), region),
                }
            )
    return out


@dataclass
class Cue:
    """One segmentation-derived cue for the multi-cue refinement.

    ``synthetic`` is a Volume in EM world (a blurred mask or probability), ``channel`` the LM
    channel it should match, ``sign`` +1 when the organelle is bright in that channel and -1
    when it is an exclusion (a nucleus in a cytoplasmic channel), ``weight`` its share of the
    objective, ``region`` an optional reference boolean voxel set on the synthetic grid the
    NCC always includes (see :func:`ncc_on_synthetic`).
    """

    name: str
    synthetic: Volume
    channel: int
    sign: float = 1.0
    weight: float = 1.0
    region: np.ndarray | None = None


def multi_cue_ncc(cues: list[Cue], moving: Volume, lm_to_em: AffineTransform) -> dict[str, float]:
    """Per-cue NCC (signed) and their weighted sum for one transform."""
    out: dict[str, float] = {}
    total = 0.0
    wsum = 0.0
    for c in cues:
        n = ncc_on_synthetic(c.synthetic, moving, c.channel, lm_to_em, c.region)
        n = 0.0 if not np.isfinite(n) else n
        out[c.name] = float(c.sign * n)
        total += c.weight * c.sign * n
        wsum += c.weight
    out["combined"] = float(total / wsum) if wsum else float("nan")
    return out


def refine_affine_multicue(
    cues: list[Cue],
    moving: Volume,
    initial_lm_to_em: AffineTransform,
    linear_step: float = 0.02,
    translation_step_nm: float = 200.0,
    maxfev: int = 600,
    xtol: float = 1e-3,
) -> tuple[AffineTransform, dict[str, Any]]:
    """Derivative-free (Powell) refinement of a 3D affine on a weighted sum of cue NCCs.

    ``register-seg`` runs this as its final polish on the mask cue alone; extra cues let an
    exclusion (sign -1) pull on the fit alongside a bright one. The 12 parameters are offsets
    from ``initial_lm_to_em`` scaled so one unit is ``linear_step`` on the linear part and
    ``translation_step_nm`` on the translation. Returns the refined transform and a dict
    with the per-cue NCCs before/after and the evaluation count. Cues without a ``region``
    are scored over the voxels the LM covers at ``initial_lm_to_em`` (united with the
    current coverage) throughout, so the search cannot improve its score by shrinking the
    slab or by growing it into voxels it does not match.
    """
    from dataclasses import replace

    from scipy.optimize import minimize

    cues = [
        c
        if c.region is not None
        else replace(
            c, region=coverage_on_synthetic(c.synthetic, moving, c.channel, initial_lm_to_em)
        )
        for c in cues
    ]
    M0 = initial_lm_to_em.matrix.copy()
    scale = np.concatenate([np.full(9, linear_step), np.full(3, translation_step_nm)])
    evals: list[float] = []

    def unpack(x: np.ndarray) -> AffineTransform:
        M = M0.copy()
        d = x * scale
        M[:3, :3] += d[:9].reshape(3, 3)
        M[:3, 3] += d[9:]
        return AffineTransform(M, "affine-multicue")

    def objective(x: np.ndarray) -> float:
        v = multi_cue_ncc(cues, moving, unpack(x))["combined"]
        evals.append(v)
        return -v

    x0 = np.zeros(12)
    before = multi_cue_ncc(cues, moving, initial_lm_to_em)
    res = minimize(
        objective, x0, method="Powell", options={"maxfev": int(maxfev), "xtol": xtol, "ftol": 1e-5}
    )
    T = unpack(res.x)
    after = multi_cue_ncc(cues, moving, T)
    info = {
        "before": before,
        "after": after,
        "n_eval": len(evals),
        "success": bool(res.success),
        "message": str(res.message),
    }
    log.info(
        "multi-cue refine: %s -> %s (%d evaluations)",
        {k: round(v, 4) for k, v in before.items()},
        {k: round(v, 4) for k, v in after.items()},
        len(evals),
    )
    return T, info


def register_affine(
    fixed: Volume,
    moving: Volume,
    channel: int,
    initial_lm_to_em: AffineTransform,
    metric: str = "correlation",
    shrink: tuple[int, ...] = (4, 2, 1),
    sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
    iterations: int = 300,
    learning_rate: float = 1.0,
    sampling: float = 0.25,
    mask_margin_nm: float = 500.0,
    seed: int = 0,
) -> IntensityResult:
    """Optimise a 3D affine (EM->LM in ITK) from ``initial_lm_to_em``; returns the LM->EM map."""
    sitk = _sitk()
    f_img = _to_sitk(fixed)
    m_img = _to_sitk(moving, channel)
    tx = _affine_to_sitk(initial_lm_to_em.inverse())
    mask, frac = fixed_mask_from_footprint(fixed, moving, initial_lm_to_em, mask_margin_nm)
    reg = sitk.ImageRegistrationMethod()
    if metric == "mi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling), int(seed))
    reg.SetMetricFixedMask(mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=float(learning_rate),
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=20,
        estimateLearningRate=reg.EachIteration,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(list(shrink))
    reg.SetSmoothingSigmasPerLevel(list(sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(tx, inPlace=True)
    before = float(reg.MetricEvaluate(f_img, m_img))
    final = reg.Execute(f_img, m_img)
    after = float(reg.GetMetricValue())
    em_to_lm = _sitk_to_affine(final)
    res = IntensityResult(
        lm_to_em=em_to_lm.inverse(),
        metric_before=before,
        metric_after=after,
        iterations=int(reg.GetOptimizerIteration()),
        stop=str(reg.GetOptimizerStopConditionDescription()),
        info={
            "metric": metric,
            "mask_fraction": frac,
            "shrink": list(shrink),
            "sigmas": list(sigmas),
        },
    )
    res.lm_to_em.kind = "affine-intensity"
    log.info(res.summary())
    return res


def register_bspline(
    fixed: Volume,
    moving: Volume,
    channel: int,
    lm_to_em: AffineTransform,
    grid_spacing_nm: float = 1500.0,
    iterations: int = 100,
    sampling: float = 0.25,
    mask_margin_nm: float = 500.0,
    displacement_spacing_nm: float = 500.0,
    seed: int = 0,
) -> tuple[DisplacementGrid, dict[str, Any]]:
    """B-spline refinement on top of the affine; returns the EM->LM displacement grid."""
    sitk = _sitk()
    f_img = _to_sitk(fixed)
    m_img = _to_sitk(moving, channel)
    affine_tx = _affine_to_sitk(lm_to_em.inverse())
    size = np.asarray(f_img.GetSize()) * np.asarray(f_img.GetSpacing())
    mesh = [int(max(1, round(s / grid_spacing_nm))) for s in size]
    bspline = sitk.BSplineTransformInitializer(f_img, mesh, order=3)
    composite = sitk.CompositeTransform([affine_tx, bspline])
    mask, frac = fixed_mask_from_footprint(fixed, moving, lm_to_em, mask_margin_nm)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling), int(seed))
    reg.SetMetricFixedMask(mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=int(iterations),
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=int(iterations) * 5,
        costFunctionConvergenceFactor=1e7,
    )
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetInitialTransform(composite, inPlace=True)
    before = float(reg.MetricEvaluate(f_img, m_img))
    final = reg.Execute(f_img, m_img)
    after = float(reg.GetMetricValue())
    # sample the composite EM->LM map on a coarse world grid -> DisplacementGrid (zyx nm)
    bb = fixed.bbox_world()
    sp = np.full(3, float(displacement_spacing_nm))
    lo = bb[0] - sp
    hi = bb[1] + sp
    n = np.maximum(2, np.ceil((hi - lo) / sp).astype(int) + 1)
    zz, yy, xx = np.meshgrid(*[lo[i] + sp[i] * np.arange(n[i]) for i in range(3)], indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    mapped = np.array(
        [final.TransformPoint((float(p[2]), float(p[1]), float(p[0])))[::-1] for p in pts]
    )
    values = (mapped - pts).reshape(*n, 3).astype(np.float32)
    dg = DisplacementGrid(lo, sp, values, lm_to_em.inverse())
    info = {
        "metric_before": before,
        "metric_after": after,
        "mesh": mesh,
        "stop": str(reg.GetOptimizerStopConditionDescription()),
        "mask_fraction": frac,
    }
    log.info("bspline: metric %.4f -> %.4f, mesh %s", before, after, mesh)
    return dg, info
