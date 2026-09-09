"""Synthetic phantoms with known truth (plan §11).

* ``make_phantom``: an EM block with nucleus-like ellipsoids and small organelle-like
  blobs, plus a "confocal" of the same scene seen through a known anisotropic affine
  (shrinkage + tilt + z scaling), an anisotropic PSF blur, coarser voxels and noise.
  The EM block is a sub-region of the larger confocal field, as in real CLEM.
* ``make_misaligned_stack``: per-slice jitter + slow drift + a few bad slices applied
  to an EM stack, with the truth, for the alignment module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from ..io.readers import volume_from_array
from ..io.volume import Channel, Volume, apply_affine


@dataclass
class PhantomTruth:
    em_voxel_nm: tuple[float, float, float]
    lm_voxel_nm: tuple[float, float, float]
    lm_to_em: np.ndarray  # 4x4, LM world nm -> EM world nm
    nuclei_em_world: np.ndarray  # (N, 3)
    nuclei_lm_world: np.ndarray  # (N, 3)
    nuclei_radii_nm: np.ndarray  # (N, 3)
    organelles_em_world: np.ndarray  # (M, 3)
    organelles_lm_world: np.ndarray
    organelle_radii_nm: np.ndarray  # (M,)
    psf_fwhm_nm: tuple[float, float]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def em_to_lm(self) -> np.ndarray:
        return np.linalg.inv(self.lm_to_em)


def _random_lm_to_em(rng: np.random.Generator, shrink=(0.80, 0.88, 0.88), tilt_deg=(4.0, 2.0, 2.0)):
    """EM = R S (LM - offset): anisotropic shrinkage plus small rotations. Returns em_to_lm too."""
    R = Rotation.from_euler(
        "zyx",
        [
            rng.uniform(-tilt_deg[0], tilt_deg[0]),
            rng.uniform(-tilt_deg[1], tilt_deg[1]),
            rng.uniform(-tilt_deg[2], tilt_deg[2]),
        ],
        degrees=True,
    ).as_matrix()
    S = np.diag(shrink)
    M = np.eye(4)
    M[:3, :3] = R @ S
    return M


def _grid_world(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    z, y, x = (np.arange(s, dtype=float) for s in shape)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    vox = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1)
    return apply_affine(affine, vox).reshape(*shape, 3)


def _paint_ellipsoids(
    world: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    value: np.ndarray | float,
    soft: float = 0.0,
) -> np.ndarray:
    """Accumulate ellipsoid membership (soft edge in units of normalized radius)."""
    out = np.zeros(world.shape[:3], dtype=np.float32)
    shape = np.array(world.shape[:3])
    # per-voxel world spacing estimate for bounding boxes
    for k, (c, r) in enumerate(zip(centers, radii)):
        r = np.broadcast_to(np.asarray(r, dtype=float), (3,))
        # bounding box in voxel indices via the world extent of the grid (grid is affine, so use projections)
        d = world - c
        q = np.sqrt(np.sum((d / r) ** 2, axis=-1))
        if soft > 0:
            m = np.clip((1.0 + soft - q) / soft, 0.0, 1.0)
        else:
            m = (q <= 1.0).astype(np.float32)
        v = value[k] if np.ndim(value) else value
        out += (m * v).astype(np.float32)
        del d, q, m
    assert out.shape == tuple(shape)
    return out


def _paint_spheres_boxed(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    value: float,
    soft: float = 0.3,
) -> np.ndarray:
    """Like _paint_ellipsoids but evaluates only inside each sphere's voxel bounding box (fast for many blobs)."""
    out = np.zeros(shape, dtype=np.float32)
    inv = np.linalg.inv(affine)
    vs = np.abs(np.diag(affine)[:3])
    for c, r in zip(centers, radii):
        cv = apply_affine(inv, c)
        rad = (r * (1 + soft)) / vs + 1
        lo = np.maximum(np.floor(cv - rad).astype(int), 0)
        hi = np.minimum(np.ceil(cv + rad).astype(int) + 1, np.array(shape))
        if (hi <= lo).any():
            continue
        sub = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        zz, yy, xx = np.meshgrid(
            *[np.arange(a, b, dtype=float) for a, b in zip(lo, hi)], indexing="ij"
        )
        vox = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
        w = apply_affine(affine, vox)
        q = np.linalg.norm(w - c, axis=1) / r
        m = np.clip((1.0 + soft - q) / soft, 0.0, 1.0) if soft > 0 else (q <= 1).astype(float)
        out[sub] += (m * value).reshape(zz.shape).astype(np.float32)
    return out


def make_phantom(
    em_shape: tuple[int, int, int] = (48, 192, 192),
    em_voxel_nm: tuple[float, float, float] = (20.0, 20.0, 20.0),
    lm_voxel_nm: tuple[float, float, float] = (300.0, 100.0, 100.0),
    n_nuclei: int = 10,
    n_organelles: int = 40,
    seed: int = 0,
    lm_to_em: np.ndarray | None = None,
    psf_fwhm_nm: tuple[float, float] = (800.0, 250.0),
    lm_margin_nm: float = 3000.0,
    noise: float = 0.05,
    em_dtype=np.uint8,
    lm_dtype=np.uint16,
) -> tuple[Volume, Volume, PhantomTruth]:
    """Build (em, lm, truth). Nuclei are dark in EM and bright in LM channel 0; organelles are
    dark dots in EM and bright in LM channel 1."""
    rng = np.random.default_rng(seed)
    em_vs = np.asarray(em_voxel_nm, dtype=float)
    em_extent = em_vs * np.asarray(em_shape)
    em_affine = np.eye(4)
    em_affine[:3, :3] = np.diag(em_vs)

    # scene in EM world nm
    nuc_r = np.stack(
        [
            rng.uniform(0.12, 0.2, n_nuclei) * em_extent[0],
            rng.uniform(0.08, 0.14, n_nuclei) * em_extent[1],
            rng.uniform(0.08, 0.14, n_nuclei) * em_extent[2],
        ],
        axis=1,
    )
    nuc_c = np.stack([rng.uniform(0.15, 0.85, n_nuclei) * em_extent[i] for i in range(3)], axis=1)
    org_r = rng.uniform(3.0, 6.0, n_organelles) * em_vs.max() * 1.5
    org_c = np.stack(
        [rng.uniform(0.05, 0.95, n_organelles) * em_extent[i] for i in range(3)], axis=1
    )

    # ---- EM image
    world_em = _grid_world(em_shape, em_affine)
    nuc_mask = _paint_ellipsoids(world_em, nuc_c, nuc_r, 1.0, soft=0.05)
    org_mask = _paint_spheres_boxed(em_shape, em_affine, org_c, org_r, 1.0, soft=0.3)
    texture = gaussian_filter(rng.normal(0, 1, em_shape).astype(np.float32), 1.5)
    texture /= texture.std() + 1e-6
    em = 0.62 + 0.06 * texture - 0.28 * np.clip(nuc_mask, 0, 1) - 0.35 * np.clip(org_mask, 0, 1)
    # nuclear texture (chromatin) so that centroiding has structure to work with
    em += (
        0.05
        * np.clip(nuc_mask, 0, 1)
        * gaussian_filter(rng.normal(0, 1, em_shape).astype(np.float32), 2.5)
        * 3
    )
    em += rng.normal(0, noise, em_shape).astype(np.float32)
    em = np.clip(em, 0, 1)
    if np.issubdtype(em_dtype, np.integer):
        em_img = (em * np.iinfo(em_dtype).max).astype(em_dtype)
    else:
        em_img = em.astype(em_dtype)

    # ---- transform and LM grid
    if lm_to_em is None:
        M = _random_lm_to_em(rng)
        em_to_lm = np.linalg.inv(M)
        # place the EM block inside the LM field with a margin
        corners = np.array(
            [
                [a, b, c]
                for a in (0, em_extent[0])
                for b in (0, em_extent[1])
                for c in (0, em_extent[2])
            ]
        )
        lm_corners = apply_affine(em_to_lm, corners)
        offset = lm_corners.min(0) - lm_margin_nm
        em_to_lm[:3, 3] -= offset
        lm_to_em = np.linalg.inv(em_to_lm)
    else:
        em_to_lm = np.linalg.inv(lm_to_em)
    corners = np.array(
        [[a, b, c] for a in (0, em_extent[0]) for b in (0, em_extent[1]) for c in (0, em_extent[2])]
    )
    lm_corners = apply_affine(em_to_lm, corners)
    lm_vs = np.asarray(lm_voxel_nm, dtype=float)
    lm_shape = tuple(
        int(np.ceil((lm_corners.max(0)[i] + lm_margin_nm) / lm_vs[i])) for i in range(3)
    )
    lm_affine = np.eye(4)
    lm_affine[:3, :3] = np.diag(lm_vs)

    # LM samples the EM-world scene through lm_to_em. A confocal voxel integrates over its
    # slab, so the scene is painted on a z-supersampled grid and box-averaged down to dz.
    ss = int(np.clip(np.round(lm_vs[0] / (2.0 * em_vs[0])), 1, 8))
    fine_shape = (lm_shape[0] * ss, lm_shape[1], lm_shape[2])
    fine_affine = np.eye(4)
    fine_affine[:3, :3] = np.diag([lm_vs[0] / ss, lm_vs[1], lm_vs[2]])
    fine_affine[0, 3] = -lm_vs[0] / 2.0 + lm_vs[0] / (
        2.0 * ss
    )  # sub-slices centred on each LM slice
    world_lm = _grid_world(fine_shape, fine_affine)
    world_lm_in_em = apply_affine(lm_to_em, world_lm.reshape(-1, 3)).reshape(*fine_shape, 3)
    lm_nuc = _paint_ellipsoids(world_lm_in_em, nuc_c, nuc_r, 1.0, soft=0.1)
    org_c_lm = apply_affine(em_to_lm, org_c)
    scale = np.cbrt(abs(np.linalg.det(em_to_lm[:3, :3])))
    lm_org = _paint_spheres_boxed(fine_shape, fine_affine, org_c_lm, org_r * scale, 1.0, soft=0.5)
    lm_nuc = lm_nuc.reshape(lm_shape[0], ss, lm_shape[1], lm_shape[2]).mean(axis=1)
    lm_org = lm_org.reshape(lm_shape[0], ss, lm_shape[1], lm_shape[2]).mean(axis=1)
    sig = np.array([psf_fwhm_nm[0], psf_fwhm_nm[1], psf_fwhm_nm[1]]) / 2.355 / lm_vs
    lm0 = gaussian_filter(lm_nuc, sig) + rng.normal(0, noise * 0.5, lm_shape) + 0.05
    lm1 = gaussian_filter(lm_org, sig) * 4.0 + rng.normal(0, noise * 0.5, lm_shape) + 0.05
    lm_stack = np.stack([lm0, lm1]).astype(np.float32)
    lm_stack = np.clip(lm_stack, 0, None)
    if np.issubdtype(lm_dtype, np.integer):
        mx = np.iinfo(lm_dtype).max
        lm_img = (np.clip(lm_stack / max(lm_stack.max(), 1e-6), 0, 1) * (mx * 0.8)).astype(lm_dtype)
    else:
        lm_img = lm_stack.astype(lm_dtype)

    em_vol = volume_from_array(em_img, em_voxel_nm, kind="em", source="phantom-em")
    lm_vol = volume_from_array(
        lm_img,
        lm_voxel_nm,
        kind="lm",
        source="phantom-lm",
        psf_nm=psf_fwhm_nm,
        channels=[Channel("nuclei", 0, "0000FF"), Channel("organelles", 1, "00FF00")],
    )
    truth = PhantomTruth(
        em_voxel_nm=tuple(em_voxel_nm),  # type: ignore[arg-type]
        lm_voxel_nm=tuple(lm_voxel_nm),  # type: ignore[arg-type]
        lm_to_em=lm_to_em,
        nuclei_em_world=nuc_c,
        nuclei_lm_world=apply_affine(em_to_lm, nuc_c),
        nuclei_radii_nm=nuc_r,
        organelles_em_world=org_c,
        organelles_lm_world=org_c_lm,
        organelle_radii_nm=org_r,
        psf_fwhm_nm=psf_fwhm_nm,
        extra={"seed": seed, "lm_shape": lm_shape, "em_shape": em_shape},
    )
    return em_vol, lm_vol, truth


@dataclass
class MisalignTruth:
    jitter: np.ndarray  # (n, 2) integer (dy, dx) fast component of the content shift
    drift: np.ndarray  # (n, 2) slow component
    bad_slices: list[int]
    total: np.ndarray  # jitter + drift = shift of slice content relative to the reference crop
    reference: np.ndarray | None = (
        None  # the un-shifted central crop (what alignment should recover)
    )
    margin: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "jitter": self.jitter.tolist(),
            "drift": self.drift.tolist(),
            "bad_slices": [int(b) for b in self.bad_slices],
            "total": self.total.tolist(),
            "margin": int(self.margin),
        }


def shift_int(img: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """Shift a 2D image by integer pixels, filling exposed edges."""
    out = np.full_like(img, fill)
    H, W = img.shape
    ys = slice(max(dy, 0), H + min(dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    ys_src = slice(max(-dy, 0), H + min(-dy, 0))
    xs_src = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = img[ys_src, xs_src]
    return out


def make_misaligned_stack(
    stack: np.ndarray,
    jitter_px: float = 2.0,
    drift_px: float = 12.0,
    n_bad: int = 2,
    seed: int = 0,
    bad_kind: str = "noise",
    margin: int | None = None,
) -> tuple[np.ndarray, MisalignTruth]:
    """Per-slice integer jitter + slow drift + a few bad slices, with the truth.

    Like a real SBF-SEM/FIB-SEM acquisition, every output slice is a *full frame* of content:
    slice i is the window of the larger input field cropped at an offset shifted by the truth,
    so there are no padded edges to bias the aligner. The output is smaller than the input by
    2*margin in y and x; ``truth.reference`` is the un-shifted central crop. The content of
    slice i is displaced by ``truth.total[i]`` relative to the reference, so the correction
    that re-aligns it is ``-truth.total[i]``.
    """
    rng = np.random.default_rng(seed)
    n = stack.shape[0]
    t = np.arange(n) / max(n - 1, 1)
    drift = np.stack(
        [drift_px * np.sin(2 * np.pi * t * 0.7), drift_px * 0.6 * (t**2) - drift_px * 0.2 * t],
        axis=1,
    )
    jitter = rng.normal(0, jitter_px, (n, 2))
    jitter[0] = 0
    drift[0] = 0
    total = np.round(jitter + drift).astype(int)
    drift_i = np.round(drift).astype(int)
    m = int(np.abs(total).max()) + 1 if margin is None else int(margin)
    H, W = stack.shape[1:]
    if 2 * m >= min(H, W):
        raise ValueError("stack too small for the requested shifts")
    Hc, Wc = H - 2 * m, W - 2 * m
    out = np.empty((n, Hc, Wc), dtype=stack.dtype)
    for i in range(n):
        dy, dx = int(total[i, 0]), int(total[i, 1])
        out[i] = stack[i, m - dy : m - dy + Hc, m - dx : m - dx + Wc]
    reference = stack[:, m : m + Hc, m : m + Wc].copy()
    bad = (
        sorted(
            rng.choice(np.arange(2, n - 2), size=min(n_bad, max(n - 4, 0)), replace=False).tolist()
        )
        if n_bad
        else []
    )
    for b in bad:
        if bad_kind == "noise":
            out[b] = rng.integers(0, int(stack.max()) + 1, out[b].shape).astype(stack.dtype)
        elif bad_kind == "blank":
            out[b] = 0
        else:
            out[b] = gaussian_filter(out[b].astype(float), 8).astype(stack.dtype)
    truth = MisalignTruth(
        jitter=total - drift_i,
        drift=drift_i,
        bad_slices=bad,
        total=total,
        reference=reference,
        margin=m,
    )
    return out, truth
