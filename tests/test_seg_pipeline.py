"""Segmentation-to-image registration on the phantom (plan Phase 6): an organelle mask on the EM
grid -> synthetic fluorescence -> z scan recovers the depth and sign -> affine registration
recovers the true LM->EM transform. Needs SimpleITK (seg/itk extra)."""

from __future__ import annotations

import numpy as np
import pytest

from pyclem3d.io import ensure_pyramid
from pyclem3d.io.volume import apply_affine
from pyclem3d.phantom import make_phantom
from pyclem3d.register import AffineTransform
from pyclem3d.seg.synthetic import synthetic_fluorescence

sitk = pytest.importorskip("SimpleITK")
from pyclem3d.seg.intensity import (  # noqa: E402
    Cue,
    multi_cue_ncc,
    ncc_on_synthetic,
    refine_affine_multicue,
    register_affine,
    z_scan,
)


@pytest.fixture(scope="module")
def phantom():
    em, lm, truth = make_phantom(
        em_shape=(128, 192, 192),
        n_nuclei=0,
        n_organelles=160,
        seed=41,
        lm_voxel_nm=(150.0, 50.0, 50.0),
        psf_fwhm_nm=(400.0, 160.0),
        noise=0.02,
    )
    em = ensure_pyramid(em, cache_dir=None, min_size=16)
    # a perfect "segmentation" of the organelles on the EM grid
    Z, Y, X = em.shape_zyx
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=float) for s in (Z, Y, X)], indexing="ij")
    world = apply_affine(em.world_affine, np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1))
    mask = np.zeros(Z * Y * X, bool)
    for c, r in zip(truth.organelles_em_world, truth.organelle_radii_nm):
        mask |= np.linalg.norm(world - c, axis=1) <= r
    import dask.array as da

    return em, lm, truth, da.from_array(mask.reshape(Z, Y, X).astype(np.uint8), chunks=(1, Y, X))


def test_synthetic_matches_true_channel(phantom):
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    assert syn.voxel_size_nm == (40.0, 40.0, 40.0)
    T = AffineTransform(truth.lm_to_em)
    good = ncc_on_synthetic(syn, lm, 1, T)  # organelle channel
    # a 600 nm z error or a wrong z sign must score clearly worse
    M = T.matrix.copy()
    M[0, 3] += 600.0
    worse = ncc_on_synthetic(syn, lm, 1, AffineTransform(M))
    Mf = T.matrix.copy()
    Mf[0, :3] *= -1
    Mf[0, 3] = 2 * (em.bbox_world()[0][0] + em.bbox_world()[1][0]) / 2 - Mf[0, 3]
    flipped = ncc_on_synthetic(syn, lm, 1, AffineTransform(Mf))
    assert good > 0.6
    assert good > worse + 0.15
    assert good > flipped + 0.15


def test_probability_weighted_synthetic(phantom):
    """A soft mask (per-voxel probability) is a valid synthetic source and scores no worse than
    the hard mask; confident voxels dominate, uncertain ones are down-weighted."""
    import dask.array as da
    from scipy.ndimage import gaussian_filter

    em, lm, truth, mask = phantom
    hard = np.asarray(mask.compute()).astype(np.float32)
    # a plausible probability map: the mask blurred by a voxel, plus a low-confidence halo
    prob = np.clip(
        gaussian_filter(hard, 1.0) * 0.9 + 0.05 * (gaussian_filter(hard, 3.0) > 0.05), 0, 1
    )
    soft = da.from_array(prob.astype(np.float32), chunks=mask.chunks)
    T = AffineTransform(truth.lm_to_em)
    syn_hard = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    syn_soft = synthetic_fluorescence(soft, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    n_hard = ncc_on_synthetic(syn_hard, lm, 1, T)
    n_soft = ncc_on_synthetic(syn_soft, lm, 1, T)
    assert n_soft > 0.6 and n_soft >= n_hard - 0.05
    M = T.matrix.copy()
    M[0, 3] += 600.0
    assert n_soft > ncc_on_synthetic(syn_soft, lm, 1, AffineTransform(M)) + 0.15


def test_z_scan_and_affine_recover_truth(phantom):
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    T = AffineTransform(truth.lm_to_em)
    # start from a perturbed transform: 400 nm off in z, 150 nm in y, 3% scale error
    M = T.matrix.copy()
    M[0, 3] += 400.0
    M[1, 3] -= 150.0
    M[:3, :3] *= 1.03
    T0 = AffineTransform(M)
    scan = z_scan(syn, lm, 1, T0, np.arange(-800.0, 801.0, 100.0))
    best = max(scan, key=lambda r: r["ncc"])
    assert abs(best["offset_nm"] + 400.0) <= 150.0  # the scan finds the injected z error
    Mb = M.copy()
    Mb[0, 3] += best["offset_nm"]
    res = register_affine(
        syn,
        lm,
        1,
        AffineTransform(Mb),
        shrink=(2, 1),
        sigmas=(1.0, 0.0),
        iterations=200,
        learning_rate=2.0,
        sampling=0.5,
    )
    assert res.metric_after <= res.metric_before
    # compare mappings of test points against the truth, split into z and xy
    rng = np.random.default_rng(0)
    pts = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (200, 3))
    d0 = T0.apply(pts) - T.apply(pts)
    d1 = res.lm_to_em.apply(pts) - T.apply(pts)
    assert np.linalg.norm(d0, axis=1).mean() > 400.0
    assert np.linalg.norm(d1, axis=1).mean() < 100.0  # nm: about two LM xy voxels
    assert np.abs(d1[:, 0]).mean() < 100.0  # z is recovered, not just xy
    assert np.linalg.norm(d1[:, 1:], axis=1).mean() < 60.0


def test_multi_cue_ncc_signs_and_weights(phantom):
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    T = AffineTransform(truth.lm_to_em)
    n = ncc_on_synthetic(syn, lm, 1, T)
    bright = Cue("organelle", syn, 1, sign=1.0, weight=1.0)
    excl = Cue("exclusion", syn, 1, sign=-1.0, weight=3.0)
    out = multi_cue_ncc([bright, excl], lm, T)
    assert out["organelle"] == pytest.approx(n)
    assert out["exclusion"] == pytest.approx(-n)
    assert out["combined"] == pytest.approx((n - 3 * n) / 4)


def test_multicue_polish_refines_affine(phantom):
    """The Powell polish pulls a slightly wrong affine towards the truth and never lowers the NCC."""
    em, lm, truth, mask = phantom
    syn = synthetic_fluorescence(mask, em, target_voxel_nm=40.0, psf_fwhm_nm=(400.0, 160.0))
    T = AffineTransform(truth.lm_to_em)
    M = T.matrix.copy()
    M[0, 3] += 160.0
    M[2, 3] -= 90.0
    M[:3, :3] *= 1.02
    T0 = AffineTransform(M)
    T1, info = refine_affine_multicue([Cue("organelle", syn, 1)], lm, T0, maxfev=300)
    assert info["after"]["combined"] >= info["before"]["combined"]
    assert info["n_eval"] <= 300 + 12
    rng = np.random.default_rng(1)
    pts = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (200, 3))
    d0 = np.linalg.norm(T0.apply(pts) - T.apply(pts), axis=1).mean()
    d1 = np.linalg.norm(T1.apply(pts) - T.apply(pts), axis=1).mean()
    print(
        f"polish: mean error {d0:.0f} -> {d1:.0f} nm, NCC {info['before']['combined']:.4f} -> {info['after']['combined']:.4f}"
    )
    assert d0 > 150.0
    assert d1 < 0.5 * d0


def test_register_seg_cli_with_and_without_polish(phantom, tmp_path):
    """End to end through the CLI: mask zarr + session with a perturbed transform -> affine (+ polish)."""
    from pyclem3d.cli import main
    from pyclem3d.io.writers import write_ome_tiff
    from pyclem3d.seg.mitonet import open_group_v2
    from pyclem3d.session import Session, VolumeSpec

    em, lm, truth, mask = phantom
    write_ome_tiff(tmp_path / "em.ome.tif", em.data, em.voxel_size_nm, em.channels)
    write_ome_tiff(tmp_path / "lm.ome.tif", lm.data, lm.voxel_size_nm, lm.channels)
    Z, Y, X = mask.shape
    g = open_group_v2(tmp_path / "mask.zarr", mode="w")
    arr = (
        g.create_array("0", shape=(Z, Y, X), chunks=(1, Y, X), dtype="uint8")
        if hasattr(g, "create_array")
        else g.create_dataset("0", shape=(Z, Y, X), chunks=(1, Y, X), dtype="uint8")
    )
    arr[:] = np.asarray(mask.compute()).astype(np.uint8)
    g.attrs["pyclem3d_seg"] = {"backend": "test", "model": "truth"}
    T = AffineTransform(truth.lm_to_em)
    M = T.matrix.copy()
    M[0, 3] += 300.0
    M[1, 3] -= 100.0
    M[:3, :3] *= 1.02
    rng = np.random.default_rng(2)
    pts = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (200, 3))

    def run(tag, extra):
        sess = Session(
            em=VolumeSpec("em.ome.tif", "em", voxel_size_nm=em.voxel_size_nm),
            lm=VolumeSpec(
                "lm.ome.tif", "lm", voxel_size_nm=lm.voxel_size_nm, psf_nm=(400.0, 160.0)
            ),
            transform=AffineTransform(M).to_dict(),
        )
        sp = sess.save(tmp_path / f"session_{tag}.json")
        args = [
            "register-seg",
            str(sp),
            "--mask",
            str(tmp_path / "mask.zarr"),
            "--channel",
            "1",
            "--voxel",
            "40",
            "--z-range-nm",
            "600",
            "--z-step",
            "100",
            "--z-scales",
            "1.0",
            "--iterations",
            "150",
            "--polish-maxfev",
            "150",
        ]
        assert main(args + extra) == 0
        sess = Session.load(sp)
        Tf = sess.transform_obj()
        err = np.linalg.norm(Tf.apply(pts) - T.apply(pts), axis=1).mean()
        print(f"{tag}: mean error {err:.0f} nm")
        return sess.outputs["register_seg"], err

    out_np, err_np = run("nopolish", ["--no-polish"])
    assert out_np["polish"] is None
    out_p, err_p = run("polish", [])
    assert out_p["polish"] is not None
    assert out_p["polish"]["cues"] == ["mask"]
    assert 0 < out_p["polish"]["n_eval"] <= 150 + 12
    assert isinstance(out_p["polish"]["accepted"], bool)
    assert err_np < 150.0
    assert err_p < 150.0
    if out_p["polish"]["accepted"]:
        assert out_p["polish"]["ncc"] >= out_p["ncc"]["affine"] - 1e-4
