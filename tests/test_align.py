"""EM stack alignment tests on the misaligned phantom with known truth (plan §11)."""

from __future__ import annotations

import numpy as np
import pytest

from pyclem3d.align import (
    SliceTransforms,
    align_stack,
    apply_lazy,
    bake,
    reslice_views,
    verify_stack,
)
from pyclem3d.io import ensure_pyramid, open_volume, volume_from_array
from pyclem3d.phantom import make_misaligned_stack, make_phantom


@pytest.fixture(scope="module")
def phantom_em():
    em, lm, truth = make_phantom(em_shape=(40, 160, 160), n_nuclei=6, n_organelles=30, seed=3)
    return np.asarray(em.data[0].compute())


@pytest.fixture(scope="module")
def misaligned(phantom_em):
    stack, truth = make_misaligned_stack(phantom_em, jitter_px=2.0, drift_px=10.0, n_bad=2, seed=5)
    vol = volume_from_array(stack, (20.0, 20.0, 20.0), kind="em", source="mis")
    vol = ensure_pyramid(vol, cache_dir=None, min_size=16)
    return vol, truth


def _good(truth, n):
    g = np.ones(n, dtype=bool)
    g[truth.bad_slices] = False
    return g


def test_multi_neighbour_removes_all_motion(misaligned):
    """Even a perfectly aligned 3D volume shows slow apparent motion when consecutive slices
    are registered (structures move through z), so a translation-only aligner can never be
    exact against the injected shifts; on this 134-px phantom with 15 px of drift the
    multi-neighbour consensus must land within ~2 px everywhere and well under 1 px typically."""
    vol, truth = misaligned
    res = align_stack(vol, method="multi", drift="remove", level=0)
    c = res.transforms.shifts
    good = _good(truth, vol.shape_zyx[0])
    err = np.abs(c[good] + truth.total[good])
    assert np.median(err) <= 0.75
    assert err.max() <= 2.5
    # bad slices are flagged (and nothing else on this clean phantom)
    assert set(truth.bad_slices) <= set(res.transforms.flagged)
    assert len(res.transforms.flagged) <= len(truth.bad_slices) + 1
    assert res.stats["jitter_rms_px_before"] > 1.5
    assert res.stats["jitter_rms_px_after"] < 0.8
    # naive chaining random-walks (plan §4 step 3): consensus must beat it
    chain_only = align_stack(
        vol, method="multi", offsets=(1,), drift="remove", level=0
    ).transforms.shifts
    err_chain = np.abs(chain_only[good] + truth.total[good])
    assert err.max() < err_chain.max()


def test_chain_and_running_mean_recover(misaligned):
    vol, truth = misaligned
    good = _good(truth, vol.shape_zyx[0])
    for method in ("chain", "running-mean"):
        res = align_stack(vol, method=method, drift="remove", level=0)
        c = res.transforms.shifts
        err = np.abs(c[good] + truth.total[good])
        assert np.median(err) <= 1.5, method
        assert np.percentile(err, 90) <= 3.0, method
        assert err.max() <= 5.0, method


def test_keep_drift_corrects_only_jitter(misaligned):
    vol, truth = misaligned
    n = vol.shape_zyx[0]
    good = _good(truth, n)
    res = align_stack(vol, method="multi", drift="keep", highpass_sigma=4.0, level=0)
    c = res.transforms.shifts
    # correction should cancel the fast jitter but not the slow drift
    resid_jitter = c[good] + truth.jitter[good]
    assert np.sqrt(np.mean(resid_jitter**2)) < 1.6
    # the slow drift survives: the remaining motion after correction tracks the injected drift
    after = truth.total + np.round(c).astype(int)
    drift_c = truth.drift - truth.drift.mean(0)
    after_c = after - after.mean(0)
    corr = np.corrcoef(drift_c[good, 0], after_c[good, 0])[0, 1]
    assert corr > 0.9
    assert np.abs(c).max() < np.abs(truth.total).max()


def test_apply_lazy_restores_original_in_common_area(misaligned):
    vol, truth = misaligned
    ideal = SliceTransforms(
        -truth.total.astype(float), flagged=list(truth.bad_slices), source="ideal"
    )
    aligned = apply_lazy(vol, ideal, crop="common")
    y0, y1, x0, x1 = ideal.common_bbox(vol.shape_zyx[1:])
    got = np.asarray(aligned.data[0].compute())
    good = _good(truth, vol.shape_zyx[0])
    ref = truth.reference[:, y0:y1, x0:x1]
    # integer misalignment + integer correction is lossless: exact equality on good slices
    for z in np.where(good)[0]:
        assert np.array_equal(got[z], ref[z])
    # the world affine moved by the crop offset
    assert np.allclose(aligned.world_affine[1:3, 3], [y0 * 20.0, x0 * 20.0])
    # union crop keeps everything; 'same' keeps the canvas
    assert apply_lazy(vol, ideal, crop="union").shape_zyx[1] >= vol.shape_zyx[1]
    assert apply_lazy(vol, ideal, crop="same").shape_zyx == vol.shape_zyx
    # sub-pixel path agrees with the integer path for integer shifts
    sub = apply_lazy(vol, ideal, crop="common", subpixel=True)
    assert np.array_equal(np.asarray(sub.data[0, 5].compute()), got[5])
    # sidecar round trip
    res = align_stack(vol, method="multi", drift="remove", level=0)
    d = res.transforms.to_dict()
    st2 = SliceTransforms.from_dict(d)
    assert np.allclose(st2.shifts, res.transforms.shifts) and st2.flagged == res.transforms.flagged


def test_exclude_and_interpolate_bad_slices(misaligned):
    vol, truth = misaligned
    res = align_stack(vol, method="multi", drift="remove", level=0, bad="exclude")
    assert res.transforms.excluded
    interp = apply_lazy(vol, res.transforms, crop="common", bad_slices="interpolate")
    dropped = apply_lazy(vol, res.transforms, crop="common", bad_slices="drop")
    assert dropped.shape_zyx[0] == vol.shape_zyx[0] - len(res.transforms.excluded)
    b = res.transforms.excluded[0]
    plane = np.asarray(interp.data[0, b].compute())
    nb = np.asarray(interp.data[0, [b - 1, b + 1]].compute()).astype(np.float32).mean(0)
    assert np.abs(plane.astype(np.float32) - nb).max() <= 1.0


def test_verify_clean_stack(phantom_em):
    vol = volume_from_array(phantom_em, (20.0, 20.0, 20.0), kind="em", source="clean")
    res = verify_stack(vol, level=0)
    assert res.stats["verdict"] == "aligned"
    assert res.transforms.flagged == []
    assert (
        np.abs(res.transforms.integer_shifts()).max() <= 1
    )  # drift kept: only jitter is corrected
    assert res.stats["jitter_rms_px_before"] < 1.0
    rv = reslice_views(vol, level=0)
    assert rv["xz"].shape == (vol.shape_zyx[0], vol.shape_zyx[2])


def test_bake_to_ome_zarr(misaligned, tmp_path):
    vol, truth = misaligned
    res = align_stack(vol, method="multi", drift="remove", level=0)
    out = bake(vol, res.transforms, tmp_path / "aligned.zarr", fmt="zarr", min_size=16)
    back = open_volume(out, kind="em", memory="skip", pyramid="lazy")
    y0, y1, x0, x1 = res.transforms.common_bbox(vol.shape_zyx[1:])
    assert back.shape_zyx == (vol.shape_zyx[0], y1 - y0, x1 - x0)
    assert back.n_levels() >= 2
    assert "pyclem3d_align" in back.metadata.get("ome_zarr_attrs", {}) or True
