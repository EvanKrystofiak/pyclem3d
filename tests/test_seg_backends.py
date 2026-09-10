"""Segmentation backends share one contract (zarr mask with the EM geometry); these tests use
stand-in engines so they need neither empanada nor QuantEM weights."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyclem3d.io.readers import volume_from_array
from pyclem3d.seg.mitonet import open_mask, segment_volume
from pyclem3d.seg.quantem import is_quantem_model, segment_any, segment_volume_quantem


class _FakeQuantEM:
    """Mimics quantem_em.api.QuantEMModel.segment: bright blobs above a threshold."""

    spec = SimpleNamespace(canonical_nm=8.0)
    threshold = 0.5

    def __init__(self):
        self.calls = []

    def segment(self, image, *, pixel_size_nm=None, threshold=None):
        self.calls.append((image.shape, pixel_size_nm, threshold))
        prob = (image.astype(np.float32) / 255.0).clip(0, 1)
        mask = prob > (0.5 if threshold is None else threshold)
        from scipy.ndimage import label

        labels, n = label(mask)
        return SimpleNamespace(
            labels=labels.astype(np.int32),
            mask=mask,
            probability=prob,
            n_objects=int(n),
            contract={},
        )


class _FakeEmpanada:
    def infer(self, image):
        return (image > 127).astype(np.int32) * 1001


def _volume():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 60, (6, 40, 48)).astype(np.uint8)
    vol[:, 10:20, 10:30] = 220
    vol[2:5, 25:35, 5:15] = 200
    return volume_from_array(vol, (8.0, 8.0, 8.0), kind="em", source="fake")


def test_is_quantem_model():
    assert is_quantem_model("quantem/mito") and is_quantem_model("omniem/nucleus")
    assert not is_quantem_model("mito") and not is_quantem_model("MitoNet_v1")


def test_quantem_backend_writes_mask_and_probability(tmp_path):
    em = _volume()
    eng = _FakeQuantEM()
    out = segment_volume_quantem(
        em,
        tmp_path / "q.zarr",
        model="quantem/mito",
        save_probability=True,
        engine=eng,
        z_range=(1, 5),
    )
    mask, attrs = open_mask(out)
    assert (
        mask.shape == em.shape_zyx
        and attrs["backend"] == "quantem"
        and attrs["model"] == "quantem/mito"
    )
    assert attrs["pixel_size_nm"] == 8.0 and attrs["z_range"] == [1, 5]
    m = np.asarray(mask.compute())
    assert m[0].sum() == 0 and m[5].sum() == 0  # outside z_range untouched
    assert m[2, 15, 20] == 1 and m[2, 30, 10] == 1 and m[2, 5, 40] == 0
    assert all(c[1] == 8.0 for c in eng.calls) and len(eng.calls) == 4
    import zarr

    g = zarr.open_group(str(out), mode="r")
    prob = np.asarray(g["prob"][2])
    assert prob.dtype == np.uint8 and prob[15, 20] > 200 and prob[5, 40] < 80
    assert 0.0 < attrs["foreground_fraction"] < 0.5 and attrs["objects_per_slice"] > 0


def test_segment_any_dispatches_by_model_id(tmp_path):
    em = _volume()
    out_q = segment_any(em, tmp_path / "q.zarr", model="quantem/mito", engine=_FakeQuantEM())
    out_e = segment_any(em, tmp_path / "e.zarr", model="mito", engine=_FakeEmpanada())
    mq, aq = open_mask(out_q)
    me, ae = open_mask(out_e)
    assert aq["backend"] == "quantem" and "backend" not in ae  # empanada writer predates the key
    assert np.array_equal(np.asarray(mq.compute()) > 0, np.asarray(me.compute()) > 0)
    # explicit backend override wins over the model id heuristic
    out_x = segment_any(
        em, tmp_path / "x.zarr", model="anything", backend="quantem", engine=_FakeQuantEM()
    )
    assert open_mask(out_x)[1]["backend"] == "quantem"
    # the empanada path is untouched
    out_direct = segment_volume(em, tmp_path / "d.zarr", model="mito", engine=_FakeEmpanada())
    assert np.asarray(open_mask(out_direct)[0].compute()).max() == 1
