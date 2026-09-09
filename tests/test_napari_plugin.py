"""napari plugin smoke test with hidden windows (plan §11 'GUI tests offscreen').

On Windows the Qt offscreen platform has no OpenGL context, so the viewers are created with
``show=False`` on the default platform instead; on Linux CI set ``QT_QPA_PLATFORM=offscreen``.

Drives the dock widget end to end on the phantom: load, landmarks, pre-align remap, fit,
overlay affine, residual vectors, LM-driven slab view, compare layer, z readout, TPS
deformable overlay, alignment verify, export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

if os.name != "nt":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PYCLEM3D_SYNC"] = "1"
napari = pytest.importorskip("napari")

from pyclem3d.cli import main  # noqa: E402
from pyclem3d.io import open_volume  # noqa: E402
from pyclem3d.napari._reader import napari_get_reader, reader_function  # noqa: E402
from pyclem3d.session import Session  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def phantom_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("gui_phantom")
    assert (
        main(
            [
                "phantom",
                "--out",
                str(d),
                "--em-shape",
                "32",
                "128",
                "128",
                "--n-nuclei",
                "8",
                "--seed",
                "3",
                "--misalign",
                "--lm-voxel",
                "240",
                "80",
                "80",
            ]
        )
        == 0
    )
    return d


@pytest.fixture
def viewer():
    v = napari.Viewer(show=False)
    yield v
    v.close()


def test_reader_contribution(phantom_dir, viewer):
    p = str(phantom_dir / "em.ome.tif")
    assert napari_get_reader(p) is reader_function
    assert napari_get_reader("x.txt") is None
    layers = reader_function(p)
    assert len(layers) == 1 and layers[0][2] == "image"
    lay = viewer.add_image(layers[0][0], **layers[0][1])
    A = np.asarray(lay.affine.affine_matrix)
    assert np.allclose(np.diag(A)[:3], (20.0, 20.0, 20.0))  # world units are nm
    lm_layers = reader_function(str(phantom_dir / "lm.ome.tif"))
    assert len(lm_layers) == 2  # one layer per channel


def test_widget_end_to_end(phantom_dir, viewer, tmp_path):
    from pyclem3d.napari._widget import PyCLEM3DWidget

    w = PyCLEM3DWidget(viewer)
    assert w.tabs.count() == 8
    # --- Load (synchronously, through the same callbacks the buttons use)
    w.em_path.setText(str(phantom_dir / "em.ome.tif"))
    w.lm_path.setText(str(phantom_dir / "lm.ome.tif"))
    w._load_em()
    assert w.em is not None and len(w.em_layers) == 1 and w.em_points is not None
    w._load_lm()
    assert w.lm is not None and w.lm_viewer is not None
    assert len(w.lm_raw_layers) == 2 and len(w.lm_overlay) == 2
    assert "EM" in w.load_status.text() and "LM" in w.load_status.text()
    # --- landmarks from the phantom session
    sess = Session.load(phantom_dir / "session.json")
    w.session.landmarks = sess.landmarks
    w._refresh_points()
    n = len(sess.landmarks)
    assert w.table.rowCount() == n
    assert len(w.em_points.data) == n and len(w.lm_points.data) == n
    # add a pair from the last points of both layers
    w.em_points.data = np.vstack([w.em_points.data, [[100.0, 200.0, 300.0]]])
    w.lm_points.data = np.vstack([w.lm_points.data, [[110.0, 220.0, 330.0]]])
    w._add_pair()
    assert len(w.session.landmarks) == n + 1
    w.table.selectRow(n)
    w._remove_selected()
    assert len(w.session.landmarks) == n
    # --- pre-align remaps landmarks and keeps them pinned
    lm_before = np.array([l.lm_world_nm for l in w.session.landmarks.landmarks])
    w.pa_rot.setValue(10.0)
    w._apply_prealign()
    lm_after = np.array([l.lm_world_nm for l in w.session.landmarks.landmarks])
    assert not np.allclose(lm_before, lm_after)
    w.pa_rot.setValue(0.0)
    w._apply_prealign()
    assert np.allclose(
        np.array([l.lm_world_nm for l in w.session.landmarks.landmarks]), lm_before, atol=1e-6
    )
    # --- locate from 3 rough pairs, then a full affine fit
    w._locate()
    assert w.session.locate is not None
    w.fit_kind.setCurrentText("affine")
    w._do_fit()
    assert w.session.transform is not None
    truth = json.loads((phantom_dir / "truth.json").read_text())
    assert np.allclose(
        np.asarray(w.session.transform["matrix"]), np.asarray(truth["lm_to_em"]), atol=1e-6
    )
    assert "RMS" in w.fit_label.text()
    T = np.asarray(w.session.transform["matrix"])
    for lay in w.lm_overlay:
        assert np.allclose(np.asarray(lay.affine.affine_matrix), T @ w.lm.world_affine)
    assert w.residual_layer is not None and w.residual_layer.data.shape == (n, 2, 3)
    assert w.table.item(0, 5).text() != ""
    # --- verify panel: z readout, LM-driven slab, compare layer
    w._on_em_dims()
    assert w.z_label.text().startswith("EM z")
    w.zmode.setCurrentIndex(1)
    assert w.slab_layer is not None and w.slab_layer in w.lm_viewer.layers
    w._on_lm_dims()
    assert w.z_label.text().startswith("LM z")
    w.cmp_mode.setCurrentText("swipe")
    assert w.compare_layer is not None and w.compare_layer in viewer.layers
    w.cmp_mode.setCurrentText("blend")
    assert all(lay.visible for lay in w.lm_overlay)
    w._zprofile()
    assert "peaks at" in w.zprof_label.text()
    # cursor sync callbacks do not raise
    w._on_em_mouse(viewer, None)
    w._on_lm_mouse(w.lm_viewer, None)
    # --- refinement on a selected landmark shows what it did
    w.table.selectRow(0)
    w.box_nm.setValue(1200.0)
    w._paired()
    assert "paired centroids" in w.refine_status.text()
    assert any(lay.name.startswith("refine mask") for lay in viewer.layers)
    w.table.selectRow(0)
    w._snap()
    assert "snap xy" in w.refine_status.text()
    # --- TPS gives a computed deformable overlay
    w._do_fit()  # refit after refinement (affine)
    w.fit_kind.setCurrentText("tps")
    w._do_fit()
    assert (
        w.disp is not None
        and w.deform_layer
        and all(lay in viewer.layers for lay in w.deform_layer)
    )
    w.fit_kind.setCurrentText("affine")
    w._do_fit()
    assert w.deform_layer is None
    # --- alignment verify on the (clean) EM stack
    w.al_level.setValue(0)
    w._align(verify=True)
    assert w.align_result is not None and "verdict" in w.al_status.text()
    # --- export (synchronous) and session save
    w.ex_tf.setText(str(tmp_path / "tf"))
    w.ex_figs.setText(str(tmp_path / "figs"))
    w.ex_report.setText(str(tmp_path / "report.json"))
    w.ex_fused.setText(str(tmp_path / "fused.zarr"))
    w._export()
    assert (tmp_path / "tf" / "transform.json").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "fused.zarr").exists()
    assert len(list((tmp_path / "figs").glob("*.png"))) == 2
    w.session_path.setText(str(tmp_path / "gui_session.json"))
    w._save_session()
    back = Session.load(tmp_path / "gui_session.json")
    assert len(back.landmarks) == n and back.transform is not None
    assert open_volume(tmp_path / "fused.zarr", kind="em", memory="skip").n_channels == 4
    w.teardown()
    w.lm_viewer.close()


def test_open_session_in_widget(phantom_dir, viewer):
    from pyclem3d.napari._widget import PyCLEM3DWidget

    assert main(["register", str(phantom_dir / "session.json")]) == 0
    w = PyCLEM3DWidget(viewer)
    w.session_path.setText(str(phantom_dir / "session.json"))
    w._open_session()
    assert w.em is not None and w.lm is not None and w.session.transform is not None
    assert "RMS" in w.fit_label.text()
    assert w.table.rowCount() == len(w.session.landmarks)
    w.teardown()
    w.lm_viewer.close()
