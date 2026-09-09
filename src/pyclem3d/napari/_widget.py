"""pyCLEM-3D workflow dock widget for napari (plan §8).

Panels: Load -> Align EM -> Pre-align -> Locate -> Landmarks -> Fit -> Verify -> Export.
The main viewer shows the EM (reference grid, never resampled) with the LM overlaid through
its layer affine; a second viewer shows the raw (pre-aligned) LM for picking landmarks, with
a synced world cursor. The core never imports Qt; this module is the only Qt code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..io.readers import open_volume
from ..io.volume import Volume
from ..register.landmarks import default_sigma_nm
from ..register.prealign import PreAlign3D
from ..register.transforms import AffineTransform, TPSTransform, Transform
from ..resample.grid import DisplacementGrid, Grid
from ..resample.slab import SlabCache, SlabParams, em_level_for_lm
from ..session import Session, VolumeSpec, apply_prealign
from ._reader import volume_layers

log = logging.getLogger(__name__)

PERMUTATIONS = {
    "z y x (as is)": (0, 1, 2),
    "z x y": (0, 2, 1),
    "y z x": (1, 0, 2),
    "y x z": (1, 2, 0),
    "x z y": (2, 0, 1),
    "x y z": (2, 1, 0),
}


def _show_scale_bar(viewer) -> None:
    """Scale bar in nm; napari >= 0.9 reads units from the layers (set in _reader)."""
    try:
        viewer.canvas.overlays.scale_bar.visible = True
    except Exception:
        try:
            viewer.scale_bar.visible = True
        except Exception:  # pragma: no cover
            pass


def _spin(lo: float, hi: float, val: float, step: float = 1.0, decimals: int = 1) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    return s


def _vs_widgets() -> tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
    return _spin(0, 1e6, 0, 1, 2), _spin(0, 1e6, 0, 1, 2), _spin(0, 1e6, 0, 1, 2)


def _vs_value(ws) -> tuple[float, float, float] | None:
    v = tuple(w.value() for w in ws)
    return None if any(x <= 0 for x in v) else v  # type: ignore[return-value]


class PyCLEM3DWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.lm_viewer = None
        self.session: Session | None = None
        self.em: Volume | None = None
        self.lm_raw: Volume | None = None
        self.lm: Volume | None = None
        self.fit = None
        self.disp: DisplacementGrid | None = None
        self.slab_cache: SlabCache | None = None
        self.align_result = None
        self.em_layers: list = []
        self.lm_overlay: list = []
        self.lm_raw_layers: list = []
        self.em_points = None
        self.lm_points = None
        self.residual_layer = None
        self.cursor_em = None
        self.cursor_lm = None
        self.deform_layer = None
        self.slab_layer = None
        self.compare_layer = None
        self._sync = True
        self._closing = False
        self._flicker = QTimer(self)
        self._flicker.timeout.connect(self._flicker_tick)
        self._workers: list = []

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._tab_load(), "Load")
        self.tabs.addTab(self._tab_align(), "Align EM")
        self.tabs.addTab(self._tab_prealign(), "Pre-align")
        self.tabs.addTab(self._tab_locate(), "Locate")
        self.tabs.addTab(self._tab_landmarks(), "Landmarks")
        self.tabs.addTab(self._tab_fit(), "Fit")
        self.tabs.addTab(self._tab_verify(), "Verify")
        self.tabs.addTab(self._tab_export(), "Export")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        layout.addWidget(self.log)
        self.viewer.dims.events.current_step.connect(self._on_em_dims)
        self.viewer.mouse_move_callbacks.append(self._on_em_mouse)
        _show_scale_bar(self.viewer)

    # ------------------------------------------------------------------ utils
    def teardown(self) -> None:
        """Disconnect from both viewers (call before closing them; layer removal fires dims
        events that would otherwise rebuild computed layers in a closing viewer)."""
        self._closing = True
        self._flicker.stop()
        for v, cb, mcb in (
            (self.viewer, self._on_em_dims, self._on_em_mouse),
            (self.lm_viewer, self._on_lm_dims, self._on_lm_mouse),
        ):
            if v is None:
                continue
            try:
                v.dims.events.current_step.disconnect(cb)
            except Exception:
                pass
            try:
                v.mouse_move_callbacks.remove(mcb)
            except ValueError:
                pass

    def _alive(self, viewer, layers: list) -> bool:
        """True while ``viewer`` still holds this widget's base layers (not closing/cleared)."""
        if self._closing or viewer is None or not layers:
            return False
        try:
            return layers[0] in viewer.layers
        except Exception:
            return False

    def _say(self, msg: str) -> None:
        log.info(msg)
        self.log.append(msg)

    def _error(self, msg: str) -> None:
        log.error(msg)
        self.log.append(f"ERROR: {msg}")
        self.last_error = msg
        if not os.environ.get("PYCLEM3D_SYNC"):
            QMessageBox.warning(self, "pyCLEM-3D", msg)

    def _run_bg(self, fn, on_done, label: str, *args, **kwargs) -> None:
        """Run ``fn`` in a napari thread worker; ``on_done(result)`` on the GUI thread.

        With ``PYCLEM3D_SYNC=1`` in the environment (tests, scripting) it runs inline."""
        if os.environ.get("PYCLEM3D_SYNC"):
            try:
                on_done(fn(*args, **kwargs))
            except Exception as e:
                self._error(f"{label}: {e}")
            return
        try:
            from napari.qt.threading import create_worker
        except Exception:  # pragma: no cover
            try:
                on_done(fn(*args, **kwargs))
            except Exception as e:
                self._error(f"{label}: {e}")
            return
        self._say(f"{label}...")
        worker = create_worker(fn, *args, **kwargs)
        worker.returned.connect(on_done)
        worker.errored.connect(lambda e: self._error(f"{label}: {e}"))
        worker.start()
        self._workers.append(worker)

    def _ensure_session(self) -> Session:
        if self.session is None:
            self.session = Session(em=VolumeSpec("", "em"), lm=VolumeSpec("", "lm"))
        return self.session

    def _transform(self) -> Transform:
        if self.session is not None and self.session.transform is not None:
            return self.session.transform_obj()  # type: ignore[return-value]
        if self.session is not None and self.session.locate is not None:
            return AffineTransform(np.asarray(self.session.locate), "similarity")
        return AffineTransform.identity()

    def _browse(
        self, edit: QLineEdit, save: bool = False, directory: bool = False, filt: str = ""
    ) -> None:
        if directory:
            p = QFileDialog.getExistingDirectory(
                self, "Choose directory", edit.text() or os.getcwd()
            )
        elif save:
            p, _ = QFileDialog.getSaveFileName(self, "Save as", edit.text() or os.getcwd(), filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Open", edit.text() or os.getcwd(), filt)
        if p:
            edit.setText(p)

    def _path_row(self, edit: QLineEdit, **kw) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        b = QPushButton("...")
        b.setMaximumWidth(30)
        b.clicked.connect(lambda: self._browse(edit, **kw))
        h.addWidget(b)
        return w

    # ------------------------------------------------------------------- load
    def _tab_load(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.em_path = QLineEdit()
        self.lm_path = QLineEdit()
        f.addRow("EM stack", self._path_row(self.em_path))
        self.em_vs = _vs_widgets()
        row = QHBoxLayout()
        for s in self.em_vs:
            row.addWidget(s)
        f.addRow("EM voxel nm (z y x; 0 = metadata)", row)
        self.y_scale = _spin(0.1, 10, 1.0, 0.01, 4)
        f.addRow("FIB-SEM y tilt factor", self.y_scale)
        f.addRow("LM stack", self._path_row(self.lm_path))
        self.lm_vs = _vs_widgets()
        row = QHBoxLayout()
        for s in self.lm_vs:
            row.addWidget(s)
        f.addRow("LM voxel nm (z y x; 0 = metadata)", row)
        self.psf_z = _spin(0, 1e5, 0, 10, 0)
        self.psf_xy = _spin(0, 1e5, 0, 10, 0)
        row = QHBoxLayout()
        row.addWidget(self.psf_z)
        row.addWidget(self.psf_xy)
        f.addRow("PSF FWHM nm (z xy; 0 = metadata/fallback)", row)
        self.memory = QComboBox()
        self.memory.addItems(["auto", "ram", "lazy"])
        f.addRow("Memory strategy", self.memory)
        self.cache_dir = QLineEdit()
        f.addRow("Pyramid cache dir", self._path_row(self.cache_dir, directory=True))
        b = QPushButton("Load EM")
        b.clicked.connect(self._load_em)
        b2 = QPushButton("Load LM")
        b2.clicked.connect(self._load_lm)
        row = QHBoxLayout()
        row.addWidget(b)
        row.addWidget(b2)
        f.addRow(row)
        self.load_status = QLabel("nothing loaded")
        self.load_status.setWordWrap(True)
        f.addRow(self.load_status)
        self.session_path = QLineEdit()
        f.addRow("Session", self._path_row(self.session_path, save=True, filt="Session (*.json)"))
        b3 = QPushButton("Open session")
        b3.clicked.connect(self._open_session)
        b4 = QPushButton("Save session")
        b4.clicked.connect(self._save_session)
        row = QHBoxLayout()
        row.addWidget(b3)
        row.addWidget(b4)
        f.addRow(row)
        return w

    def _open_kwargs(self) -> dict[str, Any]:
        m = self.memory.currentText()
        return {"memory": None if m == "auto" else m, "cache_dir": self.cache_dir.text() or None}

    def _load_em(self) -> None:
        p = self.em_path.text()
        if not p:
            return self._error("choose an EM stack")
        sess = self._ensure_session()
        sess.em = VolumeSpec(
            p, "em", voxel_size_nm=_vs_value(self.em_vs), y_scale=self.y_scale.value()
        )

        def work():
            return open_volume(
                p,
                kind="em",
                voxel_size_nm=_vs_value(self.em_vs),
                y_scale=self.y_scale.value(),
                **self._open_kwargs(),
            )

        self._run_bg(work, self._em_loaded, "loading EM")

    def _em_loaded(self, vol: Volume) -> None:
        self.em = vol
        for lay in self.em_layers:
            try:
                self.viewer.layers.remove(lay)
            except ValueError:
                pass
        self.em_layers = [self.viewer.add_image(d, **m) for d, m, _ in volume_layers(vol, "EM ")]
        self._status()
        if self.em_points is None:
            self.em_points = self.viewer.add_points(
                np.empty((0, 3)),
                name="EM landmarks",
                size=max(vol.voxel_size_nm) * 20,
                face_color="yellow",
                ndim=3,
            )
            self.cursor_em = self.viewer.add_points(
                np.empty((0, 3)),
                name="cursor (from LM)",
                size=max(vol.voxel_size_nm) * 15,
                face_color="cyan",
                ndim=3,
                symbol="cross",
            )
        self._say(vol.describe())
        for wmsg in vol.metadata.get("warnings", []):
            self._say(f"warning: {wmsg}")
        if vol.metadata.get("recommend_zarr"):
            self._say(
                "recommendation: convert to OME-Zarr (pyclem3d convert) because: "
                + "; ".join(vol.metadata["recommend_zarr"])
            )
        self._refresh_overlay()

    def _load_lm(self) -> None:
        p = self.lm_path.text()
        if not p:
            return self._error("choose an LM stack")
        sess = self._ensure_session()
        psf = (
            (self.psf_z.value(), self.psf_xy.value())
            if self.psf_z.value() > 0 and self.psf_xy.value() > 0
            else None
        )
        sess.lm = VolumeSpec(p, "lm", voxel_size_nm=_vs_value(self.lm_vs), psf_nm=psf)

        def work():
            return open_volume(
                p, kind="lm", voxel_size_nm=_vs_value(self.lm_vs), psf_nm=psf, **self._open_kwargs()
            )

        self._run_bg(work, self._lm_loaded, "loading LM")

    def _lm_loaded(self, vol: Volume) -> None:
        self.lm_raw = vol
        sess = self._ensure_session()
        self.lm = apply_prealign(vol, sess.prealign)
        self._say(vol.describe())
        self._ensure_lm_viewer()
        self._status()
        self._refresh_overlay()
        self._refresh_points()

    def _ensure_lm_viewer(self) -> None:
        import napari

        if self.lm_viewer is None:
            headless = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen" or bool(
                os.environ.get("PYCLEM3D_SYNC")
            )
            self.lm_viewer = napari.Viewer(
                title="pyCLEM-3D: LM (pre-aligned world nm)", show=not headless
            )
            self.lm_viewer.dims.events.current_step.connect(self._on_lm_dims)
            self.lm_viewer.mouse_move_callbacks.append(self._on_lm_mouse)
            _show_scale_bar(self.lm_viewer)
        for lay in self.lm_raw_layers:
            try:
                self.lm_viewer.layers.remove(lay)
            except ValueError:
                pass
        assert self.lm is not None
        self.lm_raw_layers = [
            self.lm_viewer.add_image(d, **m) for d, m, _ in volume_layers(self.lm, "LM ")
        ]
        if self.lm_points is None:
            s = max(self.lm.voxel_size_nm) * 4
            self.lm_points = self.lm_viewer.add_points(
                np.empty((0, 3)), name="LM landmarks", size=s, face_color="yellow", ndim=3
            )
            self.cursor_lm = self.lm_viewer.add_points(
                np.empty((0, 3)),
                name="cursor (from EM)",
                size=s,
                face_color="cyan",
                ndim=3,
                symbol="cross",
            )
            self.slab_layer = None

    def _status(self) -> None:
        parts = []
        for v in (self.em, self.lm):
            if v is not None:
                md = v.metadata.get("memory_decision", {})
                parts.append(f"{v.describe()}  [{md.get('reason', '')}]")
        self.load_status.setText("\n".join(parts) or "nothing loaded")

    def _open_session(self) -> None:
        p = self.session_path.text()
        if not p:
            return self._error("choose a session file")
        try:
            self.session = Session.load(p)
        except Exception as e:
            return self._error(f"cannot open session: {e}")
        self.em_path.setText(str(self.session.em.resolve(self.session.base_dir)))
        self.lm_path.setText(str(self.session.lm.resolve(self.session.base_dir)))

        def work():
            assert self.session is not None
            return self.session.open_volumes(cache_dir=self.cache_dir.text() or None)

        def done(res):
            em, lm = res
            self._em_loaded(em)
            self.lm_raw = self.session.lm.open(self.session.base_dir, self.cache_dir.text() or None)  # type: ignore[union-attr]
            self.lm = lm
            self._ensure_lm_viewer()
            self._status()
            self._refresh_points()
            self._refresh_overlay()
            self._update_fit_labels()
            self._say(
                f"session opened: {len(self.session.landmarks)} landmarks, kind {self.session.kind}"
            )  # type: ignore[union-attr]

        self._run_bg(work, done, "opening session volumes")

    def _save_session(self) -> None:
        if self.session is None:
            return self._error("nothing to save")
        p = self.session_path.text() or self.session.path
        if not p:
            return self._error("choose a session path")
        self.session.save(p)
        self._say(f"session saved: {p}")

    # ------------------------------------------------------------------ align
    def _tab_align(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.al_method = QComboBox()
        self.al_method.addItems(["multi", "chain", "running-mean"])
        self.al_drift = QComboBox()
        self.al_drift.addItems(["keep", "remove"])
        self.al_sigma = _spin(0, 1000, 5, 1, 1)
        self.al_level = QSpinBox()
        self.al_level.setRange(-1, 10)
        self.al_level.setValue(-1)
        self.al_subpixel = QCheckBox("sub-pixel (interpolates)")
        self.al_exclude = QCheckBox("exclude flagged slices")
        f.addRow("Method", self.al_method)
        f.addRow("Drift", self.al_drift)
        f.addRow("High-pass sigma (slices)", self.al_sigma)
        f.addRow("Pyramid level (-1 auto)", self.al_level)
        f.addRow(self.al_subpixel)
        f.addRow(self.al_exclude)
        b = QPushButton("Verify (is it aligned?)")
        b.clicked.connect(lambda: self._align(verify=True))
        b2 = QPushButton("Align")
        b2.clicked.connect(lambda: self._align(verify=False))
        row = QHBoxLayout()
        row.addWidget(b)
        row.addWidget(b2)
        f.addRow(row)
        self.al_status = QLabel("")
        self.al_status.setWordWrap(True)
        f.addRow(self.al_status)
        b3 = QPushButton("Apply lazily (no write)")
        b3.clicked.connect(self._align_apply)
        b4 = QPushButton("Bake to OME-Zarr...")
        b4.clicked.connect(self._align_bake)
        b5 = QPushButton("Reslice xz / yz")
        b5.clicked.connect(self._reslice)
        b6 = QPushButton("Trajectory plot")
        b6.clicked.connect(self._align_plot)
        row = QHBoxLayout()
        for x in (b3, b4, b5, b6):
            row.addWidget(x)
        f.addRow(row)
        return w

    def _align(self, verify: bool) -> None:
        if self.em is None:
            return self._error("load an EM stack first")
        from ..align.pipeline import align_stack, verify_stack

        kw = dict(
            highpass_sigma=self.al_sigma.value(),
            level=None if self.al_level.value() < 0 else self.al_level.value(),
            subpixel=self.al_subpixel.isChecked(),
            bad="exclude" if self.al_exclude.isChecked() else "flag",
        )
        em = self.em

        def work():
            if verify:
                return verify_stack(em, **kw)
            return align_stack(
                em, method=self.al_method.currentText(), drift=self.al_drift.currentText(), **kw
            )

        def done(res):
            self.align_result = res
            s = res.summary()
            if verify:
                s += f"\nverdict: {res.stats['verdict']}"
            self.al_status.setText(s)
            self._say(s)

        self._run_bg(work, done, "verifying stack" if verify else "aligning stack")

    def _align_apply(self) -> None:
        if self.align_result is None or self.em is None:
            return self._error("run an alignment first")
        from ..align.apply import apply_lazy
        from ..align.sidecar import sidecar_path
        from ..io.pyramid import ensure_pyramid

        st = self.align_result.transforms
        sc = sidecar_path(self.em.source)
        try:
            st.save(sc)
            self._say(f"sidecar written: {sc}")
        except OSError as e:
            self._say(f"could not write sidecar next to the data ({e}); it stays in memory")
        vol = apply_lazy(
            self.em, st, crop="common", bad_slices="interpolate" if st.excluded else "keep"
        )
        vol = ensure_pyramid(vol, cache_dir=self.cache_dir.text() or None)
        sess = self._ensure_session()
        sess.em.align_sidecar = str(sc)
        self._em_loaded(vol)

    def _align_bake(self) -> None:
        if self.align_result is None or self.em is None:
            return self._error("run an alignment first")
        p, _ = QFileDialog.getSaveFileName(
            self, "Bake aligned stack", "", "OME-Zarr (*.zarr);;OME-TIFF (*.ome.tif);;MRC (*.mrc)"
        )
        if not p:
            return
        from ..align.apply import bake

        fmt = (
            "zarr"
            if p.endswith(".zarr")
            else "ome-tiff"
            if p.lower().endswith((".tif", ".tiff"))
            else "mrc"
        )
        em, st = self.em, self.align_result.transforms
        self._run_bg(
            lambda: bake(em, st, p, fmt=fmt),
            lambda out: self._say(f"baked: {out}"),
            "baking aligned stack",
        )

    def _reslice(self) -> None:
        if self.em is None:
            return self._error("load an EM stack first")
        import napari

        from ..align.apply import reslice_views

        rv = reslice_views(self.em)
        v = napari.Viewer(title="pyCLEM-3D: reslice (misalignment shows as jagged edges)")
        vs = rv["voxel_size_nm"]
        v.add_image(rv["xz"], name="xz", scale=(vs[0], vs[2]))
        v.add_image(
            rv["yz"],
            name="yz",
            scale=(vs[0], vs[1]),
            translate=(0, rv["xz"].shape[1] * vs[2] * 1.05),
        )
        _show_scale_bar(v)

    def _align_plot(self) -> None:
        if self.align_result is None:
            return self._error("run an alignment first")
        try:
            import matplotlib

            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
        except Exception:
            return self._error("matplotlib is not installed (pip install matplotlib)")
        s = self.align_result.stats
        before = np.asarray(s["trajectory_before_px"])
        after = np.asarray(s["trajectory_after_px"])
        conf = [np.nan if c is None else c for c in s["pair_conf"]]
        fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        for a, name in enumerate(("dy", "dx")):
            ax[a].plot(before[:, a], label="before")
            ax[a].plot(after[:, a], label="after")
            ax[a].set_ylabel(f"{name} (px)")
            ax[a].legend()
        ax[2].plot(np.arange(1, len(conf) + 1), conf, ".-")
        for fz in self.align_result.transforms.flagged:
            ax[2].axvline(fz, color="r", alpha=0.4)
        ax[2].set_ylabel("pair NCC")
        ax[2].set_xlabel("slice")
        fig.tight_layout()
        fig.show()

    # --------------------------------------------------------------- prealign
    def _tab_prealign(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.pa_perm = QComboBox()
        self.pa_perm.addItems(list(PERMUTATIONS))
        self.pa_flip = [QCheckBox(a) for a in ("flip z", "flip y", "flip x")]
        self.pa_rot = _spin(-180, 180, 0, 1, 1)
        f.addRow("Axis order (LM -> EM)", self.pa_perm)
        row = QHBoxLayout()
        for c in self.pa_flip:
            row.addWidget(c)
        f.addRow(row)
        f.addRow("Rotation about z (deg)", self.pa_rot)
        b = QPushButton("Apply pre-align (landmarks are remapped)")
        b.clicked.connect(self._apply_prealign)
        f.addRow(b)
        f.addRow(
            QLabel("Use this for gross axis swaps/flips/rotation only; the fit handles the rest.")
        )
        return w

    def _apply_prealign(self) -> None:
        if self.lm_raw is None:
            return self._error("load the LM first")
        new = PreAlign3D(
            PERMUTATIONS[self.pa_perm.currentText()],
            tuple(c.isChecked() for c in self.pa_flip),
            self.pa_rot.value(),
        )  # type: ignore[arg-type]
        sess = self._ensure_session()
        sess.set_prealign(new, self.lm_raw)
        self.lm = apply_prealign(self.lm_raw, new)
        self.slab_cache = None
        self.disp = None
        self._ensure_lm_viewer()
        self._refresh_points()
        self._refresh_overlay()
        self._update_fit_labels()
        self._say(f"pre-align applied: {new.to_dict()}")

    # ----------------------------------------------------------------- locate
    def _tab_locate(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(
            QLabel(
                "The EM block is usually a small part of the confocal field. Click 2-3 rough\npairs (EM landmarks layer in this viewer, LM landmarks layer in the LM viewer),\nthen fit a similarity to put the overlay in the right neighbourhood."
            )
        )
        b = QPushButton("Fit similarity from current pairs (locate)")
        b.clicked.connect(self._locate)
        v.addWidget(b)
        b2 = QPushButton("Centre both viewers on last EM point")
        b2.clicked.connect(self._center_on_last)
        v.addWidget(b2)
        self.sync_box = QCheckBox("Sync cursor and z between viewers")
        self.sync_box.setChecked(True)
        self.sync_box.toggled.connect(lambda s: setattr(self, "_sync", s))
        v.addWidget(self.sync_box)
        v.addStretch()
        return w

    def _pairs_from_layers(self) -> tuple[np.ndarray, np.ndarray]:
        if self.em_points is None or self.lm_points is None:
            raise ValueError("load both volumes first")
        a = np.asarray(self.em_points.data, dtype=float)
        b = np.asarray(self.lm_points.data, dtype=float)
        n = min(len(a), len(b))
        return a[:n], b[:n]

    def _locate(self) -> None:
        try:
            em_pts, lm_pts = self._pairs_from_layers()
            from ..register.estimators import fit_similarity

            if len(em_pts) < 3:
                raise ValueError("need at least 3 rough pairs")
            M = fit_similarity(lm_pts, em_pts)
        except Exception as e:
            return self._error(str(e))
        sess = self._ensure_session()
        sess.locate = M.tolist()
        sess.transform = None
        sess.fit = None
        self._refresh_overlay()
        self._say(f"locate: similarity from {len(em_pts)} pairs; overlay updated")

    def _center_on_last(self) -> None:
        if self.em_points is None or len(self.em_points.data) == 0:
            return
        p = np.asarray(self.em_points.data[-1], dtype=float)
        self.viewer.dims.set_point(0, p[0])
        self.viewer.camera.center = tuple(p)
        if self.lm_viewer is not None:
            q = (
                self._transform().inverse().apply(p)
                if self._transform().is_linear
                else (self.disp.inverse(p[None])[0] if self.disp is not None else p)
            )
            self.lm_viewer.dims.set_point(0, q[0])
            self.lm_viewer.camera.center = tuple(q)

    # -------------------------------------------------------------- landmarks
    def _tab_landmarks(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "id",
                "feature",
                "EM z,y,x (nm)",
                "LM z,y,x (nm)",
                "sigma z/xy",
                "resid nm",
                "LOO nm",
                "on",
            ]
        )
        v.addWidget(self.table)
        row = QHBoxLayout()
        self.feature_edit = QLineEdit("nucleus")
        row.addWidget(QLabel("feature"))
        row.addWidget(self.feature_edit)
        b = QPushButton("Add pair from last points")
        b.clicked.connect(self._add_pair)
        row.addWidget(b)
        b2 = QPushButton("Remove selected")
        b2.clicked.connect(self._remove_selected)
        row.addWidget(b2)
        b3 = QPushButton("Toggle enabled")
        b3.clicked.connect(self._toggle_selected)
        row.addWidget(b3)
        v.addLayout(row)
        g = QGroupBox(
            "Refine selected landmark (endogenous features are big and fuzzy: centre them consistently)"
        )
        f = QFormLayout(g)
        self.box_nm = _spin(100, 1e6, 6000, 500, 0)
        f.addRow("Box size (nm)", self.box_nm)
        row = QHBoxLayout()
        for name, fn in (
            ("Centroid EM", lambda: self._centroid("em")),
            ("Centroid LM", lambda: self._centroid("lm")),
            ("Paired centroids", self._paired),
            ("Snap xy", self._snap),
        ):
            b = QPushButton(name)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow(row)
        row = QHBoxLayout()
        self.mid_first = _spin(0, 1e6, 0, 1, 0)
        self.mid_last = _spin(0, 1e6, 0, 1, 0)
        row.addWidget(QLabel("first slice"))
        row.addWidget(self.mid_first)
        row.addWidget(QLabel("last slice"))
        row.addWidget(self.mid_last)
        b = QPushButton("Midpoint z -> EM")
        b.clicked.connect(lambda: self._midpoint("em"))
        row.addWidget(b)
        b = QPushButton("Midpoint z -> LM")
        b.clicked.connect(lambda: self._midpoint("lm"))
        row.addWidget(b)
        f.addRow(row)
        self.refine_status = QLabel("")
        self.refine_status.setWordWrap(True)
        f.addRow(self.refine_status)
        v.addWidget(g)
        return w

    def _selected_landmark(self):
        if self.session is None:
            return None
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        id_ = int(self.table.item(rows[0].row(), 0).text())
        try:
            return self.session.landmarks.get(id_)
        except KeyError:
            return None

    def _add_pair(self) -> None:
        if self.em is None or self.lm is None:
            return self._error("load both volumes first")
        try:
            em_pts, lm_pts = self._pairs_from_layers()
        except ValueError as e:
            return self._error(str(e))
        if len(em_pts) == 0 or len(lm_pts) == 0:
            return self._error(
                "click a point in each viewer first (EM landmarks / LM landmarks layers)"
            )
        sess = self._ensure_session()
        sig = default_sigma_nm(self.lm, self.em)
        lmk = sess.landmarks.add(
            tuple(em_pts[-1]),
            tuple(lm_pts[-1]),
            sig,
            feature=self.feature_edit.text(),
            em_voxel=tuple(self.em.world_to_voxel(em_pts[-1])),
            lm_voxel=tuple(self.lm.world_to_voxel(lm_pts[-1])),
        )
        sess.transform = None
        sess.fit = None
        self._say(f"landmark {lmk.id} added ({sess.landmarks.n_enabled} enabled)")
        self._refresh_points()
        self._refresh_table()

    def _remove_selected(self) -> None:
        lm = self._selected_landmark()
        if lm is None or self.session is None:
            return
        self.session.landmarks.remove(lm.id)
        self.session.transform = None
        self._refresh_points()
        self._refresh_table()

    def _toggle_selected(self) -> None:
        lm = self._selected_landmark()
        if lm is None or self.session is None:
            return
        lm.enabled = not lm.enabled
        self.session.transform = None
        self._refresh_table()

    def _refresh_points(self) -> None:
        if self.session is None:
            return
        ls = self.session.landmarks
        if self.em_points is not None:
            self.em_points.data = np.array(
                [l.em_world_nm for l in ls.landmarks], dtype=float
            ).reshape(-1, 3)
        if self.lm_points is not None:
            self.lm_points.data = np.array(
                [l.lm_world_nm for l in ls.landmarks], dtype=float
            ).reshape(-1, 3)
        self._refresh_table()

    def _refresh_table(self) -> None:
        if self.session is None:
            return
        ls = self.session.landmarks
        fit = self.session.fit or {}
        ids = fit.get("ids") or []
        res = fit.get("rms", {}).get("per_point", [])
        loo = (fit.get("loo") or {}).get("per_point_full", [])
        rmap = {
            i: (res[k] if k < len(res) else None, loo[k] if k < len(loo) else None)
            for k, i in enumerate(ids)
        }
        self.table.setRowCount(len(ls))
        for r, l in enumerate(ls.landmarks):
            rr, ll = rmap.get(l.id, (None, None))
            vals = [
                str(l.id),
                l.feature,
                ", ".join(f"{v:.0f}" for v in l.em_world_nm),
                ", ".join(f"{v:.0f}" for v in l.lm_world_nm),
                f"{l.sigma_nm[0]:.0f}/{l.sigma_nm[1]:.0f}",
                "" if rr is None else f"{rr:.0f}",
                "" if ll is None else f"{ll:.0f}",
                "yes" if l.enabled else "no",
            ]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()

    def _show_mask(self, viewer, vol: Volume, res, name: str) -> None:
        """Show what a refinement did: the crop mask as a Labels layer in world nm."""
        (z0, _), (y0, _), (x0, _) = res.bbox_voxel
        lvl = res.info.get("level", 0)
        A = vol.level_affine(lvl) @ np.array(
            [[1, 0, 0, z0], [0, 1, 0, y0], [0, 0, 1, x0], [0, 0, 0, 1]], dtype=float
        )
        for lay in list(viewer.layers):
            if lay.name == name:
                viewer.layers.remove(lay)
        viewer.add_labels(res.mask.astype(np.uint8), name=name, affine=A, opacity=0.4)

    def _centroid(self, side: str) -> None:
        lmk = self._selected_landmark()
        if lmk is None or self.em is None or self.lm is None:
            return self._error("select a landmark in the table")
        from ..refine.centroid import local_centroid

        try:
            if side == "em":
                res = local_centroid(
                    self.em,
                    self.em.world_to_voxel(np.asarray(lmk.em_world_nm)),
                    box_nm=self.box_nm.value(),
                    level=self.em.level_for_voxel_size(max(self.lm.voxel_size_nm[1:]) / 2),
                )
                lmk.em_world_nm = tuple(float(v) for v in res.centroid_world_nm)
                lmk.method["em"] = "centroid"
                self._show_mask(self.viewer, self.em, res, "refine mask (EM)")
            else:
                res = local_centroid(
                    self.lm,
                    self.lm.world_to_voxel(np.asarray(lmk.lm_world_nm)),
                    box_nm=self.box_nm.value(),
                    invert=False,
                )
                lmk.lm_world_nm = tuple(float(v) for v in res.centroid_world_nm)
                lmk.sigma_nm = tuple(
                    float(max(a, b))
                    for a, b in zip(res.sigma_nm, (0.5 * self.lm.voxel_size_nm[0], 0, 0))
                )
                lmk.method["lm"] = "centroid"
                if self.lm_viewer is not None:
                    self._show_mask(self.lm_viewer, self.lm, res, "refine mask (LM)")
        except Exception as e:
            return self._error(f"centroid: {e}")
        self.refine_status.setText(
            f"{side.upper()} centroid: {np.round(res.centroid_world_nm).tolist()} nm, {res.n_voxels} voxels, sigma {np.round(res.sigma_nm).tolist()} nm, inverted={res.inverted}"
        )
        self._after_landmark_change()

    def _paired(self) -> None:
        lmk = self._selected_landmark()
        if lmk is None or self.em is None or self.lm is None:
            return self._error("select a landmark in the table")
        from ..refine.centroid import paired_centroids

        try:
            r_em, r_lm = paired_centroids(
                self.em,
                self.lm,
                self.em.world_to_voxel(np.asarray(lmk.em_world_nm)),
                self.lm.world_to_voxel(np.asarray(lmk.lm_world_nm)),
                box_nm=self.box_nm.value(),
            )
        except Exception as e:
            return self._error(f"paired centroids: {e}")
        lmk.em_world_nm = tuple(float(v) for v in r_em.centroid_world_nm)
        lmk.lm_world_nm = tuple(float(v) for v in r_lm.centroid_world_nm)
        lmk.method = {"em": "paired-centroid", "lm": "paired-centroid"}
        self._show_mask(self.viewer, self.em, r_em, "refine mask (EM)")
        if self.lm_viewer is not None:
            self._show_mask(self.lm_viewer, self.lm, r_lm, "refine mask (LM)")
        self.refine_status.setText(
            f"paired centroids: EM {r_em.n_voxels} voxels, LM {r_lm.n_voxels} voxels"
        )
        self._after_landmark_change()

    def _midpoint(self, side: str) -> None:
        lmk = self._selected_landmark()
        if lmk is None or self.em is None or self.lm is None:
            return self._error("select a landmark in the table")
        from ..refine.centroid import midpoint_z

        vol = self.em if side == "em" else self.lm
        zc, sig = midpoint_z(self.mid_first.value(), self.mid_last.value(), vol.voxel_size_nm[0])
        p = np.asarray(lmk.em_world_nm if side == "em" else lmk.lm_world_nm, dtype=float)
        vox = vol.world_to_voxel(p)
        vox[0] = zc
        new = vol.voxel_to_world(vox)
        if side == "em":
            lmk.em_world_nm = tuple(float(v) for v in new)
            lmk.method["em"] = "midpoint"
        else:
            lmk.lm_world_nm = tuple(float(v) for v in new)
            lmk.sigma_nm = (float(sig), lmk.sigma_nm[1], lmk.sigma_nm[2])
            lmk.method["lm"] = "midpoint"
        self.refine_status.setText(
            f"{side.upper()} midpoint: z slice {zc:.1f}, sigma_z {sig:.0f} nm"
        )
        self._after_landmark_change()

    def _snap(self) -> None:
        lmk = self._selected_landmark()
        if lmk is None or self.em is None or self.lm is None:
            return self._error("select a landmark in the table")
        from ..refine.snap import snap_xy

        try:
            res = snap_xy(
                self.lm,
                self.em,
                self._transform(),
                np.asarray(lmk.lm_world_nm),
                crop_nm=self.box_nm.value() * 0.7,
                search_nm=self.box_nm.value() * 0.5,
            )
        except Exception as e:
            return self._error(f"snap: {e}")
        lmk.em_world_nm = tuple(float(v) for v in res.em_world_nm_new)
        lmk.method["em"] = "snap-xy"
        self.refine_status.setText(
            f"snap xy: moved EM point by {np.round(res.delta_lm_px, 1).tolist()} LM px, NCC {res.score:.2f}"
        )
        self._after_landmark_change()

    def _after_landmark_change(self) -> None:
        if self.session is not None:
            self.session.transform = None
            self.session.fit = None
        self._refresh_points()

    # -------------------------------------------------------------------- fit
    def _tab_fit(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.fit_kind = QComboBox()
        self.fit_kind.addItems(["rigid", "similarity", "affine", "tps"])
        self.fit_kind.setCurrentText("affine")
        self.fit_auto = QCheckBox("lambda by leave-one-out")
        self.fit_auto.setChecked(True)
        self.fit_lam = _spin(0, 1e4, 1.0, 0.1, 3)
        self.fit_loo = QCheckBox("leave-one-out RMS")
        self.fit_loo.setChecked(True)
        self.psf_aware = QCheckBox("PSF-aware LM sampling in the computed overlay")
        f.addRow("Model (3D affine is the default: shrinkage + z scaling)", self.fit_kind)
        row = QHBoxLayout()
        row.addWidget(self.fit_auto)
        row.addWidget(self.fit_lam)
        f.addRow("TPS lambda", row)
        f.addRow(self.fit_loo)
        f.addRow(self.psf_aware)
        b = QPushButton("Fit")
        b.clicked.connect(self._do_fit)
        f.addRow(b)
        self.fit_label = QLabel("no fit")
        self.fit_label.setWordWrap(True)
        f.addRow(self.fit_label)
        return w

    def _do_fit(self) -> None:
        if self.session is None or self.em is None or self.lm is None:
            return self._error("load both volumes and add landmarks first")
        sess = self.session
        sess.kind = self.fit_kind.currentText()
        sess.lam = "auto" if self.fit_auto.isChecked() else float(self.fit_lam.value())
        try:
            fr = sess.do_fit(with_loo=self.fit_loo.isChecked())
        except Exception as e:
            return self._error(f"fit: {e}")
        self.fit = fr
        self.disp = None
        if isinstance(fr.transform, TPSTransform):
            self.disp = sess.ensure_displacement(self.em)
        self.slab_cache = None
        self._update_fit_labels()
        self._refresh_table()
        self._refresh_overlay()
        self._refresh_residuals()
        self._say(fr.summary())
        for wmsg in fr.warnings:
            self._say(f"warning: {wmsg}")

    def _update_fit_labels(self) -> None:
        if self.session is None or not self.session.fit:
            self.fit_label.setText("no fit")
            return
        f = self.session.fit
        r = f["rms"]
        s = f"{f['kind']}: n={f['n']}  RMS {r['rms_total']:.0f} nm (xy {r['rms_xy']:.0f}, z {r['rms_z']:.0f}), max {r['max']:.0f}"
        if f.get("loo") and f["loo"].get("rms_total") is not None:
            s += f"\nLOO {f['loo']['rms_total']:.0f} nm (xy {f['loo']['rms_xy']:.0f}, z {f['loo']['rms_z']:.0f})"
        if f.get("lam") is not None:
            s += f"\nlambda {f['lam']:g}"
        if f.get("round_trip"):
            s += f"; inverse round-trip {f['round_trip']['rms_nm']:.1f} nm"
        zc = (f.get("qc") or {}).get("z_check") or {}
        if zc.get("suspect_z_scale"):
            s += "\nWARNING: z residual grows with z -> check z scale / tilt factor"
        if f.get("warnings"):
            s += "\n" + "\n".join(f["warnings"])
        self.fit_label.setText(s)

    def _refresh_overlay(self) -> None:
        """LM shown in the EM viewer: layer affine for linear transforms (free), a lazily
        computed image for deformable ones."""
        if self.lm is None:
            return
        t = self._transform()
        for lay in self.deform_layer or []:
            try:
                self.viewer.layers.remove(lay)
            except ValueError:
                pass
        self.deform_layer = None
        if t.is_linear:
            A = t.matrix @ self.lm.world_affine  # type: ignore[attr-defined]
            if not self.lm_overlay:
                self.lm_overlay = [
                    self.viewer.add_image(d, **m)
                    for d, m, _ in volume_layers(self.lm, "LM->EM ", transform=t.matrix)
                ]  # type: ignore[attr-defined]
            else:
                for lay in self.lm_overlay:
                    lay.affine = A
            for lay in self.lm_overlay:
                lay.visible = True
        else:
            if self.em is None or self.disp is None:
                return
            from ..resample.lazy import resample_to_grid

            for lay in self.lm_overlay:
                lay.visible = False
            lvl = em_level_for_lm(self.em, self.lm)
            grid = Grid.from_volume(self.em, lvl)
            data, _ = resample_to_grid(self.lm, t, grid, displacement=self.disp)
            layers = self.viewer.add_image(
                data,
                name="LM->EM (deformable, computed)",
                affine=grid.affine,
                blending="additive",
                colormap="green",
                channel_axis=0 if data.shape[0] > 1 else None,
            )
            self.deform_layer = list(layers) if isinstance(layers, list) else [layers]

    def _refresh_residuals(self) -> None:
        if (
            self.session is None
            or not self.session.fit
            or self.residual_layer is not None
            and False
        ):
            return
        f = self.session.fit
        src, dst, _, _ = self.session.landmarks.arrays()
        t = self._transform()
        pred = t.apply(src) if len(src) else np.zeros((0, 3))
        res = np.asarray(f["residuals_nm"], dtype=float).reshape(-1, 3)
        vec = np.stack([pred, res], axis=1) if len(pred) else np.zeros((0, 2, 3))
        if self.residual_layer is not None:
            try:
                self.viewer.layers.remove(self.residual_layer)
            except ValueError:
                pass
        self.residual_layer = self.viewer.add_vectors(
            vec,
            name="residuals (model -> pick)",
            edge_color="magenta",
            edge_width=max(self.em.voxel_size_nm) * 4 if self.em else 20,
            length=1,
        )

    # ----------------------------------------------------------------- verify
    def _tab_verify(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.zmode = QComboBox()
        self.zmode.addItems(
            [
                "EM-driven (scroll EM z, LM resampled at that plane)",
                "LM-driven (scroll LM z, EM slab projected into the LM grid)",
            ]
        )
        self.zmode.currentIndexChanged.connect(self._zmode_changed)
        f.addRow("Z mode", self.zmode)
        self.slab_thick = QComboBox()
        self.slab_thick.addItems(["slice", "psf"])
        self.slab_proj = QComboBox()
        self.slab_proj.addItems(["mean", "min", "max", "gaussian"])
        row = QHBoxLayout()
        row.addWidget(self.slab_thick)
        row.addWidget(self.slab_proj)
        f.addRow("Slab thickness / projection", row)
        self.cmp_mode = QComboBox()
        self.cmp_mode.addItems(["blend", "swipe", "checker", "flicker"])
        self.cmp_mode.currentIndexChanged.connect(self._compare_changed)
        f.addRow("Compare", self.cmp_mode)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(60)
        self.opacity.valueChanged.connect(self._opacity_changed)
        f.addRow("LM opacity / swipe position", self.opacity)
        self.z_label = QLabel("")
        self.z_label.setWordWrap(True)
        f.addRow("z readout", self.z_label)
        b = QPushButton("z-profile at last EM point")
        b.clicked.connect(self._zprofile)
        f.addRow(b)
        self.zprof_label = QLabel("")
        self.zprof_label.setWordWrap(True)
        f.addRow(self.zprof_label)
        return w

    def _zmode_changed(self, idx: int) -> None:
        if self.lm_viewer is None or self.em is None or self.lm is None:
            return
        if idx == 1:
            self._update_slab_layer()
        elif self.slab_layer is not None:
            self.slab_layer.visible = False

    def _slab_params(self) -> SlabParams:
        return SlabParams(
            thickness=self.slab_thick.currentText(), projection=self.slab_proj.currentText()
        )

    def _update_slab_layer(self) -> None:
        if self.lm_viewer is None or self.em is None or self.lm is None:
            return
        if not self._alive(self.lm_viewer, self.lm_raw_layers):
            return
        if self.slab_cache is None:
            self.slab_cache = SlabCache(self.em, self.lm, self._transform())
        k = (
            int(round(self.lm_viewer.dims.point[0] / self.lm.voxel_size_nm[0]))
            if self.lm_viewer.dims.ndim >= 3
            else 0
        )
        k = int(np.clip(k, 0, self.lm.shape_zyx[0] - 1))
        img, cov, info = self.slab_cache.get(k, self._slab_params())
        Z, Y, X = self.lm.shape_zyx
        vol3 = np.zeros((Z, Y, X), np.float32)
        vol3[k] = img
        if self.slab_layer is None or self.slab_layer not in self.lm_viewer.layers:
            self.slab_layer = self.lm_viewer.add_image(
                vol3,
                name="EM slab (LM-driven)",
                affine=self.lm.world_affine,
                blending="additive",
                colormap="gray",
                opacity=0.7,
            )
        else:
            self.slab_layer.data = vol3
            self.slab_layer.visible = True
        self.slab_layer.contrast_limits = (
            float(img[cov > 0].min()) if (cov > 0).any() else 0.0,
            float(img[cov > 0].max()) if (cov > 0).any() else 1.0,
        )

    def _compare_changed(self, idx: int) -> None:
        mode = self.cmp_mode.currentText()
        self._flicker.stop()
        if mode == "flicker":
            self._flicker.start(500)
        else:
            for lay in self.lm_overlay:
                lay.visible = True
        if mode in ("swipe", "checker"):
            self._update_compare_layer()
        elif self.compare_layer is not None:
            self.compare_layer.visible = False
            for lay in self.lm_overlay:
                lay.visible = True

    def _flicker_tick(self) -> None:
        for lay in self.lm_overlay:
            lay.visible = not lay.visible
        for lay in self.deform_layer or []:
            lay.visible = not lay.visible

    def _opacity_changed(self, v: int) -> None:
        for lay in self.lm_overlay:
            lay.opacity = v / 100.0
        for lay in self.deform_layer or []:
            lay.opacity = v / 100.0
        if self.cmp_mode.currentText() in ("swipe", "checker"):
            self._update_compare_layer()

    def _update_compare_layer(self) -> None:
        """Swipe / checker on the current EM slice via a computed RGB plane (any transform)."""
        if self.em is None or self.lm is None or not self._alive(self.viewer, self.em_layers):
            return
        from ..export.report import composite_rgb
        from ..resample.zview import lm_plane_at_em_slice

        lvl = em_level_for_lm(self.em, self.lm)
        j = (
            int(round(self.viewer.dims.point[0] / self.em.level_voxel_size_nm(lvl)[0]))
            if self.viewer.dims.ndim >= 3
            else 0
        )
        j = int(np.clip(j, 0, self.em.level_data(lvl).shape[1] - 1))
        em_plane = np.asarray(self.em.level_data(lvl)[0, j].compute())
        planes, _ = lm_plane_at_em_slice(
            self.lm,
            self.em,
            self._transform(),
            j,
            em_level=lvl,
            displacement=self.disp,
            psf_aware=self.psf_aware.isChecked(),
        )
        rgb = composite_rgb(
            em_plane,
            [planes[i] for i in range(planes.shape[0])],
            self.lm.channels,
            mode=self.cmp_mode.currentText(),
            swipe_frac=self.opacity.value() / 100.0,
        )
        vs = self.em.level_voxel_size_nm(lvl)
        A = self.em.level_affine(lvl) @ np.array(
            [[1, 0, 0, j], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )
        data = rgb[None]
        for lay in self.lm_overlay:
            lay.visible = False
        if self.compare_layer is None or self.compare_layer not in self.viewer.layers:
            self.compare_layer = self.viewer.add_image(
                data, name="compare (computed)", rgb=True, affine=A
            )
        else:
            self.compare_layer.data = data
            self.compare_layer.affine = A
            self.compare_layer.visible = True
        _ = vs

    def _zprofile(self) -> None:
        if (
            self.em is None
            or self.lm is None
            or self.em_points is None
            or len(self.em_points.data) == 0
        ):
            return self._error("click an EM point first")
        from ..resample.zview import z_profile

        p = np.asarray(self.em_points.data[-1], dtype=float)
        prof = z_profile(self.lm, self.em, self._transform(), p, displacement=self.disp)
        self.zprof_label.setText(
            f"LM intensity along z through the mapped EM point peaks at {prof['peak_offset_nm']:+.0f} nm from the mapped z (LM voxel {np.round(prof['lm_voxel'], 1).tolist()})"
        )

    # ---------------------------------------------------------------- events
    def _on_em_dims(self, event=None) -> None:
        if self.em is None or self.lm is None or not self._alive(self.viewer, self.em_layers):
            return
        try:
            from ..resample.zview import z_readout

            j = (
                int(round(self.viewer.dims.point[0] / self.em.voxel_size_nm[0]))
                if self.viewer.dims.ndim >= 3
                else 0
            )
            j = int(np.clip(j, 0, self.em.shape_zyx[0] - 1))
            self.z_label.setText(
                z_readout(self.lm, self.em, self._transform(), em_j=j, displacement=self.disp)
            )
            if self.cmp_mode.currentText() in ("swipe", "checker"):
                self._update_compare_layer()
            if self._sync and self.lm_viewer is not None and self.zmode.currentIndex() == 0:
                q = self._inverse_point(
                    np.array(
                        [
                            self.viewer.dims.point[0],
                            self.viewer.camera.center[1],
                            self.viewer.camera.center[2],
                        ]
                    )
                )
                self.lm_viewer.dims.set_point(0, float(q[0]))
        except Exception as e:  # pragma: no cover - never break the viewer
            log.debug("dims callback: %s", e)

    def _on_lm_dims(self, event=None) -> None:
        if (
            self.em is None
            or self.lm is None
            or not self._alive(self.lm_viewer, self.lm_raw_layers)
        ):
            return
        try:
            from ..resample.zview import z_readout

            k = (
                int(round(self.lm_viewer.dims.point[0] / self.lm.voxel_size_nm[0]))
                if self.lm_viewer.dims.ndim >= 3
                else 0
            )
            k = int(np.clip(k, 0, self.lm.shape_zyx[0] - 1))
            self.z_label.setText(z_readout(self.lm, self.em, self._transform(), lm_k=k))
            if self.zmode.currentIndex() == 1:
                self._update_slab_layer()
                if self._sync:
                    r = self._transform().apply(
                        np.array(
                            [
                                self.lm_viewer.dims.point[0],
                                self.lm_viewer.camera.center[1],
                                self.lm_viewer.camera.center[2],
                            ]
                        )
                    )
                    self.viewer.dims.set_point(0, float(r[0]))
        except Exception as e:  # pragma: no cover
            log.debug("lm dims callback: %s", e)

    def _inverse_point(self, p: np.ndarray) -> np.ndarray:
        t = self._transform()
        if t.is_linear:
            return t.inverse().apply(p)
        if self.disp is not None:
            return self.disp.inverse(p[None])[0]
        return t.affine_part().inverse().apply(p)  # type: ignore[attr-defined]

    def _on_em_mouse(self, viewer, event) -> None:
        if not self._sync or self.cursor_lm is None or self.lm is None:
            return
        try:
            p = np.asarray(viewer.cursor.position, dtype=float)[-3:]
            self.cursor_lm.data = self._inverse_point(p)[None]
        except Exception:  # pragma: no cover
            pass

    def _on_lm_mouse(self, viewer, event) -> None:
        if not self._sync or self.cursor_em is None:
            return
        try:
            p = np.asarray(viewer.cursor.position, dtype=float)[-3:]
            self.cursor_em.data = self._transform().apply(p)[None]
        except Exception:  # pragma: no cover
            pass

    # ----------------------------------------------------------------- export
    def _tab_export(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.ex_fused = QLineEdit()
        f.addRow(
            "Fused OME-Zarr", self._path_row(self.ex_fused, save=True, filt="OME-Zarr (*.zarr)")
        )
        self.ex_level = QSpinBox()
        self.ex_level.setRange(-1, 10)
        self.ex_level.setValue(-1)
        self.ex_roi = QCheckBox("crop to the current EM view (full-res ROI)")
        row = QHBoxLayout()
        row.addWidget(QLabel("EM level (-1 = nearest LM xy)"))
        row.addWidget(self.ex_level)
        row.addWidget(self.ex_roi)
        f.addRow(row)
        self.ex_tf = QLineEdit()
        f.addRow(
            "Transforms dir (JSON/ITK/BigWarp/NRRD)", self._path_row(self.ex_tf, directory=True)
        )
        self.ex_bdv = QLineEdit()
        f.addRow("BigDataViewer XML", self._path_row(self.ex_bdv, save=True, filt="BDV (*.xml)"))
        self.ex_emlm = QLineEdit()
        f.addRow(
            "EM-in-LM (OME-TIFF / .zarr)",
            self._path_row(self.ex_emlm, save=True, filt="OME-TIFF (*.ome.tif);;OME-Zarr (*.zarr)"),
        )
        self.ex_figs = QLineEdit()
        f.addRow(
            "Figures dir (current EM & LM slices)", self._path_row(self.ex_figs, directory=True)
        )
        self.ex_report = QLineEdit()
        f.addRow("report.json", self._path_row(self.ex_report, save=True, filt="JSON (*.json)"))
        b = QPushButton("Export selected outputs")
        b.clicked.connect(self._export)
        f.addRow(b)
        return w

    def _export(self) -> None:
        if self.session is None or self.em is None or self.lm is None:
            return self._error("load, add landmarks and fit first")
        t = self._transform()
        if self.session.transform is None:
            return self._error("fit a transform first")
        sess, em, lm, disp = self.session, self.em, self.lm, self.disp
        fused, tf, bdv, emlm, figs, rep = (
            x.text()
            for x in (
                self.ex_fused,
                self.ex_tf,
                self.ex_bdv,
                self.ex_emlm,
                self.ex_figs,
                self.ex_report,
            )
        )
        lvl = None if self.ex_level.value() < 0 else self.ex_level.value()
        roi = None
        if self.ex_roi.isChecked():
            c = np.asarray(self.viewer.camera.center, dtype=float)
            half = (
                np.asarray(
                    em.world_extent_nm()
                    if hasattr(em, "world_extent_nm")
                    else em.bbox_world()[1] - em.bbox_world()[0]
                )
                * 0.1
            )
            roi = np.array([c - half, c + half])
        em_j = (
            int(round(self.viewer.dims.point[0] / em.voxel_size_nm[0]))
            if self.viewer.dims.ndim >= 3
            else 0
        )
        lm_k = (
            int(round(self.lm_viewer.dims.point[0] / lm.voxel_size_nm[0]))
            if self.lm_viewer is not None and self.lm_viewer.dims.ndim >= 3
            else 0
        )
        mode = self.cmp_mode.currentText() if self.cmp_mode.currentText() != "flicker" else "blend"
        slab = self._slab_params()

        def work():
            from ..export.emlm import export_em_in_lm
            from ..export.fused import export_fused_ome_zarr
            from ..export.report import build_report, figure_em_slice, figure_lm_slice, write_report
            from ..export.transforms import export_all_transforms, export_bdv_xml_h5

            done: dict[str, Any] = {}
            if fused:
                done["fused_ome_zarr"] = str(
                    export_fused_ome_zarr(
                        em, lm, t, fused, em_level=lvl, roi_world=roi, displacement=disp
                    )
                )
            if tf:
                done["transforms"] = export_all_transforms(t, sess.landmarks, tf, displacement=disp)
            if bdv:
                done["bdv_xml"] = str(export_bdv_xml_h5(em, lm, t, bdv, em_level=lvl))
            if emlm:
                done["em_in_lm"] = str(
                    export_em_in_lm(
                        em,
                        lm,
                        t,
                        emlm,
                        fmt="ome-zarr" if emlm.endswith(".zarr") else "ome-tiff",
                        slab=slab,
                    )
                )
            if figs:
                done["figures"] = [
                    figure_em_slice(
                        em,
                        lm,
                        t,
                        em_j,
                        Path(figs) / f"em_z{em_j:05d}_{mode}.png",
                        mode=mode,
                        displacement=disp,
                    ),
                    figure_lm_slice(
                        em,
                        lm,
                        t,
                        lm_k,
                        Path(figs) / f"lm_z{lm_k:04d}_{mode}.png",
                        slab=slab,
                        mode=mode,
                    ),
                ]
            if rep:
                r = build_report(
                    em,
                    lm,
                    sess.landmarks,
                    self.fit,
                    align_stats=(self.align_result.stats if self.align_result else None),
                    session={
                        "path": sess.path,
                        "kind": sess.kind,
                        "lambda": sess.lam,
                        "prealign": sess.prealign.to_dict(),
                    },
                    outputs=done,
                )
                write_report(r, rep)
                done["report"] = rep
            return done

        def finished(done):
            sess.outputs.update(done)
            for k, v in done.items():
                self._say(f"{k}: {v}")
            if sess.path:
                sess.save()

        self._run_bg(work, finished, "exporting")
