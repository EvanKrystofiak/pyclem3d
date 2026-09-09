"""Session + CLI tests (plan §11): the CLI reproduces the GUI headlessly, session round-trips,
and every command runs end-to-end on the phantom."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyclem3d.cli import main
from pyclem3d.io import open_volume
from pyclem3d.session import Session


@pytest.fixture(scope="module")
def phantom_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("phantom")
    rc = main(
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
            "--n-organelles",
            "20",
            "--seed",
            "7",
            "--misalign",
            "--lm-voxel",
            "240",
            "80",
            "80",
        ]
    )
    assert rc == 0
    return d


def test_phantom_files_and_session(phantom_dir):
    assert (phantom_dir / "em.ome.tif").exists() and (phantom_dir / "lm.ome.tif").exists()
    assert (phantom_dir / "em_misaligned.ome.tif").exists()
    truth = json.loads((phantom_dir / "truth.json").read_text())
    assert "lm_to_em" in truth and "misalignment" in truth
    sess = Session.load(phantom_dir / "session.json")
    assert len(sess.landmarks) == 8 and sess.kind == "affine"
    em, lm = sess.open_volumes(memory="skip")
    assert em.kind == "em" and lm.kind == "lm" and lm.psf_nm is not None


def test_register_reproduces_truth_and_is_deterministic(phantom_dir, tmp_path):
    sess_path = phantom_dir / "session.json"
    rep1 = tmp_path / "r1.json"
    assert main(["register", str(sess_path), "--report", str(rep1)]) == 0
    r1 = json.loads(rep1.read_text())
    truth = json.loads((phantom_dir / "truth.json").read_text())
    assert r1["fit"]["rms_nm"]["rms_total"] < 1e-3
    sess = Session.load(sess_path)
    assert sess.transform is not None
    assert np.allclose(
        np.asarray(sess.transform["matrix"]), np.asarray(truth["lm_to_em"]), atol=1e-6
    )
    assert Path(sess.outputs["report"]).exists()
    # 'report' reproduces bit-for-bit
    rep2 = tmp_path / "r2.json"
    assert main(["report", str(sess_path), "--report", str(rep2)]) == 0
    r2 = json.loads(rep2.read_text())
    assert json.dumps(r1["fit"]["transform"]) == json.dumps(r2["fit"]["transform"])
    assert r1["fit"]["rms_nm"] == r2["fit"]["rms_nm"]
    # other kinds run too
    assert (
        main(
            [
                "report",
                str(sess_path),
                "--kind",
                "similarity",
                "--report",
                str(tmp_path / "r3.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "report",
                str(sess_path),
                "--kind",
                "tps",
                "--lambda",
                "auto",
                "--report",
                str(tmp_path / "r4.json"),
            ]
        )
        == 0
    )
    r4 = json.loads((tmp_path / "r4.json").read_text())
    assert r4["fit"]["kind"] == "tps" and r4["fit"]["round_trip"] is not None


def test_landmarks_command(phantom_dir, tmp_path):
    sess_path = phantom_dir / "session.json"
    n0 = len(Session.load(sess_path).landmarks)
    assert main(["landmarks", str(sess_path), "--list"]) == 0
    assert (
        main(
            [
                "landmarks",
                str(sess_path),
                "--add",
                "100",
                "200",
                "300",
                "110",
                "220",
                "330",
                "--feature",
                "mito",
            ]
        )
        == 0
    )
    s = Session.load(sess_path)
    assert len(s.landmarks) == n0 + 1 and s.transform is None
    new_id = s.landmarks.landmarks[-1].id
    assert main(["landmarks", str(sess_path), "--disable", str(new_id)]) == 0
    assert not Session.load(sess_path).landmarks.get(new_id).enabled
    csv = tmp_path / "lm.csv"
    assert main(["landmarks", str(sess_path), "--export-bigwarp", str(csv)]) == 0
    assert csv.exists()
    assert main(["landmarks", str(sess_path), "--remove", str(new_id)]) == 0
    assert len(Session.load(sess_path).landmarks) == n0
    assert main(["landmarks", str(sess_path), "--import-bigwarp", str(csv)]) == 0
    assert len(Session.load(sess_path).landmarks) == 2 * n0 + 1
    # restore: remove imported ones
    s = Session.load(sess_path)
    for lm in list(s.landmarks.landmarks)[n0:]:
        s.landmarks.remove(lm.id)
    s.save()
    assert main(["register", str(sess_path)]) == 0


def test_export_command(phantom_dir, tmp_path):
    sess_path = phantom_dir / "session.json"
    assert main(["register", str(sess_path)]) == 0
    fused = tmp_path / "fused.zarr"
    tdir = tmp_path / "tf"
    emlm = tmp_path / "em_in_lm.ome.tif"
    figs = tmp_path / "figs"
    rc = main(
        [
            "export",
            str(sess_path),
            "--fused",
            str(fused),
            "--transforms",
            str(tdir),
            "--em-in-lm",
            str(emlm),
            "--figures",
            str(figs),
            "--em-z",
            "10,16",
            "--lm-z",
            "3",
            "--min-size",
            "16",
        ]
    )
    assert rc == 0
    assert fused.exists() and (tdir / "transform.json").exists() and emlm.exists()
    assert len(list(figs.glob("*.png"))) == 3
    back = open_volume(fused, kind="em", memory="skip")
    assert back.n_channels == 4
    sess = Session.load(sess_path)
    assert "fused_ome_zarr" in sess.outputs and "figures" in sess.outputs
    # BDV export via CLI (h5py in dev extras)
    pytest.importorskip("h5py")
    assert main(["export", str(sess_path), "--bdv", str(tmp_path / "bdv.xml")]) == 0
    assert (tmp_path / "bdv.h5").exists()


def test_align_command(phantom_dir, tmp_path):
    stack = phantom_dir / "em_misaligned.ome.tif"
    sidecar = tmp_path / "align.json"
    out = tmp_path / "aligned.zarr"
    rc = main(
        [
            "align",
            str(stack),
            "--out",
            str(out),
            "--sidecar",
            str(sidecar),
            "--drift",
            "remove",
            "--level",
            "0",
            "--min-size",
            "16",
        ]
    )
    assert rc == 0
    assert sidecar.exists()
    sc = json.loads(sidecar.read_text())
    truth = json.loads((phantom_dir / "truth.json").read_text())["misalignment"]
    c = np.asarray(sc["shifts_yx"])
    total = np.asarray(truth["total"])
    good = np.ones(len(c), bool)
    good[truth["bad_slices"]] = False
    assert np.median(np.abs(c[good] + total[good])) <= 1.0
    assert set(truth["bad_slices"]) <= set(sc["flagged"])
    back = open_volume(out, kind="em", memory="skip")
    assert back.shape_zyx[0] == len(c)
    # verify mode on the aligned result: jitter is gone
    assert main(["align", str(out), "--verify", "--level", "0"]) == 0
    # a session can reference the sidecar and opens the aligned stack lazily
    sess = tmp_path / "s.json"
    assert (
        main(
            [
                "session",
                "init",
                "--em",
                str(stack),
                "--lm",
                str(phantom_dir / "lm.ome.tif"),
                "--out",
                str(sess),
                "--em-align-sidecar",
                str(sidecar),
            ]
        )
        == 0
    )
    em, lm = Session.load(sess).open_volumes(memory="skip")
    assert em.per_slice_transforms is not None and em.shape_zyx == back.shape_zyx
    assert main(["session", "show", str(sess)]) == 0


def test_info_convert_pyramid_doctor(phantom_dir, tmp_path):
    em = phantom_dir / "em.ome.tif"
    assert main(["info", str(em)]) == 0
    assert main(["info", str(em), "--json"]) == 0
    assert main(["convert", str(em), "--out", str(tmp_path / "em.zarr"), "--min-size", "16"]) == 0
    back = open_volume(tmp_path / "em.zarr", kind="em", memory="skip")
    assert back.n_levels() >= 2
    assert (
        main(["pyramid", str(em), "--cache-dir", str(tmp_path / "cache"), "--min-size", "16"]) == 0
    )
    assert any(p.name.endswith(".pyr.zarr") for p in (tmp_path / "cache").iterdir())
    assert main(["doctor", "--no-write", "--data", str(phantom_dir)]) == 0
    cfg = tmp_path / "config.json"
    assert main(["doctor", "--config", str(cfg), "--gpu", "off"]) == 0
    assert json.loads(cfg.read_text())["gpu"] is False
    assert main(["info", str(tmp_path / "nope.tif")]) == 1
