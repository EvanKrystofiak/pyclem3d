"""Export tests (plan §9, §11): fused OME-Zarr sizes/metadata, transform files, EM-in-LM, report, figures."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pyclem3d.export import (
    build_report,
    composite_rgb,
    export_all_transforms,
    export_bdv_xml_h5,
    export_em_in_lm,
    export_fused_ome_zarr,
    export_transform_json,
    figure_em_slice,
    figure_lm_slice,
    import_bigwarp_csv,
    read_itk_tfm,
    read_nrrd_header,
    write_report,
    zyx_to_xyz,
)
from pyclem3d.io import ensure_pyramid, open_volume
from pyclem3d.phantom import make_phantom
from pyclem3d.register import (
    AffineTransform,
    LandmarkSet,
    default_sigma_nm,
    fit_landmarks,
    fit_transform,
)
from pyclem3d.resample import DisplacementGrid, SlabParams, em_level_for_lm, em_slab_for_lm_slice


@pytest.fixture(scope="module")
def phantom():
    em, lm, truth = make_phantom(
        em_shape=(32, 128, 128),
        n_nuclei=6,
        n_organelles=20,
        seed=31,
        lm_voxel_nm=(240.0, 80.0, 80.0),
    )
    em = ensure_pyramid(em, cache_dir=None, min_size=16)
    T = AffineTransform(truth.lm_to_em, "affine")
    ls = LandmarkSet()
    sig = default_sigma_nm(lm, em)
    for c_em, c_lm in zip(truth.nuclei_em_world, truth.nuclei_lm_world):
        ls.add(tuple(c_em), tuple(c_lm), sig, feature="nucleus")
    return em, lm, truth, T, ls


def test_fused_ome_zarr(phantom, tmp_path):
    em, lm, truth, T, ls = phantom
    out = export_fused_ome_zarr(em, lm, T, tmp_path / "fused.zarr", min_size=8)
    back = open_volume(out, kind="em", memory="skip", pyramid="lazy")
    lvl = em_level_for_lm(em, lm)
    assert back.shape_zyx == tuple(em.level_data(lvl).shape[1:])
    assert back.n_channels == 1 + lm.n_channels + 1
    assert [c.name for c in back.channels] == ["EM:ch0", "LM:nuclei", "LM:organelles", "coverage"]
    assert np.allclose(back.voxel_size_nm, em.level_voxel_size_nm(lvl))
    # EM channel is the EM level, untouched (widened to the common dtype)
    assert np.array_equal(
        back.data[0].compute(), em.level_data(lvl)[0].compute().astype(back.dtype)
    )
    cov = np.asarray(back.data[-1].compute())
    assert (cov > 0).mean() > 0.99
    # nuclei channel is bright at the true nuclei
    nuc = np.asarray(back.data[1].compute()).astype(float)
    for c in truth.nuclei_em_world[:3]:
        v = np.clip(np.round(back.world_to_voxel(c)).astype(int), 0, np.array(back.shape_zyx) - 1)
        assert nuc[v[0], v[1], v[2]] > 2 * np.median(nuc)
    attrs = back.metadata["ome_zarr_attrs"]
    assert attrs["multiscales"][0]["axes"][1]["unit"] == "nanometer"
    assert back.n_levels() >= 2
    # a full-resolution ROI crop
    bb = em.bbox_world()
    roi = np.array([bb[0] + (bb[1] - bb[0]) * 0.3, bb[0] + (bb[1] - bb[0]) * 0.6])
    out2 = export_fused_ome_zarr(
        em, lm, T, tmp_path / "roi.zarr", em_level=0, roi_world=roi, min_size=8
    )
    back2 = open_volume(out2, kind="em", memory="skip", pyramid="none")
    assert all(s < f for s, f in zip(back2.shape_zyx, em.shape_zyx))
    assert np.allclose(back2.voxel_size_nm, em.voxel_size_nm)
    # origin of the ROI sits inside the requested box (in EM world nm)
    origin = back2.world_affine[:3, 3]
    assert np.all(origin >= roi[0] - np.asarray(em.voxel_size_nm)) and np.all(origin <= roi[1])


def test_transform_exports(phantom, tmp_path):
    em, lm, truth, T, ls = phantom
    d = tmp_path / "tf"
    written = export_all_transforms(T, ls, d, unit="um")
    js = json.loads((d / "transform.json").read_text())
    assert np.allclose(js["lm_to_em_zyx"], T.matrix)
    assert np.allclose(zyx_to_xyz(np.asarray(js["lm_to_em_xyz"])), T.matrix)  # involution
    # xyz convention check: a point (z,y,x) maps like (x,y,z) through the xyz matrix
    p = np.array([100.0, 200.0, 300.0])
    q = T.apply(p)
    M = np.asarray(js["lm_to_em_xyz"])
    assert np.allclose((M @ np.array([p[2], p[1], p[0], 1.0]))[:3], q[::-1])
    back = read_itk_tfm(written["lm_to_em"], unit="um")
    assert np.allclose(back, T.matrix, atol=1e-6)
    inv = read_itk_tfm(written["em_to_lm_resample"], unit="um")
    assert np.allclose(inv @ T.matrix, np.eye(4), atol=1e-6)
    ls2 = import_bigwarp_csv(written["bigwarp_csv"], unit="um")
    assert len(ls2) == len(ls)
    a = np.array([l.em_world_nm for l in ls])
    b = np.array([l.em_world_nm for l in ls2])
    assert np.allclose(a, b, atol=1e-2)
    a = np.array([l.lm_world_nm for l in ls])
    b = np.array([l.lm_world_nm for l in ls2])
    assert np.allclose(a, b, atol=1e-2)
    # TPS: displacement NRRD
    src, dst, sig, _ = ls.arrays()
    rng = np.random.default_rng(0)
    extra = rng.uniform(lm.bbox_world()[0], lm.bbox_world()[1], (8, 3))
    S = np.vstack([src, extra])
    D = np.vstack([dst, T.apply(extra) + rng.normal(0, 20, extra.shape)])
    t = fit_transform("tps", S, D, lam=1.0).transform
    dg = DisplacementGrid.build(t, em.bbox_world(), spacing_nm=1000.0)
    w2 = export_all_transforms(t, ls, tmp_path / "tf2", displacement=dg, unit="um")
    hdr = read_nrrd_header(w2["displacement_nrrd"])
    assert hdr["kinds"] == "vector domain domain domain"
    sizes = [int(v) for v in hdr["sizes"].split()]
    assert sizes[0] == 3 and sizes[1:] == [
        dg.values.shape[2],
        dg.values.shape[1],
        dg.values.shape[0],
    ]
    export_transform_json(t, tmp_path / "tps.json")
    assert "affine_part_xyz" in json.loads((tmp_path / "tps.json").read_text())


def test_bdv_xml_h5(phantom, tmp_path):
    h5py = pytest.importorskip("h5py")
    em, lm, truth, T, ls = phantom
    out = export_bdv_xml_h5(em, lm, T, tmp_path / "bdv.xml", unit="um")
    root = ET.parse(out).getroot()
    setups = root.findall(".//ViewSetup")
    assert len(setups) == 1 + lm.n_channels
    regs = {int(r.get("setup")): r for r in root.findall(".//ViewRegistration")}
    lm_aff = np.array(
        [float(v) for v in regs[1].find("ViewTransform/affine").text.split()]
    ).reshape(3, 4)
    expect = zyx_to_xyz(T.matrix @ lm.world_affine)
    expect[:3, 3] *= 1e-3
    assert np.allclose(lm_aff, expect[:3, :4], atol=1e-6)
    with h5py.File(out.with_suffix(".h5"), "r") as h5:
        lvl = em.level_for_voxel_size(max(lm.voxel_size_nm[1:]))
        assert h5["t00000/s00/0/cells"].shape == tuple(em.level_data(lvl).shape[1:])
        assert h5["s00/resolutions"].shape[1] == 3
        assert h5["t00000/s01/0/cells"].shape == lm.shape_zyx
        assert h5["t00000/s01/0/cells"].dtype == np.int16
        lm_back = h5["t00000/s01/0/cells"][:].view(np.uint16)
        assert np.array_equal(lm_back, lm.data[0].compute())


def test_em_in_lm_export(phantom, tmp_path):
    em, lm, truth, T, ls = phantom
    p = SlabParams(thickness="slice", projection="mean")
    out = export_em_in_lm(em, lm, T, tmp_path / "em_in_lm.ome.tif", fmt="ome-tiff", slab=p)
    back = open_volume(out, kind="lm", memory="skip", pyramid="none")
    assert back.shape_zyx == lm.shape_zyx
    assert back.n_channels == 1 + lm.n_channels + 1
    assert np.allclose(back.voxel_size_nm, lm.voxel_size_nm)
    k = lm.shape_zyx[0] // 2
    ref, cov, _ = em_slab_for_lm_slice(em, lm, T, k, p)
    got = np.asarray(back.data[0, k].compute()).astype(float)
    assert np.abs(got - np.round(ref)).max() <= 1.0
    assert np.array_equal(back.data[1].compute(), lm.data[0].compute())
    out2 = export_em_in_lm(em, lm, T, tmp_path / "em_in_lm.zarr", fmt="ome-zarr", slab=p)
    assert open_volume(out2, kind="lm", memory="skip", pyramid="none").shape_zyx == lm.shape_zyx


def test_report_and_figures(phantom, tmp_path):
    em, lm, truth, T, ls = phantom
    fr = fit_landmarks(ls, "affine", lam=None)
    rep = build_report(
        em,
        lm,
        ls,
        fr,
        align_stats={"jitter_rms_px_before": 0.3, "flagged": [4]},
        outputs={"x": "y"},
    )
    assert rep["fit"]["rms_nm"]["rms_total"] < 1e-3
    assert rep["fit"]["loo_rms_nm"]["n_evaluated"] == len(ls)
    assert len(rep["fit"]["per_point"]) == len(ls)
    assert any("flagged" in w for w in rep["warnings"])
    assert rep["em"]["voxel_size_nm"] == list(em.voxel_size_nm)
    p = write_report(rep, tmp_path / "report.json")
    assert json.loads(p.read_text())["fit"]["kind"] == "affine"
    j = em.shape_zyx[0] // 2
    f1 = figure_em_slice(em, lm, T, j, tmp_path / "figs" / "em.png", mode="blend")
    f2 = figure_lm_slice(
        em, lm, T, lm.shape_zyx[0] // 2, tmp_path / "figs" / "lm.png", mode="checker"
    )
    import imageio.v3 as iio

    img = iio.imread(f1["path"])
    lvl = em_level_for_lm(em, lm)
    assert img.shape == (*em.level_data(lvl).shape[2:], 3)
    assert iio.imread(f2["path"]).shape == (*lm.shape_zyx[1:], 3)
    assert f1["scale_bar_nm"] > 0 and "EM z" in f1["readout"]
    for mode in ("blend", "checker", "swipe", "em", "lm"):
        rng = np.random.default_rng(0)
        rgb = composite_rgb(
            rng.random((20, 30)), [rng.random((20, 30))], [lm.channels[0]], mode=mode
        )
        assert rgb.shape == (20, 30, 3) and rgb.dtype == np.uint8
