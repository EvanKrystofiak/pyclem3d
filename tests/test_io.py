"""Data layer tests: axes, world affines, pyramid, memory, readers/writers round trips."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import tifffile

from pyclem3d.io import (
    Channel,
    apply_affine,
    apply_memory_strategy,
    build_pyramid,
    ensure_pyramid,
    make_world_affine,
    normalize_axes,
    open_volume,
    plan_levels,
    volume_from_array,
    write_mrc,
    write_ome_tiff,
    write_ome_zarr,
)
from pyclem3d.io.metadata import parse_ome_physical_sizes, psf_fwhm_from_optics, to_nm
from pyclem3d.io.pyramid import coarsen, next_factors


def test_normalize_axes_variants():
    a = da.zeros((5, 4, 3), chunks=(1, 4, 3))
    assert normalize_axes(a, "ZYX").shape == (1, 5, 4, 3)
    b = da.zeros((2, 5, 4, 3), chunks=-1)
    assert normalize_axes(b, "CZYX").shape == (2, 5, 4, 3)
    c = da.zeros((5, 2, 4, 3), chunks=-1)  # ImageJ ZCYX
    assert normalize_axes(c, "ZCYX").shape == (2, 5, 4, 3)
    d = da.zeros((1, 2, 5, 4, 3), chunks=-1)
    assert normalize_axes(d, "TCZYX").shape == (2, 5, 4, 3)
    e = da.zeros((7, 4, 3), chunks=-1)  # ImageJ frames
    assert normalize_axes(e, "IYX").shape == (1, 7, 4, 3)
    f = da.zeros((4, 3, 3), chunks=-1)  # RGB samples -> channels
    assert normalize_axes(f, "YXS").shape == (3, 1, 4, 3)
    warns: list[str] = []
    g = da.zeros((3, 5, 4, 3), chunks=-1)
    normalize_axes(g, "TZYX", warns)
    assert warns and "taking index 0" in warns[0]


def test_world_affine_scale_yscale_flip():
    A = make_world_affine((300.0, 100.0, 100.0), origin_nm=(10, 20, 30), y_scale=1.2)
    p = apply_affine(A, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(p, [310.0, 20 + 2 * 120.0, 330.0])
    B = make_world_affine((8.0, 8.0, 8.0), flips=(False, True, False), shape_zyx=(10, 10, 10))
    assert np.allclose(apply_affine(B, np.array([0, 0, 0])), [0, 72.0, 0])
    assert np.allclose(apply_affine(B, np.array([0, 9, 0])), [0, 0.0, 0])


def test_plan_levels_anisotropy_aware():
    fac = plan_levels((64, 2048, 2048), (300.0, 100.0, 100.0), max_levels=6, min_size=32)
    assert fac[0] == (1, 1, 1)
    assert fac[1] == (1, 2, 2)  # xy only
    assert fac[2] == (2, 4, 4)  # at 300/200/200 nm z is within 2x of xy -> all axes halve
    # once xy spacing reaches 200 nm, z (300) is within 2x -> halved from level 2 on
    assert next_factors((300.0, 200.0, 200.0)) == (2, 2, 2)
    assert next_factors((300.0, 100.0, 100.0)) == (1, 2, 2)
    assert next_factors((8.0, 8.0, 8.0)) == (2, 2, 2)
    assert next_factors((50.0, 10.0, 10.0)) == (1, 2, 2)


def test_coarsen_matches_numpy_mean():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1000, (1, 6, 8, 10)).astype(np.uint16)
    d = da.from_array(x, chunks=(1, 3, 4, 5))
    got = coarsen(d, (1, 2, 2)).compute()
    exp = x.reshape(1, 6, 4, 2, 5, 2).mean(axis=(3, 5))
    assert np.allclose(got, np.round(exp))
    got3 = coarsen(d, (2, 2, 2)).compute()
    exp3 = x.reshape(1, 3, 2, 4, 2, 5, 2).mean(axis=(2, 4, 6))
    assert np.allclose(got3, np.round(exp3))


def test_pyramid_levels_and_level_affine(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 255, (1, 16, 256, 256)).astype(np.uint8)
    vol = volume_from_array(x, (40.0, 10.0, 10.0), kind="em", source=str(tmp_path / "x"))
    vol = ensure_pyramid(vol, cache_dir=tmp_path / "cache", min_size=16)
    assert vol.n_levels() >= 3
    assert vol.pyramid_factors[1] == (1, 2, 2)
    # level-1 voxel centre (0,0,0) sits at base voxel (0, .5, .5)
    w = vol.voxel_to_world(np.array([0.0, 0.0, 0.0]), level=1)
    assert np.allclose(w, [0.0, 5.0, 5.0])
    assert vol.level_for_voxel_size(20.0) == 1
    assert vol.level_for_voxel_size(9.0) == 0
    levels, fac = build_pyramid(vol.data, vol.voxel_size_nm, min_size=16)
    assert levels[1].shape == (1, 16, 128, 128)
    # eager build writes a cache that is reused
    vol2 = ensure_pyramid(
        vol.with_(pyramid=None, pyramid_factors=None),
        cache_dir=tmp_path / "cache",
        eager=True,
        min_size=16,
    )
    assert any(p.name.endswith(".pyr.zarr") for p in (tmp_path / "cache").iterdir())
    assert np.array_equal(vol2.pyramid[1].compute(), levels[1].compute())
    vol3 = ensure_pyramid(
        vol.with_(pyramid=None, pyramid_factors=None), cache_dir=tmp_path / "cache", min_size=16
    )
    assert np.array_equal(vol3.pyramid[2].compute(), levels[2].compute())


def test_memory_strategy_forced_and_auto():
    x = np.zeros((1, 4, 8, 8), np.uint8)
    vol = volume_from_array(x, (1, 1, 1))
    ram = apply_memory_strategy(vol, force="ram")
    assert ram.memory_strategy == "ram"
    lazy = apply_memory_strategy(vol, force="lazy")
    assert lazy.memory_strategy == "lazy"
    auto = apply_memory_strategy(vol)
    assert auto.memory_strategy == "ram"  # 256 bytes always fits
    assert auto.metadata["memory_decision"]["strategy"] == "ram"


def test_ome_tiff_roundtrip(tmp_path):
    x = np.random.default_rng(2).integers(0, 4000, (2, 5, 16, 20)).astype(np.uint16)
    p = tmp_path / "lm.ome.tif"
    chans = [Channel("DAPI", 0, "0000FF", emission_nm=450.0), Channel("GFP", 1, "00FF00")]
    write_ome_tiff(p, da.from_array(x, chunks=(1, 2, 16, 20)), (300.0, 100.0, 100.0), chans)
    vol = open_volume(p, kind="lm", memory="skip", pyramid="none")
    assert vol.data.shape == (2, 5, 16, 20)
    assert np.allclose(vol.voxel_size_nm, (300.0, 100.0, 100.0))
    assert [c.name for c in vol.channels] == ["DAPI", "GFP"]
    assert np.array_equal(vol.data.compute(), x)
    assert vol.metadata["voxel_size_source"] == "metadata"


def test_imagej_tiff_voxel_size(tmp_path):
    x = np.zeros((4, 8, 8), np.uint16)
    p = tmp_path / "ij.tif"
    tifffile.imwrite(
        p,
        x,
        imagej=True,
        resolution=(1 / 0.1, 1 / 0.1),
        metadata={"spacing": 0.3, "unit": "um", "axes": "ZYX"},
    )
    vol = open_volume(p, kind="lm", memory="skip", pyramid="none")
    assert np.allclose(vol.voxel_size_nm, (300.0, 100.0, 100.0), atol=1e-6)


def test_tiff_dir_natural_sort(tmp_path):
    d = tmp_path / "stack"
    d.mkdir()
    for i in [10, 2, 1, 3]:
        tifffile.imwrite(d / f"slice_{i}.tif", np.full((6, 7), i, np.uint8))
    vol = open_volume(d, kind="em", voxel_size_nm=(8, 8, 8), memory="skip", pyramid="none")
    assert vol.data.shape == (1, 4, 6, 7)
    assert list(vol.data[0, :, 0, 0].compute()) == [1, 2, 3, 10]
    assert vol.metadata["voxel_size_source"] == "argument"


def test_mrc_roundtrip_and_header_check(tmp_path):
    x = np.random.default_rng(3).integers(0, 200, (6, 12, 10)).astype(np.uint8)
    p = tmp_path / "em.mrc"
    write_mrc(p, da.from_array(x, chunks=(2, 12, 10)), (8.0, 8.0, 8.0))
    vol = open_volume(p, kind="em", memory="skip", pyramid="none")
    assert vol.data.shape == (1, 6, 12, 10)
    assert np.allclose(vol.voxel_size_nm, (8.0, 8.0, 8.0))
    assert np.array_equal(vol.data[0].compute(), x)
    # a header with voxel size 1.0 is flagged and not trusted
    import mrcfile

    p2 = tmp_path / "bad.mrc"
    with mrcfile.new(p2, overwrite=True) as m:
        m.set_data(np.zeros((3, 4, 4), np.int8))
        m.voxel_size = (10.0, 10.0, 10.0)  # 1 nm placeholder
    vol2 = open_volume(p2, kind="em", memory="skip", pyramid="none")
    assert vol2.metadata["voxel_size_source"] == "default"
    assert any("placeholder" in w for w in vol2.metadata["warnings"])


def test_ome_zarr_roundtrip_with_pyramid(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.integers(0, 255, (1, 8, 64, 64)).astype(np.uint8)
    vol = volume_from_array(x, (16.0, 8.0, 8.0), kind="em")
    levels, fac = build_pyramid(vol.data, vol.voxel_size_nm, min_size=4, max_levels=3)
    vs = [tuple(v * f for v, f in zip(vol.voxel_size_nm, fc)) for fc in fac]
    p = tmp_path / "em.zarr"
    write_ome_zarr(p, levels, vs, vol.channels, name="em", translation_nm=(0, 0, 0))
    back = open_volume(p, kind="em", memory="skip", pyramid="lazy")
    assert back.data.shape == (1, 8, 64, 64)
    assert back.n_levels() == 3
    assert np.allclose(back.voxel_size_nm, (16.0, 8.0, 8.0))
    assert back.pyramid_factors[1] == (1, 2, 2)
    assert np.array_equal(back.data.compute(), x)
    assert np.array_equal(back.pyramid[1].compute(), levels[1].compute())


def test_metadata_helpers():
    assert to_nm(0.1, "µm") == pytest.approx(100.0)
    assert to_nm(1.0, "Å") == pytest.approx(0.1)
    assert to_nm(1.0, "furlong") is None
    fwhm = psf_fwhm_from_optics(1.4, 520.0, "oil")
    assert fwhm is not None and 150 < fwhm[1] < 250 and 400 < fwhm[0] < 900
    xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="4" SizeY="4" SizeZ="2" SizeC="1" SizeT="1" PhysicalSizeX="0.1" PhysicalSizeXUnit="µm" PhysicalSizeY="0.1" PhysicalSizeYUnit="µm" PhysicalSizeZ="0.3" PhysicalSizeZUnit="µm"><Channel ID="Channel:0:0" Name="DAPI" EmissionWavelength="450" EmissionWavelengthUnit="nm"/></Pixels></Image></OME>"""
    d = parse_ome_physical_sizes(xml)
    assert d["dz_nm"] == pytest.approx(300.0) and d["dx_nm"] == pytest.approx(100.0)
    assert d["channels"][0]["name"] == "DAPI" and d["channels"][0]["emission_nm"] == 450.0
