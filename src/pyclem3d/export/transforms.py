"""Output 2 (plan §9): transform-only exports - no resampling, no data loss.

* JSON (4x4 in both zyx and xyz conventions, nm)
* ITK ``.tfm`` (AffineTransform_double_3_3, xyz), in both directions
* BigWarp landmark CSV (moving = LM, fixed = EM, xyz, physical units) + import
* BigDataViewer XML + HDF5 where the LM setup carries the LM->EM affine
* NRRD displacement field (EM world -> LM world) for a TPS, for ITK/elastix tools
"""

from __future__ import annotations

import csv
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

from ..io.volume import Volume
from ..register.landmarks import LandmarkSet
from ..register.transforms import AffineTransform, TPSTransform, Transform
from ..resample.grid import DisplacementGrid

log = logging.getLogger(__name__)

_P3 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
_P4 = np.eye(4)
_P4[:3, :3] = _P3

UNIT_SCALE = {"nm": 1.0, "um": 1e-3, "µm": 1e-3, "mm": 1e-6}


def zyx_to_xyz(M: np.ndarray) -> np.ndarray:
    """Re-express a 4x4 acting on (z,y,x,1) as one acting on (x,y,z,1)."""
    return _P4 @ np.asarray(M, dtype=float) @ _P4


def scale_units(M: np.ndarray, unit: str) -> np.ndarray:
    """Convert the translation of a world->world 4x4 from nm to ``unit`` (linear part is unit-free)."""
    out = np.asarray(M, dtype=float).copy()
    out[:3, 3] *= UNIT_SCALE[unit]
    return out


# ------------------------------------------------------------------------- JSON
def export_transform_json(
    t: Transform, path: str | os.PathLike, meta: dict[str, Any] | None = None
) -> Path:
    path = Path(path)
    d: dict[str, Any] = {
        "tool": "pyclem3d",
        "units": "nm",
        "maps": "LM world -> EM world (moving -> fixed)",
        "transform": t.to_dict(),
    }
    if isinstance(t, AffineTransform):
        d["lm_to_em_zyx"] = t.matrix.tolist()
        d["lm_to_em_xyz"] = zyx_to_xyz(t.matrix).tolist()
        d["em_to_lm_zyx"] = t.inverse().matrix.tolist()
        d["em_to_lm_xyz"] = zyx_to_xyz(t.inverse().matrix).tolist()
        d["decomposition"] = t.decompose()
    elif isinstance(t, TPSTransform):
        d["affine_part_zyx"] = t.affine_part().matrix.tolist()
        d["affine_part_xyz"] = zyx_to_xyz(t.affine_part().matrix).tolist()
    if meta:
        d["meta"] = meta
    import json

    path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return path


# -------------------------------------------------------------------------- ITK
def _tfm_text(M_xyz: np.ndarray, comment: str) -> str:
    L = M_xyz[:3, :3]
    t = M_xyz[:3, 3]
    params = " ".join(f"{v:.10g}" for v in np.concatenate([L.ravel(), t]))
    return (
        "#Insight Transform File V1.0\n"
        f"# {comment}\n"
        "#Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        f"Parameters: {params}\n"
        "FixedParameters: 0 0 0\n"
    )


def export_itk_tfm(
    t: Transform, out_dir: str | os.PathLike, unit: str = "um", stem: str = "pyclem3d"
) -> dict[str, Path]:
    """Write ``<stem>_lm_to_em.tfm`` (point mapping LM->EM) and ``<stem>_em_to_lm_resample.tfm``
    (what ITK/SimpleITK wants to resample the LM onto the EM grid: fixed -> moving).

    Coordinates are xyz in ``unit``. ITK's LPS convention is ignored (microscopy data).
    For a TPS only the affine part is written; use the NRRD displacement field for the rest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A = t if isinstance(t, AffineTransform) else t.affine_part()  # type: ignore[union-attr]
    fwd = scale_units(zyx_to_xyz(A.matrix), unit)
    inv = scale_units(zyx_to_xyz(A.inverse().matrix), unit)
    p1 = out_dir / f"{stem}_lm_to_em.tfm"
    p2 = out_dir / f"{stem}_em_to_lm_resample.tfm"
    p1.write_text(
        _tfm_text(
            fwd, f"pyclem3d: maps LM points (xyz, {unit}) to EM points; affine part only for TPS"
        ),
        encoding="utf-8",
    )
    p2.write_text(
        _tfm_text(
            inv, f"pyclem3d: EM (fixed) -> LM (moving), xyz {unit}; use as the resampling transform"
        ),
        encoding="utf-8",
    )
    return {"lm_to_em": p1, "em_to_lm_resample": p2}


def read_itk_tfm(path: str | os.PathLike, unit: str = "um") -> np.ndarray:
    """Read an AffineTransform_double_3_3 .tfm back to a zyx 4x4 in nm (for tests / imports)."""
    params = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Parameters:"):
            params = [float(v) for v in line.split(":", 1)[1].split()]
    if params is None or len(params) != 12:
        raise ValueError("not an AffineTransform_double_3_3 file")
    M = np.eye(4)
    M[:3, :3] = np.asarray(params[:9]).reshape(3, 3)
    M[:3, 3] = np.asarray(params[9:]) / UNIT_SCALE[unit]
    return zyx_to_xyz(M)


# ---------------------------------------------------------------------- BigWarp
def export_bigwarp_csv(landmarks: LandmarkSet, path: str | os.PathLike, unit: str = "um") -> Path:
    """BigWarp landmark table: name, active, moving xyz (LM), fixed xyz (EM), in ``unit``."""
    path = Path(path)
    s = UNIT_SCALE[unit]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        for lm in landmarks.landmarks:
            mz, my, mx = (v * s for v in lm.lm_world_nm)
            fz, fy, fx = (v * s for v in lm.em_world_nm)
            w.writerow(
                [
                    f"Pt-{lm.id}",
                    "true" if lm.enabled else "false",
                    f"{mx:.6f}",
                    f"{my:.6f}",
                    f"{mz:.6f}",
                    f"{fx:.6f}",
                    f"{fy:.6f}",
                    f"{fz:.6f}",
                ]
            )
    return path


def import_bigwarp_csv(
    path: str | os.PathLike,
    unit: str = "um",
    sigma_nm: tuple[float, float, float] = (300.0, 50.0, 50.0),
) -> LandmarkSet:
    ls = LandmarkSet()
    s = UNIT_SCALE[unit]
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 8 or not row[2].strip():
                continue
            name, active = row[0], row[1].strip().lower() == "true"
            mx, my, mz, fx, fy, fz = (float(v) / s for v in row[2:8])
            lm = ls.add((fz, fy, fx), (mz, my, mx), sigma_nm, feature=name, enabled=active)
            lm.method = {"em": "bigwarp", "lm": "bigwarp"}
    return ls


# --------------------------------------------------------------------------- BDV
def _to_uint16(block: np.ndarray, src_dtype: np.dtype) -> np.ndarray:
    if src_dtype == np.uint16:
        return block.astype(np.uint16)
    if src_dtype == np.uint8:
        return block.astype(np.uint16)
    if src_dtype == np.int8:
        return (block.astype(np.int16) + 128).astype(np.uint16)
    if src_dtype == np.int16:
        return (block.astype(np.int32) + 32768).astype(np.uint16)
    b = block.astype(np.float64)
    return np.clip(b, 0, 65535).astype(np.uint16)


def _write_bdv_setup(
    h5, setup: int, levels: list[da.Array], factors: list[tuple[int, int, int]]
) -> None:
    res = np.array([[f[2], f[1], f[0]] for f in factors], dtype=np.float64)  # xyz
    subdiv = []
    g = h5.require_group(f"s{setup:02d}")
    for i, lvl in enumerate(levels):
        Z, Y, X = (int(s) for s in lvl.shape)
        chunk = (min(Z, 16), min(Y, 64), min(X, 64))
        ds = h5.require_group(f"t00000/s{setup:02d}/{i}").create_dataset(
            "cells",
            shape=(Z, Y, X),
            chunks=chunk,
            dtype=np.int16,
            compression="gzip",
            compression_opts=1,
        )
        z0 = 0
        for zc in lvl.chunks[0]:
            block = np.asarray(lvl[z0 : z0 + zc].compute())
            ds[z0 : z0 + zc] = _to_uint16(block, lvl.dtype).view(np.int16)
            z0 += zc
        subdiv.append([chunk[2], chunk[1], chunk[0]])
    g.create_dataset("resolutions", data=res)
    g.create_dataset("subdivisions", data=np.array(subdiv, dtype=np.int32))


def export_bdv_xml_h5(
    em: Volume,
    lm: Volume,
    transform: Transform,
    out_xml: str | os.PathLike,
    em_level: int | None = None,
    lm_channels: list[int] | None = None,
    unit: str = "um",
) -> Path:
    """BigDataViewer XML + HDF5: EM (from ``em_level`` down) and every LM channel as setups; the LM
    setups carry the LM->EM affine so BDV displays the *unresampled* LM registered to the EM.

    Needs ``h5py`` (``pip install 'pyclem3d[export]'``). Data are stored as 16-bit; float
    volumes are clipped to [0, 65535]. Affines are xyz in ``unit``.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError("BDV export needs h5py: pip install 'pyclem3d[export]'") from e
    out_xml = Path(out_xml)
    out_h5 = out_xml.with_suffix(".h5")
    if em_level is None:
        em_level = em.level_for_voxel_size(max(lm.voxel_size_nm[1:]))
    A = transform if isinstance(transform, AffineTransform) else transform.affine_part()  # type: ignore[union-attr]
    setups: list[dict[str, Any]] = []
    with h5py.File(out_h5, "w") as h5:
        # EM: levels em_level .. coarsest
        em_levels = [em.level_data(l)[0] for l in range(em_level, em.n_levels())]
        base_f = np.asarray(em.pyramid_factors[em_level] if em.pyramid_factors else (1, 1, 1))
        factors = (
            [
                tuple(int(v) for v in (np.asarray(em.pyramid_factors[l]) // base_f))
                for l in range(em_level, em.n_levels())
            ]
            if em.pyramid_factors
            else [(1, 1, 1)]
        )
        _write_bdv_setup(h5, 0, em_levels, factors)  # type: ignore[arg-type]
        Z, Y, X = em_levels[0].shape
        setups.append(
            {
                "id": 0,
                "name": f"EM {em.channels[0].name}",
                "size": (X, Y, Z),
                "voxel": em.level_voxel_size_nm(em_level),
                "affine": scale_units(zyx_to_xyz(em.level_affine(em_level)), unit),
                "channel": 0,
            }
        )
        chans = list(range(lm.n_channels)) if lm_channels is None else list(lm_channels)
        for k, c in enumerate(chans, start=1):
            lv = [lm.level_data(l)[c] for l in range(lm.n_levels())]
            fac = lm.pyramid_factors or [(1, 1, 1)]
            _write_bdv_setup(h5, k, lv, fac)  # type: ignore[arg-type]
            Zl, Yl, Xl = lv[0].shape
            M = scale_units(zyx_to_xyz(A.matrix @ lm.world_affine), unit)
            setups.append(
                {
                    "id": k,
                    "name": f"LM {lm.channels[c].name}",
                    "size": (Xl, Yl, Zl),
                    "voxel": lm.voxel_size_nm,
                    "affine": M,
                    "channel": k,
                }
            )
    _write_bdv_xml(out_xml, out_h5, setups, unit)
    log.info("wrote BDV %s + %s (%d setups)", out_xml, out_h5, len(setups))
    return out_xml


def _write_bdv_xml(out_xml: Path, out_h5: Path, setups: list[dict[str, Any]], unit: str) -> None:
    root = ET.Element("SpimData", version="0.2")
    ET.SubElement(root, "BasePath", type="relative").text = "."
    seq = ET.SubElement(root, "SequenceDescription")
    loader = ET.SubElement(seq, "ImageLoader", format="bdv.hdf5")
    ET.SubElement(loader, "hdf5", type="relative").text = out_h5.name
    vs = ET.SubElement(seq, "ViewSetups")
    for st in setups:
        v = ET.SubElement(vs, "ViewSetup")
        ET.SubElement(v, "id").text = str(st["id"])
        ET.SubElement(v, "name").text = st["name"]
        ET.SubElement(v, "size").text = " ".join(str(int(x)) for x in st["size"])
        vx = ET.SubElement(v, "voxelSize")
        ET.SubElement(vx, "unit").text = unit
        dz, dy, dx = st["voxel"]
        sc = UNIT_SCALE[unit]
        ET.SubElement(vx, "size").text = f"{dx * sc:.6g} {dy * sc:.6g} {dz * sc:.6g}"
        attrs = ET.SubElement(v, "attributes")
        ET.SubElement(attrs, "channel").text = str(st["channel"])
    attr = ET.SubElement(vs, "Attributes", name="channel")
    for st in setups:
        ch = ET.SubElement(attr, "Channel")
        ET.SubElement(ch, "id").text = str(st["channel"])
        ET.SubElement(ch, "name").text = str(st["channel"])
    tps = ET.SubElement(seq, "Timepoints", type="pattern")
    ET.SubElement(tps, "integerpattern").text = "0"
    regs = ET.SubElement(root, "ViewRegistrations")
    for st in setups:
        vr = ET.SubElement(regs, "ViewRegistration", timepoint="0", setup=str(st["id"]))
        vt = ET.SubElement(vr, "ViewTransform", type="affine")
        ET.SubElement(vt, "Name").text = "pyclem3d voxel->world (LM carries LM->EM)"
        M = np.asarray(st["affine"])[:3, :4]
        ET.SubElement(vt, "affine").text = " ".join(f"{v:.10g}" for v in M.ravel())
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


# -------------------------------------------------------------------------- NRRD
def export_displacement_nrrd(
    dg: DisplacementGrid, path: str | os.PathLike, unit: str = "um"
) -> Path:
    """Write the coarse EM->LM displacement field (vector, xyz components) as NRRD (ITK/elastix)."""
    path = Path(path)
    s = UNIT_SCALE[unit]
    vals = np.ascontiguousarray(
        dg.values[..., ::-1] * s, dtype=np.float32
    )  # components z,y,x -> x,y,z
    nz, ny, nx, _ = vals.shape
    sp = dg.spacing_nm * s
    org = dg.origin_nm * s
    header = "\n".join(
        [
            "NRRD0004",
            "# pyclem3d displacement field: EM world -> LM world (add to a point to get its LM position)",
            "type: float",
            "dimension: 4",
            "space: 3D-right-handed",
            f"sizes: 3 {nx} {ny} {nz}",
            f"space directions: none ({sp[2]:.10g},0,0) (0,{sp[1]:.10g},0) (0,0,{sp[0]:.10g})",
            "kinds: vector domain domain domain",
            "endian: little",
            "encoding: raw",
            f"space origin: ({org[2]:.10g},{org[1]:.10g},{org[0]:.10g})",
            "",
            "",
        ]
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vals.astype("<f4").tobytes(order="C"))
    return path


def read_nrrd_header(path: str | os.PathLike) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").rstrip("\n")
            if line == "":
                break
            if line.startswith("#") or line.startswith("NRRD"):
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def export_all_transforms(
    t: Transform,
    landmarks: LandmarkSet | None,
    out_dir: str | os.PathLike,
    displacement: DisplacementGrid | None = None,
    unit: str = "um",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {
        "json": str(export_transform_json(t, out_dir / "transform.json", meta))
    }
    written.update({k: str(v) for k, v in export_itk_tfm(t, out_dir, unit).items()})
    if landmarks is not None and len(landmarks):
        written["bigwarp_csv"] = str(
            export_bigwarp_csv(landmarks, out_dir / "landmarks_bigwarp.csv", unit)
        )
    if isinstance(t, TPSTransform) and displacement is not None:
        written["displacement_nrrd"] = str(
            export_displacement_nrrd(displacement, out_dir / "displacement_em_to_lm.nrrd", unit)
        )
    return written
