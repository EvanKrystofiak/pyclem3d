"""Importers: one entry point (``open_volume``) that returns a normalized :class:`Volume`.

The core never sees a file format. Supported without conversion (plan §1, §5):

* TIFF: single multi-page / BigTIFF / OME-TIFF / ImageJ hyperstack (lazy via tifffile's zarr store)
* a directory of single-slice TIFFs (lazy dask stack)
* MRC / REC / ST / ALI (memory-mapped via mrcfile; header voxel size is *checked*, not trusted)
* OME-Zarr (multiscales are reused as the pyramid)
* optional, via ``bioio`` plugins: .lif (Stellaris), .czi (Zeiss 980), .nd2
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np

from .memory import apply_memory_strategy
from .metadata import (
    parse_ome_physical_sizes,
    psf_fwhm_from_optics,
    suspicious_voxel_size,
    to_nm,
)
from .pyramid import ensure_pyramid
from .volume import DEFAULT_COLORS, Channel, Volume, default_chunks, make_world_affine

log = logging.getLogger(__name__)

TIFF_SUFFIXES = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".btf"}
MRC_SUFFIXES = {".mrc", ".mrcs", ".rec", ".st", ".ali", ".map"}
BIOIO_SUFFIXES = {".lif", ".czi", ".nd2"}


@dataclass
class RawRead:
    """What a format reader returns before normalization."""

    data: da.Array  # (C, Z, Y, X)
    voxel_size_nm: tuple[float, float, float] | None = None
    channel_names: list[str] | None = None
    channel_colors: list[str | None] | None = None
    emission_nm: list[float | None] | None = None
    pyramid: list[da.Array] | None = None
    pyramid_factors: list[tuple[int, int, int]] | None = None
    origin_nm: tuple[float, float, float] | None = None  # world position of voxel (0,0,0)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    handles: list[Any] = field(default_factory=list)  # keep files open for lazy reads


# --------------------------------------------------------------------------- axes
def normalize_axes(arr: da.Array, axes: str, warnings: list[str] | None = None) -> da.Array:
    """Reorder/expand any tifffile-style axes string to (C, Z, Y, X)."""
    axes = axes.upper()
    if arr.ndim != len(axes):
        raise ValueError(f"axes {axes!r} do not match array ndim {arr.ndim}")
    # ImageJ 'I' (frames) and generic 'Q' behave like Z when there is no Z
    if "Z" not in axes:
        for cand in ("I", "Q"):
            if cand in axes:
                axes = axes.replace(cand, "Z", 1)
                break
    if "C" not in axes and "S" in axes:
        axes = axes.replace("S", "C", 1)
    # drop time and any other leading axes by taking index 0
    while True:
        extra = [i for i, a in enumerate(axes) if a not in "CZYX"]
        if not extra:
            break
        i = extra[0]
        if arr.shape[i] > 1 and warnings is not None:
            warnings.append(f"axis {axes[i]!r} has size {arr.shape[i]}; taking index 0")
        arr = arr[(slice(None),) * i + (0,)]
        axes = axes[:i] + axes[i + 1 :]
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"cannot find Y/X in axes {axes!r}")
    if "C" not in axes:
        arr = arr[None]
        axes = "C" + axes
    if "Z" not in axes:
        arr = arr[:, None] if axes.startswith("C") else arr[None]
        axes = axes[0] + "Z" + axes[1:] if axes.startswith("C") else "Z" + axes
    order = [axes.index(a) for a in "CZYX"]
    if order != list(range(4)):
        arr = da.transpose(arr, order)
    return arr


_NAT = re.compile(r"(\d+)")


def natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in _NAT.split(os.fspath(s))]


# ------------------------------------------------------------------------- tiff
def _tiff_voxel_size(tf, series, raw: RawRead) -> tuple[float, float, float] | None:
    dz = dy = dx = None
    if tf.is_ome and tf.ome_metadata:
        ome = parse_ome_physical_sizes(tf.ome_metadata)
        raw.metadata["ome"] = {k: v for k, v in ome.items() if k != "channels"}
        dx, dy, dz = ome.get("dx_nm"), ome.get("dy_nm"), ome.get("dz_nm")
        if ome.get("channels"):
            raw.channel_names = [c["name"] or f"ch{i}" for i, c in enumerate(ome["channels"])]
            raw.channel_colors = [c["color"] for c in ome["channels"]]
            raw.emission_nm = [c["emission_nm"] for c in ome["channels"]]
        if ome.get("na"):
            raw.metadata["na"] = ome["na"]
            raw.metadata["immersion"] = ome.get("immersion")
    page = series.pages[0] if hasattr(series, "pages") else tf.pages[0]
    if page is None:
        page = tf.pages[0]
    if dx is None:
        unit = None
        ij = tf.imagej_metadata or {}
        if ij:
            unit = ij.get("unit")
            if ij.get("spacing") is not None:
                dz = to_nm(float(ij["spacing"]), unit or "µm")
        try:
            xres = page.tags["XResolution"].value
            yres = page.tags["YResolution"].value
            ru = page.tags.get("ResolutionUnit")
            ru_val = ru.value if ru is not None else 2
            ru_val = int(ru_val) if not hasattr(ru_val, "value") else int(ru_val.value)
            xr = xres[0] / xres[1] if xres[1] else 0
            yr = yres[0] / yres[1] if yres[1] else 0
            if unit is None:
                unit = {2: "inch", 3: "cm"}.get(ru_val)
            if xr > 0 and unit:
                dx = to_nm(1.0 / xr, unit)
                dy = to_nm(1.0 / yr, unit) if yr > 0 else dx
        except (KeyError, AttributeError, TypeError, ZeroDivisionError):
            pass
    if dx is None or dy is None:
        return None
    if dz is None:
        raw.warnings.append("no z spacing in TIFF metadata; assuming dz = dx")
        dz = dx
    return (float(dz), float(dy), float(dx))


def _recommend_conversion(path: Path, tf, series) -> list[str]:
    reasons = []
    try:
        page = series.pages[0] if hasattr(series, "pages") else tf.pages[0]
        comp = getattr(page, "compression", None)
        comp_val = getattr(comp, "value", comp)
        if comp_val not in (None, 1):
            reasons.append(f"compressed TIFF ({comp}) gives poor random access")
        if not page.is_tiled and page.rowsperstrip and page.rowsperstrip < page.imagelength // 4:
            reasons.append("striped TIFF with small strips; random z/xy access is slow")
    except Exception:  # pragma: no cover
        pass
    if _is_network_path(path):
        reasons.append("data lives on a network share")
    return reasons


def _is_network_path(path: Path) -> bool:
    s = str(path)
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    try:
        import psutil

        drive = os.path.splitdrive(os.path.abspath(s))[0]
        for part in psutil.disk_partitions(all=True):
            if part.device.rstrip("\\").lower() == drive.lower() and "remote" in part.opts:
                return True
    except Exception:  # pragma: no cover
        pass
    return False


def _parse_imagej_info(info: str, raw: RawRead) -> None:
    """Channel names, emission wavelengths, NA and immersion from the Bio-Formats 'Info' block
    that Fiji stores in ImageJ TIFFs (Zeiss .czi exports carry them as key = value lines)."""
    if not info:
        return
    kv: dict[str, str] = {}
    for line in info.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    names: dict[int, str] = {}
    ems: dict[int, float] = {}
    for k, v in kv.items():
        if k.startswith("Information|Image|Channel|EmissionWavelength #"):
            try:
                ems[int(k.rsplit("#", 1)[1])] = float(v)
            except ValueError:
                pass
        elif k.startswith("Information|Image|Channel|Name #") or k.startswith(
            "Information|Image|Channel|Fluor #"
        ):
            names.setdefault(int(k.rsplit("#", 1)[1]), v)
    if not names:  # fall back to the track channel names, skipping the '#' duplicates
        for k, v in kv.items():
            if (
                k.startswith("Experiment|AcquisitionBlock|MultiTrackSetup|Track|Channel|Name #")
                and "#" not in v
            ):
                names[len(names) + 1] = v
    n = int(kv.get("SizeC", raw.data.shape[0] if raw.data is not None else 0) or 0)
    if names and n and len(names) >= n:
        raw.channel_names = [names.get(i + 1, f"ch{i}") for i in range(n)]
    if ems and n:
        raw.emission_nm = [ems.get(i + 1) for i in range(n)]
    na = kv.get("Information|Instrument|Objective|LensNA")
    if na:
        try:
            raw.metadata["na"] = float(na)
        except ValueError:
            pass
    ri = kv.get("Information|Instrument|Objective|ImmersionRefractiveIndex")
    imm = kv.get("Information|Instrument|Objective|Immersion")
    if ri:
        try:
            raw.metadata["immersion"] = float(ri)
        except ValueError:
            raw.metadata["immersion"] = imm
    elif imm:
        raw.metadata["immersion"] = imm
    obj = kv.get("Information|Instrument|Objective|Name")
    if obj:
        raw.metadata["objective"] = obj
    if "Airyscan" in kv.get("Series 0 Name", "") or any("AiryScan" in k for k in kv):
        raw.metadata["airyscan"] = True


def read_tiff(path: Path, series_index: int = 0) -> RawRead:
    import tifffile
    import zarr

    tf = tifffile.TiffFile(str(path))
    series = tf.series[series_index]
    store = series.aszarr()
    z = zarr.open(store, mode="r")
    raw = RawRead(data=None, handles=[tf, store])  # type: ignore[arg-type]
    levels = None
    if isinstance(z, zarr.Group):  # pyramidal TIFF (SubIFDs)
        keys = sorted(z.array_keys(), key=natural_key)
        levels = [da.from_zarr(z[k]) for k in keys]
        arr = levels[0]
    else:
        arr = da.from_zarr(z)
    axes = series.axes
    arr = normalize_axes(arr, axes, raw.warnings)
    if levels is not None and len(levels) > 1:
        lv = [normalize_axes(lvl, axes, []) for lvl in levels]
        base = np.array(lv[0].shape[1:], dtype=float)
        raw.pyramid = lv
        raw.pyramid_factors = [
            tuple(int(round(b / s)) for b, s in zip(base, l.shape[1:])) for l in lv
        ]
    raw.data = arr
    raw.voxel_size_nm = _tiff_voxel_size(tf, series, raw)
    if tf.imagej_metadata and tf.imagej_metadata.get("Info"):
        _parse_imagej_info(str(tf.imagej_metadata["Info"]), raw)
    raw.metadata["format"] = "tiff"
    raw.metadata["axes_in_file"] = axes
    rec = _recommend_conversion(path, tf, series)
    if rec:
        raw.metadata["recommend_zarr"] = rec
    return raw


def read_tiff_dir(path: Path, pattern: str = "*.tif*") -> RawRead:
    import tifffile

    files = sorted(
        [p for p in path.glob(pattern) if p.is_file()], key=lambda p: natural_key(p.name)
    )
    if not files:
        raise FileNotFoundError(f"no {pattern} files in {path}")
    first = tifffile.imread(str(files[0]))
    if first.ndim != 2:
        raise ValueError(f"expected 2D slices in {path}; first file has shape {first.shape}")
    shape, dtype = first.shape, first.dtype
    read = dask.delayed(tifffile.imread)
    planes = [da.from_delayed(read(str(f)), shape=shape, dtype=dtype) for f in files]
    stack = da.stack(planes, axis=0)[None]
    raw = RawRead(data=stack)
    raw.metadata["format"] = "tiff-dir"
    raw.metadata["n_files"] = len(files)
    with tifffile.TiffFile(str(files[0])) as tf:
        vs = _tiff_voxel_size(tf, tf.series[0], raw)
    raw.voxel_size_nm = vs
    if _is_network_path(path):
        raw.metadata["recommend_zarr"] = ["data lives on a network share"]
    return raw


# -------------------------------------------------------------------------- mrc
def read_mrc(path: Path) -> RawRead:
    import mrcfile

    m = mrcfile.mmap(str(path), mode="r", permissive=True)
    data = m.data
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"MRC {path} has shape {data.shape}; expected a 3D stack")
    chunks = default_chunks((1, *data.shape), data.dtype.itemsize)[1:]
    arr = da.from_array(data, chunks=chunks, name=f"mrc-{path.name}-{id(m)}")[None]
    raw = RawRead(data=arr, handles=[m])
    vs = m.voxel_size
    vs_nm = (float(vs.z) * 0.1, float(vs.y) * 0.1, float(vs.x) * 0.1)
    raw.metadata["format"] = "mrc"
    raw.metadata["header_voxel_size_nm"] = list(vs_nm)
    warn = suspicious_voxel_size(vs_nm)
    if warn:
        raw.warnings.append(f"MRC header voxel size {vs_nm} nm: {warn}; pass --voxel-size")
        raw.voxel_size_nm = None
    else:
        raw.voxel_size_nm = vs_nm
    if _is_network_path(path):
        raw.metadata["recommend_zarr"] = ["data lives on a network share"]
    return raw


# ------------------------------------------------------------------------- zarr
def read_zarr(path: Path) -> RawRead:
    import zarr

    node = zarr.open(str(path), mode="r")
    raw = RawRead(data=None)  # type: ignore[arg-type]
    raw.metadata["format"] = "zarr"
    if isinstance(node, zarr.Array):
        arr = da.from_zarr(node)
        axes = {3: "ZYX", 4: "CZYX", 2: "YX"}.get(arr.ndim)
        if axes is None:
            raise ValueError(f"cannot interpret zarr array of ndim {arr.ndim}")
        raw.data = normalize_axes(arr, axes, raw.warnings)
        return raw
    attrs = dict(node.attrs)
    ms = attrs.get("multiscales")
    if not ms:
        keys = sorted(node.array_keys(), key=natural_key)
        if not keys:
            raise ValueError(f"zarr group {path} has no arrays")
        arr = da.from_zarr(node[keys[0]])
        raw.data = normalize_axes(arr, {3: "ZYX", 4: "CZYX"}[arr.ndim], raw.warnings)
        return raw
    ms0 = ms[0]
    axes_meta = ms0.get("axes", [])
    if axes_meta and isinstance(axes_meta[0], dict):
        axes = "".join(a["name"] for a in axes_meta).upper()
        units = {a["name"]: a.get("unit") for a in axes_meta}
    else:
        axes = "".join(axes_meta).upper() if axes_meta else None
        units = {}
    levels = []
    scales = []
    for ds in ms0["datasets"]:
        arr = da.from_zarr(node[ds["path"]])
        if axes is None:
            axes = {3: "ZYX", 4: "CZYX", 5: "TCZYX"}[arr.ndim]
        sc = None
        tr = None
        for t in ds.get("coordinateTransformations", []):
            if t.get("type") == "scale":
                sc = t["scale"]
            elif t.get("type") == "translation":
                tr = t["translation"]
        scales.append(sc)
        if not levels and tr is not None:
            raw.metadata["_translation0"] = tr
        levels.append(normalize_axes(arr, axes, raw.warnings if not levels else []))
    raw.data = levels[0]
    if len(levels) > 1:
        raw.pyramid = levels
        base = np.array(levels[0].shape[1:], dtype=float)
        raw.pyramid_factors = [
            tuple(int(round(b / s)) for b, s in zip(base, l.shape[1:])) for l in levels
        ]
    if scales[0] is not None and axes is not None:
        idx = {a: i for i, a in enumerate(axes)}
        vs = []
        for a in "ZYX":
            i = idx.get(a)
            unit = units.get(a.lower(), "micrometer") if units else "micrometer"
            v = to_nm(float(scales[0][i]), unit or "micrometer") if i is not None else None
            vs.append(v)
        if all(v is not None for v in vs):
            raw.voxel_size_nm = (vs[0], vs[1], vs[2])  # type: ignore[assignment]
        tr0 = raw.metadata.pop("_translation0", None)
        if tr0 is not None:
            org = []
            for a in "ZYX":
                i = idx.get(a)
                unit = units.get(a.lower(), "micrometer") if units else "micrometer"
                org.append(to_nm(float(tr0[i]), unit or "micrometer") if i is not None else 0.0)
            if all(o is not None for o in org):
                raw.origin_nm = (org[0], org[1], org[2])  # type: ignore[assignment]
    omero = attrs.get("omero", {})
    if omero.get("channels"):
        raw.channel_names = [c.get("label") or f"ch{i}" for i, c in enumerate(omero["channels"])]
        raw.channel_colors = [c.get("color") for c in omero["channels"]]
    raw.metadata["ome_zarr_attrs"] = {
        k: v for k, v in attrs.items() if k in ("multiscales", "omero")
    }
    return raw


# ------------------------------------------------------------------------ bioio
def read_bioio(path: Path, scene: int | None = None) -> RawRead:
    try:
        from bioio import BioImage
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError(
            f"reading {path.suffix} needs the optional readers: pip install 'pyclem3d[readers]'"
        ) from e
    img = BioImage(str(path))
    if scene is not None:
        img.set_scene(scene)
    arr = img.dask_data  # TCZYX
    raw = RawRead(data=None, handles=[img])  # type: ignore[arg-type]
    raw.data = normalize_axes(arr, "TCZYX", raw.warnings)
    pps = img.physical_pixel_sizes
    if pps.X:
        dz = pps.Z if pps.Z else pps.X
        raw.voxel_size_nm = (float(dz) * 1e3, float(pps.Y) * 1e3, float(pps.X) * 1e3)
    raw.channel_names = list(img.channel_names)
    raw.metadata["format"] = f"bioio:{path.suffix}"
    raw.metadata["scenes"] = list(img.scenes)
    return raw


# ------------------------------------------------------------------------ entry
def read_raw(path: str | os.PathLike, **kw: Any) -> RawRead:
    p = Path(path)
    if p.is_dir():
        if p.suffix == ".zarr" or (p / ".zattrs").exists() or (p / "zarr.json").exists():
            return read_zarr(p)
        return read_tiff_dir(p, kw.get("pattern", "*.tif*"))
    suf = p.suffix.lower()
    if suf in TIFF_SUFFIXES or p.name.lower().endswith((".ome.tif", ".ome.tiff")):
        return read_tiff(p, kw.get("series", 0))
    if suf in MRC_SUFFIXES:
        return read_mrc(p)
    if suf in BIOIO_SUFFIXES:
        return read_bioio(p, kw.get("scene"))
    if suf in (".zarr",):
        return read_zarr(p)
    raise ValueError(f"unsupported file type: {p}")


def open_volume(
    path: str | os.PathLike,
    kind: str = "em",
    voxel_size_nm: tuple[float, float, float] | None = None,
    y_scale: float = 1.0,
    origin_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    channel_names: list[str] | None = None,
    psf_nm: tuple[float, float] | None = None,
    memory: str | None = None,
    memory_fraction: float = 0.3,
    pyramid: str = "lazy",
    cache_dir: str | os.PathLike | None = None,
    **reader_kw: Any,
) -> Volume:
    """Open any supported file as a normalized, lazily-accessed :class:`Volume`.

    ``memory``: None (auto), "ram", "lazy", or "skip" (leave undecided).
    ``pyramid``: "lazy" (compute levels on demand), "eager" (build into cache), "none".
    """
    raw = read_raw(path, **reader_kw)
    for w in raw.warnings:
        log.warning("%s: %s", path, w)
    vs = voxel_size_nm or raw.voxel_size_nm
    vs_source = "argument" if voxel_size_nm else "metadata"
    if vs is None:
        vs = (1.0, 1.0, 1.0)
        vs_source = "default"
        log.warning("%s: no voxel size found; using 1 nm. Pass voxel_size_nm.", path)
    vs = (float(vs[0]), float(vs[1]), float(vs[2]))
    shape = tuple(int(s) for s in raw.data.shape[1:])
    if raw.origin_nm is not None and tuple(origin_nm) == (0.0, 0.0, 0.0):
        origin_nm = raw.origin_nm  # OME-Zarr translation metadata
    affine = make_world_affine(vs, origin_nm=origin_nm, y_scale=y_scale, shape_zyx=shape)  # type: ignore[arg-type]
    C = int(raw.data.shape[0])
    names = channel_names or raw.channel_names or [f"ch{i}" for i in range(C)]
    names = (list(names) + [f"ch{i}" for i in range(len(names), C)])[:C]
    colors = raw.channel_colors or [None] * C
    em = raw.emission_nm or [None] * C
    channels = []
    for i in range(C):
        col = colors[i] if i < len(colors) and colors[i] else None
        if col is None:
            col = "FFFFFF" if kind == "em" else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        channels.append(
            Channel(name=names[i], index=i, color=col, emission_nm=em[i] if i < len(em) else None)
        )
    meta = dict(raw.metadata)
    meta["voxel_size_source"] = vs_source
    meta["y_scale"] = float(y_scale)
    meta["warnings"] = list(raw.warnings)
    meta["_handles"] = raw.handles
    psf = psf_nm
    if psf is None and kind == "lm":
        em0 = next((e for e in em if e), None)
        psf = psf_fwhm_from_optics(meta.get("na"), em0, meta.get("immersion"))
        meta["psf_source"] = "metadata" if psf else "none"
    else:
        meta["psf_source"] = "argument" if psf else "none"
    vol = Volume(
        data=raw.data,
        voxel_size_nm=vs,
        world_affine=affine,
        kind=kind,  # type: ignore[arg-type]
        channels=channels,
        source=os.fspath(path),
        metadata=meta,
        psf_nm=psf,
        pyramid=raw.pyramid,
        pyramid_factors=raw.pyramid_factors,
    )
    if memory != "skip":
        vol = apply_memory_strategy(vol, fraction=memory_fraction, force=memory)
    if pyramid != "none":
        vol = ensure_pyramid(vol, cache_dir=cache_dir, eager=(pyramid == "eager"))
    log.info(vol.describe())
    return vol


def volume_from_array(
    data: np.ndarray | da.Array,
    voxel_size_nm: tuple[float, float, float],
    kind: str = "em",
    source: str = "",
    chunks: tuple[int, ...] | None = None,
    **kw: Any,
) -> Volume:
    """Wrap an in-memory or dask array as a Volume (used by tests and the phantom)."""
    if not isinstance(data, da.Array):
        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = arr[None]
        data = da.from_array(arr, chunks=chunks or default_chunks(arr.shape, arr.dtype.itemsize))
    elif data.ndim == 3:
        data = data[None]
    shape = tuple(int(s) for s in data.shape[1:])
    affine = make_world_affine(
        voxel_size_nm,
        shape_zyx=shape,
        **{k: v for k, v in kw.items() if k in ("origin_nm", "y_scale", "flips")},
    )  # type: ignore[arg-type]
    rest = {k: v for k, v in kw.items() if k not in ("origin_nm", "y_scale", "flips")}
    return Volume(
        data=data,
        voxel_size_nm=voxel_size_nm,
        world_affine=affine,
        kind=kind,
        source=source,
        **rest,
    )  # type: ignore[arg-type]
