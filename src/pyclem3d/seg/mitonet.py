"""Organelle segmentation of an EM Volume with empanada (MitoNet / NucleoNet), slice by slice.

Optional dependency (``pyclem3d[seg]``). Runs the 2D panoptic engine on every z slice at a
chosen inference scale and streams a semantic mask (uint8, 0/1) or instance labels (uint32,
2D-consistent only) into a zarr array next to the source, so the 1.5 GB stack never has to
sit in memory twice. The mask is what the segmentation-to-image registration consumes
(:mod:`pyclem3d.seg.synthetic`); it does not need to be instance-perfect.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

from ..io.volume import Volume
from ..io.writers import open_group_v2

log = logging.getLogger(__name__)

MODEL_ALIASES = {"mito": "MitoNet_v1", "mito-mini": "MitoNet_v1_mini", "nucleus": "NucleoNet_base_v2"}


def load_engine(model: str = "MitoNet_v1", inference_scale: int = 2, use_gpu: bool = True, semantic: bool = True, **kw: Any):
    """An empanada 2D engine for a bundled model name (or alias: mito, mito-mini, nucleus)."""
    try:
        import yaml
        from empanada_napari.inference import Engine2d
        from empanada_napari.utils import get_configs
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError("segmentation needs empanada-napari: uv sync --extra seg") from e
    name = MODEL_ALIASES.get(model, model)
    cfgs = get_configs()
    if name not in cfgs:
        raise ValueError(f"unknown empanada model {name!r}; available: {sorted(cfgs)}")
    with open(cfgs[name]) as f:
        cfg = yaml.safe_load(f)
    return Engine2d(cfg, inference_scale=inference_scale, use_gpu=use_gpu, semantic_only=semantic, **kw)


def segment_volume(
    volume: Volume,
    out_path: str | os.PathLike,
    model: str = "MitoNet_v1",
    inference_scale: int = 2,
    semantic: bool = True,
    channel: int = 0,
    z_range: tuple[int, int] | None = None,
    use_gpu: bool = True,
    progress=None,
    engine=None,
) -> Path:
    """Segment every slice of ``volume`` and write ``out_path`` (zarr, v2 layout).

    The output has the EM's shape (Z, Y, X) at level 0 and inherits its voxel size; slices
    outside ``z_range`` are left zero. Returns the zarr path. ~1 s per 1462x1142 slice on an
    RTX 4070 at scale 2.
    """
    out_path = Path(out_path)
    eng = engine or load_engine(model, inference_scale, use_gpu, semantic)
    Z, Y, X = volume.shape_zyx
    z0, z1 = (0, Z) if z_range is None else (max(0, z_range[0]), min(Z, z_range[1]))
    g = open_group_v2(out_path, mode="w")
    dtype = np.uint8 if semantic else np.uint32
    arr = g.create_array("0", shape=(Z, Y, X), chunks=(1, Y, X), dtype=dtype) if hasattr(g, "create_array") else g.create_dataset("0", shape=(Z, Y, X), chunks=(1, Y, X), dtype=dtype)
    g.attrs["pyclem3d_seg"] = {
        "model": MODEL_ALIASES.get(model, model),
        "inference_scale": int(inference_scale),
        "semantic": bool(semantic),
        "source": volume.source,
        "voxel_size_nm": list(volume.voxel_size_nm),
        "world_affine": volume.world_affine.tolist(),
        "z_range": [int(z0), int(z1)],
    }
    data = volume.data[channel]
    t0 = time.time()
    n_fg = 0
    for z in range(z0, z1):
        img = np.asarray(data[z].compute())
        if img.dtype != np.uint8:
            lo, hi = np.percentile(img, [0.5, 99.5])
            img = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        seg = eng.infer(img)
        if semantic:
            plane = (seg > 0).astype(np.uint8)
        else:
            plane = seg.astype(np.uint32)
        arr[z] = plane
        n_fg += int((plane > 0).sum())
        if progress is not None:
            progress(z - z0 + 1, z1 - z0)
        elif (z - z0) % 50 == 0:
            log.info("segment %s: slice %d/%d (%.1f s/slice)", model, z, Z, (time.time() - t0) / max(z - z0 + 1, 1))
    g.attrs["pyclem3d_seg"] = {**g.attrs["pyclem3d_seg"], "foreground_fraction": n_fg / float((z1 - z0) * Y * X), "seconds": time.time() - t0}
    log.info("segmentation written to %s (%.1f%% foreground, %.0f s)", out_path, 100 * n_fg / float((z1 - z0) * Y * X), time.time() - t0)
    return out_path


def open_mask(path: str | os.PathLike) -> tuple[da.Array, dict[str, Any]]:
    """Lazy (Z, Y, X) mask + the attributes written by :func:`segment_volume`."""
    g = open_group_v2(path, mode="r")
    return da.from_zarr(str(path), component="0"), dict(g.attrs.get("pyclem3d_seg", {}))
