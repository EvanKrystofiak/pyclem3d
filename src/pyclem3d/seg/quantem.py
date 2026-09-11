"""QuantEM backend (Arrojo e Drigo lab, https://huggingface.co/ArrojoeDrigoLab/quantem).

Same contract as :mod:`pyclem3d.seg.mitonet`: slice-by-slice 2D inference streamed into a zarr
mask with the EM's geometry, so everything downstream (synthetic fluorescence, registration,
fused export) is backend-agnostic. QuantEM models take the pixel size and rescale to their
canonical resolution themselves (8 nm for mitochondria), return instance labels plus a
probability map, and cache their weights through ``huggingface_hub``.

Optional dependency: ``quantem-core`` (installed by the ``seg`` extra).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..io.volume import Volume
from ..io.writers import open_group_v2

log = logging.getLogger(__name__)

QUANTEM_ALIASES = {
    "mito": "quantem/mito",
    "nucleus": "omniem/nucleus",
    "er": "omniem/er",
    "ld": "omniem/ld",
}


def is_quantem_model(name: str) -> bool:
    """'quantem/mito', 'omniem/nucleus' (or a bare organelle handled by QuantEM) -> True."""
    return "/" in name and name.split("/", 1)[0] in ("quantem", "omniem")


def load_quantem(model: str = "quantem/mito", device: str = "auto", allow_network: bool = True):
    """A ``QuantEMModel`` for a model id ('quantem/mito', 'omniem/nucleus', ...) or alias."""
    try:
        from quantem_em.api import load_model
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError("QuantEM segmentation needs quantem-core: uv sync --extra seg") from e
    model_id = QUANTEM_ALIASES.get(model, model)
    return load_model(model_id, device=device, allow_network=allow_network)


def segment_volume_quantem(
    volume: Volume,
    out_path: str | os.PathLike,
    model: str = "quantem/mito",
    semantic: bool = True,
    channel: int = 0,
    z_range: tuple[int, int] | None = None,
    device: str = "auto",
    threshold: float | None = None,
    save_probability: bool = True,
    progress=None,
    engine=None,
) -> Path:
    """Segment every slice of ``volume`` with a QuantEM model and write ``out_path`` (zarr).

    Output array ``0`` is a uint8 semantic mask (or uint32 2D instance labels with
    ``semantic=False``); by default a second array ``prob`` holds the foreground probability
    scaled to uint8. The probability is what the registration uses when present: blurred into
    the synthetic fluorescence it weights each voxel by the model's confidence instead of a
    hard 0/1 decision.
    """
    out_path = Path(out_path)
    m = engine or load_quantem(model, device)
    Z, Y, X = volume.shape_zyx
    z0, z1 = (0, Z) if z_range is None else (max(0, z_range[0]), min(Z, z_range[1]))
    g = open_group_v2(out_path, mode="w")
    dtype = np.uint8 if semantic else np.uint32
    arr = g.create_array("0", shape=(Z, Y, X), chunks=(1, Y, X), dtype=dtype)
    prob_arr = (
        g.create_array("prob", shape=(Z, Y, X), chunks=(1, Y, X), dtype=np.uint8)
        if save_probability
        else None
    )
    px = float(max(volume.voxel_size_nm[1], volume.voxel_size_nm[2]))
    spec = getattr(m, "spec", None)
    g.attrs["pyclem3d_seg"] = {
        "backend": "quantem",
        "model": QUANTEM_ALIASES.get(model, model),
        "semantic": bool(semantic),
        "pixel_size_nm": px,
        "canonical_nm": getattr(spec, "canonical_nm", None),
        "threshold": threshold,
        "source": volume.source,
        "voxel_size_nm": list(volume.voxel_size_nm),
        "world_affine": volume.world_affine.tolist(),
        "z_range": [int(z0), int(z1)],
    }
    data = volume.data[channel]
    t0 = time.time()
    n_fg = 0
    n_obj = 0
    for z in range(z0, z1):
        img = np.asarray(data[z].compute())
        res = m.segment(img, pixel_size_nm=px, threshold=threshold)
        plane = res.mask.astype(np.uint8) if semantic else res.labels.astype(np.uint32)
        arr[z] = plane
        if prob_arr is not None:
            prob_arr[z] = np.clip(np.round(res.probability * 255.0), 0, 255).astype(np.uint8)
        n_fg += int(res.mask.sum())
        n_obj += int(res.n_objects)
        if progress is not None:
            progress(z - z0 + 1, z1 - z0)
        elif (z - z0) % 50 == 0:
            log.info(
                "quantem %s: slice %d/%d (%.2f s/slice)",
                model,
                z,
                Z,
                (time.time() - t0) / max(z - z0 + 1, 1),
            )
    g.attrs["pyclem3d_seg"] = {
        **g.attrs["pyclem3d_seg"],
        "foreground_fraction": n_fg / float((z1 - z0) * Y * X),
        "objects_per_slice": n_obj / float(max(z1 - z0, 1)),
        "seconds": time.time() - t0,
    }
    log.info(
        "quantem segmentation written to %s (%.1f%% foreground, %.0f s)",
        out_path,
        100 * n_fg / float((z1 - z0) * Y * X),
        time.time() - t0,
    )
    return out_path


def segment_any(
    volume: Volume,
    out_path: str | os.PathLike,
    model: str = "mito",
    backend: str = "auto",
    **kw: Any,
) -> Path:
    """Dispatch to empanada or QuantEM by ``backend`` ('empanada' | 'quantem' | 'auto' from the model id)."""
    if backend == "auto":
        backend = "quantem" if is_quantem_model(model) else "empanada"
    if backend == "quantem":
        return segment_volume_quantem(volume, out_path, model=model, **kw)
    from .mitonet import segment_volume

    return segment_volume(volume, out_path, model=model, **kw)


def open_probability(path: str | os.PathLike):
    """Lazy (Z, Y, X) float32 probability in [0, 1] from a QuantEM mask zarr, or None."""
    import dask.array as da

    g = open_group_v2(path, mode="r")
    if "prob" not in g:
        return None
    return da.from_zarr(str(path), component="prob").astype(np.float32) / 255.0
