"""Per-volume memory strategy (plan §5): load into RAM if it fits a fraction of free memory."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dask.array as da
import numpy as np
import psutil

from .volume import Volume

log = logging.getLogger(__name__)


@dataclass
class MemoryDecision:
    strategy: str  # "ram" | "lazy"
    nbytes: int
    available: int
    fraction: float
    reason: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def available_ram() -> int:
    return int(psutil.virtual_memory().available)


def decide(nbytes: int, fraction: float = 0.3, force: str | None = None) -> MemoryDecision:
    avail = available_ram()
    budget = int(avail * fraction)
    if force in ("ram", "lazy"):
        return MemoryDecision(force, nbytes, avail, fraction, f"forced {force}")
    if nbytes <= budget:
        return MemoryDecision(
            "ram", nbytes, avail, fraction, f"{nbytes / 1e9:.2f} GB <= {budget / 1e9:.2f} GB budget"
        )
    return MemoryDecision(
        "lazy", nbytes, avail, fraction, f"{nbytes / 1e9:.2f} GB > {budget / 1e9:.2f} GB budget"
    )


def apply_memory_strategy(
    volume: Volume, fraction: float = 0.3, force: str | None = None
) -> Volume:
    """Return a Volume whose data is in RAM (numpy-backed dask) or left lazy."""
    d = decide(volume.nbytes, fraction, force)
    meta = dict(volume.metadata)
    meta["memory_decision"] = d.to_dict()
    if d.strategy == "ram":
        arr = np.asarray(volume.data.compute())
        data = da.from_array(arr, chunks=volume.data.chunksize, name=f"ram-{id(arr)}")
        log.info("memory: loaded %s into RAM (%s)", volume.source, d.reason)
        return volume.with_(data=data, memory_strategy="ram", metadata=meta)
    log.info("memory: keeping %s lazy (%s)", volume.source, d.reason)
    return volume.with_(memory_strategy="lazy", metadata=meta)
