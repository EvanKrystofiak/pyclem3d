"""User configuration written by ``pyclem3d doctor`` (plan §5): chunk sizes, cache dir, dask
scheduler, GPU on/off, memory fraction. Everything has a CPU default; GPU/cluster are opt-in."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "cache_dir": str(Path.home() / ".pyclem3d_cache"),
    "chunk_bytes": 32 << 20,
    "scheduler": "threads",  # threads | processes | synchronous | distributed
    "n_workers": None,
    "gpu": False,
    "memory_fraction": 0.3,
    "pyramid_min_size": 64,
}


def config_path() -> Path:
    env = os.environ.get("PYCLEM3D_CONFIG")
    return Path(env) if env else Path.home() / ".pyclem3d" / "config.json"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else config_path()
    cfg = dict(DEFAULTS)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:  # pragma: no cover
            log.warning("could not read config %s: %s", p, e)
    return cfg


def save_config(cfg: dict[str, Any], path: str | os.PathLike | None = None) -> Path:
    p = Path(path) if path is not None else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def apply_dask_config(cfg: dict[str, Any]) -> Any:
    """Configure dask from the config; returns a distributed Client when that scheduler is chosen."""
    import dask

    sched = cfg.get("scheduler", "threads")
    n = cfg.get("n_workers")
    if sched == "distributed":
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:  # pragma: no cover - optional
            log.warning("dask.distributed not installed; falling back to threads")
            dask.config.set(scheduler="threads")
            return None
        cluster = LocalCluster(n_workers=n or None, threads_per_worker=2)
        return Client(cluster)
    kw: dict[str, Any] = {"scheduler": sched}
    if n:
        kw["num_workers"] = int(n)
    dask.config.set(**kw)
    return None
