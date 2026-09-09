"""``pyclem3d doctor`` (plan §5): measure RAM, cores, GPU and data-path read throughput, then
write a config with chunk sizes, cache location, dask scheduler and GPU on/off."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

from .config import DEFAULTS, load_config, save_config

log = logging.getLogger(__name__)


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cupy": False, "nvidia_smi": None}
    try:
        import cupy  # type: ignore

        n = cupy.cuda.runtime.getDeviceCount()
        info["cupy"] = n > 0
        if n > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            info["name"] = (
                props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            )
            info["memory_bytes"] = int(props["totalGlobalMem"])
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                info["nvidia_smi"] = out.stdout.strip().splitlines()[0]
        except Exception:  # pragma: no cover
            pass
    return info


def read_throughput(path: str | os.PathLike | None, max_bytes: int = 256 << 20) -> dict[str, Any]:
    """MB/s of sequential reads of ``path`` (a file, or the largest file in a directory), or of
    a temporary file written in ``path``/cache dir when no data is given."""
    target: Path | None = None
    tmp = None
    if path is not None:
        p = Path(path)
        if p.is_file():
            target = p
        elif p.is_dir():
            files = sorted(
                (f for f in p.rglob("*") if f.is_file()),
                key=lambda f: f.stat().st_size,
                reverse=True,
            )
            target = files[0] if files else None
    if target is None:
        d = Path(path) if path is not None and Path(path).is_dir() else Path(tempfile.gettempdir())
        tmp = d / f"pyclem3d_doctor_{os.getpid()}.bin"
        with tmp.open("wb") as f:
            block = os.urandom(1 << 20)
            for _ in range(min(max_bytes, 128 << 20) >> 20):
                f.write(block)
        target = tmp
    size = target.stat().st_size
    n = min(size, max_bytes)
    t0 = time.perf_counter()
    read = 0
    with target.open("rb") as f:
        while read < n:
            chunk = f.read(min(8 << 20, n - read))
            if not chunk:
                break
            read += len(chunk)
    dt = max(time.perf_counter() - t0, 1e-6)
    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:  # pragma: no cover
            pass
    return {
        "path": str(target),
        "bytes": read,
        "seconds": dt,
        "mb_per_s": read / 1e6 / dt,
        "synthetic": tmp is not None,
    }


def run_doctor(
    data_path: str | os.PathLike | None = None,
    write: bool = True,
    config_path: str | os.PathLike | None = None,
    gpu: str = "auto",
) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    cores = psutil.cpu_count(logical=False) or 1
    threads = psutil.cpu_count(logical=True) or cores
    g = gpu_info()
    tp = read_throughput(data_path)
    cfg = load_config(config_path)
    # chunk size: larger chunks on slow storage (fewer requests), smaller on fast NVMe
    mbps = tp["mb_per_s"]
    if mbps < 100:
        chunk = 128 << 20
    elif mbps < 500:
        chunk = 64 << 20
    else:
        chunk = 32 << 20
    scheduler = "threads"
    n_workers = min(threads, 16)
    use_gpu = (g["cupy"] if gpu == "auto" else gpu == "on") and gpu != "off"
    cfg.update(
        {
            "chunk_bytes": int(chunk),
            "scheduler": scheduler,
            "n_workers": int(n_workers),
            "gpu": bool(use_gpu),
            "memory_fraction": DEFAULTS["memory_fraction"],
        }
    )
    if data_path is not None and Path(data_path).is_dir():
        cfg["cache_dir"] = (
            str(Path(data_path) / ".pyclem3d_cache") if mbps >= 500 else cfg["cache_dir"]
        )
    report = {
        "ram_total_gb": vm.total / 1e9,
        "ram_available_gb": vm.available / 1e9,
        "cores": cores,
        "threads": threads,
        "gpu": g,
        "throughput": tp,
        "config": cfg,
        "notes": [],
    }
    if vm.total < 16e9:
        report["notes"].append(
            "under 16 GB RAM: confocal stacks may stay lazy; consider OME-Zarr conversion of the EM"
        )
    if g["nvidia_smi"] and not g["cupy"]:
        report["notes"].append(
            "an NVIDIA GPU is present but cupy is not installed: pip install 'pyclem3d[gpu]' to enable GPU phase correlation"
        )
    if mbps < 100:
        report["notes"].append(
            "slow data path (<100 MB/s): convert to OME-Zarr on local storage for interactive use"
        )
    if write:
        p = save_config(cfg, config_path)
        report["config_path"] = str(p)
    return report


def format_doctor(rep: dict[str, Any]) -> str:
    g = rep["gpu"]
    gpu = g.get("name") or g.get("nvidia_smi") or "none"
    lines = [
        f"RAM: {rep['ram_total_gb']:.1f} GB total, {rep['ram_available_gb']:.1f} GB available",
        f"CPU: {rep['cores']} cores / {rep['threads']} threads",
        f"GPU: {gpu} (cupy {'available' if g['cupy'] else 'not installed'})",
        f"Read throughput: {rep['throughput']['mb_per_s']:.0f} MB/s ({'synthetic file' if rep['throughput']['synthetic'] else rep['throughput']['path']})",
        f"Config: chunks {rep['config']['chunk_bytes'] >> 20} MB, scheduler {rep['config']['scheduler']} x{rep['config']['n_workers']}, gpu {'on' if rep['config']['gpu'] else 'off'}, cache {rep['config']['cache_dir']}",
    ]
    if rep.get("config_path"):
        lines.append(f"Written to {rep['config_path']}")
    for n in rep["notes"]:
        lines.append(f"note: {n}")
    return "\n".join(lines)
