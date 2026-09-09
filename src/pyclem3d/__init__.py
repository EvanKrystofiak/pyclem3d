"""pyclem3d: volumetric confocal <-> volume-EM correlation.

Headless core (never imports Qt/napari). See ``pyclem3d.napari`` for the viewer plugin.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyclem3d")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+src"

__all__ = ["__version__"]
