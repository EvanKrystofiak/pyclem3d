"""Metadata helpers: unit conversion, OME voxel sizes, PSF defaults (plan §1, §5, §13)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

_UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometre": 1.0,
    "um": 1e3,
    "µm": 1e3,  # micro sign
    "μm": 1e3,  # greek mu
    "micron": 1e3,
    "microns": 1e3,
    "micrometer": 1e3,
    "micrometre": 1e3,
    "mm": 1e6,
    "millimeter": 1e6,
    "cm": 1e7,
    "centimeter": 1e7,
    "m": 1e9,
    "meter": 1e9,
    "å": 0.1,  # angstrom sign
    "angstrom": 0.1,
    "a": 0.1,
    "inch": 2.54e7,
}


def to_nm(value: float | None, unit: str | None) -> float | None:
    """Convert a length to nm. Returns None when the unit is unknown or value missing."""
    if value is None:
        return None
    u = (unit or "").strip().lower()
    f = _UNIT_TO_NM.get(u)
    if f is None:
        return None
    return float(value) * f


def parse_ome_physical_sizes(ome_xml: str | None) -> dict[str, Any]:
    """Extract voxel sizes (nm), channel names/colours/emission and NA from OME-XML."""
    out: dict[str, Any] = {}
    if not ome_xml:
        return out
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return out

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    pixels = None
    for el in root.iter():
        if local(el.tag) == "Pixels":
            pixels = el
            break
    if pixels is None:
        return out
    a = pixels.attrib
    for ax in ("X", "Y", "Z"):
        v = a.get(f"PhysicalSize{ax}")
        u = a.get(f"PhysicalSize{ax}Unit", "µm")
        nm = to_nm(float(v), u) if v is not None else None
        if nm is not None:
            out[f"d{ax.lower()}_nm"] = nm
    out["dimension_order"] = a.get("DimensionOrder")
    chans = []
    for el in pixels:
        if local(el.tag) == "Channel":
            ca = el.attrib
            em = ca.get("EmissionWavelength")
            emu = ca.get("EmissionWavelengthUnit", "nm")
            color = ca.get("Color")
            hexcol = None
            if color is not None:
                try:
                    rgba = int(color) & 0xFFFFFFFF
                    hexcol = f"{(rgba >> 24) & 0xFF:02X}{(rgba >> 16) & 0xFF:02X}{(rgba >> 8) & 0xFF:02X}"
                except ValueError:
                    hexcol = None
            chans.append(
                {
                    "name": ca.get("Name"),
                    "emission_nm": to_nm(float(em), emu) if em else None,
                    "color": hexcol,
                }
            )
    if chans:
        out["channels"] = chans
    for el in root.iter():
        if local(el.tag) == "Objective":
            na = el.attrib.get("LensNA")
            if na:
                out["na"] = float(na)
            imm = el.attrib.get("Immersion")
            if imm:
                out["immersion"] = imm
    return out


IMMERSION_RI = {"oil": 1.515, "water": 1.33, "glycerol": 1.47, "air": 1.0, "silicone": 1.41}


def psf_fwhm_from_optics(
    na: float | None, emission_nm: float | None, ri: float | str | None = None
) -> tuple[float, float] | None:
    """Approximate confocal PSF FWHM (axial, lateral) in nm from NA, emission and RI.

    lateral ~ 0.51 lambda / NA ; axial ~ 0.88 lambda / (n - sqrt(n^2 - NA^2)).
    These are defaults for landmark uncertainty and the PSF-aware z view (plan §1),
    not a claim about the instrument; the user can override them.
    """
    if not na or not emission_nm:
        return None
    if isinstance(ri, str):
        ri = IMMERSION_RI.get(ri.lower(), 1.515)
    n = float(ri) if ri else 1.515
    na = float(na)
    if na >= n:
        na = n * 0.999
    lateral = 0.51 * emission_nm / na
    axial = 0.88 * emission_nm / (n - math.sqrt(n * n - na * na))
    return (float(axial), float(lateral))


def psf_fallback(voxel_size_nm: tuple[float, float, float]) -> tuple[float, float]:
    """Fallback sigma ~ voxel size (plan §1), reported as FWHM = 2.355 sigma."""
    dz, dy, dx = voxel_size_nm
    return (2.355 * dz, 2.355 * max(dy, dx))


def suspicious_voxel_size(vs: tuple[float, float, float]) -> str | None:
    """Return a warning if a voxel size looks like a missing header value (plan §13)."""
    v = [float(x) for x in vs]
    if any(x <= 0 for x in v):
        return "voxel size has a non-positive entry; header is probably missing it"
    if all(abs(x - 1.0) < 1e-9 for x in v):
        return "voxel size is exactly 1 nm in all axes; header value is probably a placeholder"
    if max(v) / min(v) > 100:
        return "voxel size anisotropy > 100x; check units"
    return None
