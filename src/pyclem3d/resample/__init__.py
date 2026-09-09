"""Lazy resampling, EM slab projections and z views (plan §7)."""

from .grid import DisplacementGrid, Grid, InverseMapper, invert_numerically
from .lazy import resample_plane, resample_to_grid
from .slab import (
    SlabCache,
    SlabParams,
    em_in_lm_stack,
    em_level_for_lm,
    em_slab_for_lm_slice,
    slab_brute_force,
    slab_thickness_nm,
)
from .zview import (
    em_z_range_for_lm_slice,
    lm_plane_at_em_slice,
    lm_z_at_em_slice,
    z_profile,
    z_readout,
)

__all__ = [
    "DisplacementGrid",
    "Grid",
    "InverseMapper",
    "SlabCache",
    "SlabParams",
    "em_in_lm_stack",
    "em_level_for_lm",
    "em_slab_for_lm_slice",
    "em_z_range_for_lm_slice",
    "invert_numerically",
    "lm_plane_at_em_slice",
    "lm_z_at_em_slice",
    "resample_plane",
    "resample_to_grid",
    "slab_brute_force",
    "slab_thickness_nm",
    "z_profile",
    "z_readout",
]
