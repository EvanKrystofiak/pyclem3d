"""EM stack alignment module (plan §4)."""

from .apply import apply_lazy, bake, reslice_views, transform_slice
from .badslices import detect_bad_slices, interpolate_over
from .pairwise import HAVE_CUPY, pairwise_measurements, phase_shift, rigid_pair
from .pipeline import AlignResult, align_stack, choose_level, plot_alignment, verify_stack
from .sidecar import SliceTransforms, sidecar_path
from .trajectory import chain, highpass, solve_multi_neighbour

__all__ = [
    "HAVE_CUPY",
    "AlignResult",
    "SliceTransforms",
    "align_stack",
    "apply_lazy",
    "bake",
    "chain",
    "choose_level",
    "detect_bad_slices",
    "highpass",
    "interpolate_over",
    "pairwise_measurements",
    "phase_shift",
    "plot_alignment",
    "reslice_views",
    "rigid_pair",
    "sidecar_path",
    "solve_multi_neighbour",
    "transform_slice",
    "verify_stack",
]
