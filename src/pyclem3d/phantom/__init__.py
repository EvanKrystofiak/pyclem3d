"""Synthetic phantoms with known truth (plan §11)."""

from .generate import (
    MisalignTruth,
    PhantomTruth,
    make_misaligned_stack,
    make_phantom,
    shift_int,
)

__all__ = ["MisalignTruth", "PhantomTruth", "make_misaligned_stack", "make_phantom", "shift_int"]
