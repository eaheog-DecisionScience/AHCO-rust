"""Python bindings for the Rust AHCO optimizer."""

from .ahco import run_ahco3

optimize = run_ahco3

__all__ = ["optimize", "run_ahco3"]
