"""
Deprecated shim: the real classes are in `exojax.opacity`.
Kept only so that legacy code `from exojax.opacity import OpaPremodit`
continues to work until v3.0.
"""

from __future__ import annotations   # ← must be the first **statement**

import warnings
from importlib import import_module

warnings.warn(
    "`exojax.opacity.opacalc` is deprecated. "
    "Import from `exojax.opacity` instead (scheduled for removal in v3.0).",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public classes
_OPACITY = import_module("exojax.opacity")
OpaPremodit = _OPACITY.OpaPremodit
OpaDirect   = _OPACITY.OpaDirect
OpaModit    = _OPACITY.OpaModit

__all__ = ["OpaPremodit", "OpaDirect", "OpaModit"]
