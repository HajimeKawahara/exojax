from __future__ import annotations  # ← must be the first **statement**

"""
Deprecated shim: the real classes are in `exojax.rt`.
Kept only so that legacy code `from exojax.rt import ArtEmisPure`
continues to work until v3.0.
"""


import warnings
from importlib import import_module

warnings.warn(
    "`exojax.rt.atmrt` is deprecated. "
    "Import from `exojax.rt` instead (scheduled for removal in v3.0).",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public classes
_RT = import_module("exojax.rt")
_COMMON = import_module("exojax.rt.common")
ArtCommon = _COMMON.ArtCommon
ArtEmisPure = _RT.ArtEmisPure
ArtEmisScat = _RT.ArtEmisScat
ArtTransPure = _RT.ArtTransPure
ArtAbsPure = _RT.ArtAbsPure
ArtReflectPure = _RT.ArtReflectPure
ArtReflectEmis = _RT.ArtReflectEmis


__all__ = [
    "ArtCommon",
    "ArtEmisPure",
    "ArtEmisScat",
    "ArtTransPure",
    "ArtAbsPure",
    "ArtReflectPure",
    "ArtReflectEmis",
]
