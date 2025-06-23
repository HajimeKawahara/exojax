"""
Deprecated shim: the real classes are in `exojax.rt`.
Kept only so that legacy code `from exojax.rt import OpartEmisPure`
continues to work until v3.0.
"""


import warnings
from importlib import import_module

warnings.warn(
    "`exojax.rt.opart` is deprecated. "
    "Import from `exojax.rt` instead (scheduled for removal in v3.0).",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public classes
_RT = import_module("exojax.rt")
OpartEmisPure = _RT.OprtEmisPure
OpartEmisScat = _RT.OpartEmisScat
OpartReflectPure = _RT.OpartReflectPure
OpartReflectEmis = _RT.OPartReflectEmis


__all__ = [
    "OpartEmisPure",
    "OpartEmisScat",
    "OpartReflectPure",
    "OpartReflectEmis",
]
