# src/exojax/rt/__init__.py
"""
Public interface for rt (radiative transfer) algorithms.

Heavy sub-modules are imported lazily so that
`import exojax.rt` stays lightweight.
"""

from __future__ import annotations
import importlib
import sys
from types import ModuleType
from typing import Final

# --------------------------------------------------------------------
# Mapping: public name  →  "module.path:attribute"
# --------------------------------------------------------------------
_ALIAS: Final[dict[str, str]] = {
    "ArtEmisPure": "exojax.rt.emis:ArtEmisPure",
    "ArtEmisScat": "exojax.rt.emis:ArtEmisScat",
    "ArtTransPure": "exojax.rt.trans:ArtTransPure",
    "ArtAbsPure": "exojax.rt.reflect:ArtAbsPure",
    "ArtReflectPure": "exojax.rt.reflect:ArtReflectPure",
    "ArtReflectEmis": "exojax.rt.reflect:ArtReflectEmis",
    "OpartEmisPure": "exojax.rt.emis:OpartEmisPure",
    "OpartEmisScat": "exojax.rt.emis:OpartEmisScat",
    "OpartReflectPure": "exojax.rt.reflect:OpartReflectPure",
    "OpartReflectEmis": "exojax.rt.reflect:OpartReflectEmis",
}

__all__ = list(_ALIAS)          # tab completion & help()

# --------------------------------------------------------------------
# Lazy loader
# --------------------------------------------------------------------
def __getattr__(name: str):  # noqa: D401
    """Resolve public classes on first access (lazy import)."""
    target = _ALIAS.get(name)
    if target is None:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module: ModuleType = importlib.import_module(module_path)
    obj = getattr(module, attr)

    # Cache the resolved object so future look-ups are fast
    globals()[name] = obj
    return obj
