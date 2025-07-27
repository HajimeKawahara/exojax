# src/exojax/opacity/__init__.py
"""
Public interface for opacity algorithms.

Heavy sub-modules are imported lazily so that
`import exojax.opacity` stays lightweight.
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
    "OpaPremodit": "exojax.opacity.premodit.api:OpaPremodit",
    "OpaDirect":   "exojax.opacity.lpf.api:OpaDirect",
    "OpaModit":    "exojax.opacity.modit.api:OpaModit",
    "OpaCKD":      "exojax.opacity.ckd.api:OpaCKD",
    "OpaCIA": "exojax.opacity.opacont:OpaCIA",
    "OpaRayleigh": "exojax.opacity.opacont:OpaRayleigh",
    "OpaHminus": "exojax.opacity.opacont:OpaHminus",
    "OpaMie": "exojax.opacity.opacont:OpaMie", 
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
