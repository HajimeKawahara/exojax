# exojax/database/api.py  (legacy shim kept for backward compatibility)
"""
Thin compatibility wrapper.

Please import the classes directly from their new modules:

    from exojax.database.exomol.api import MdbExomol
    from exojax.database.hitemp.api import MdbHitemp
    from exojax.database.hitran.api import MdbHitran
    
This shim will be removed in a future **major** release.
"""
from __future__ import annotations

import importlib
import warnings

__all__ = ["MdbExomol", "MdbHitemp", "MdbHitran"]

_ALIASES = {
    "MdbExomol": ("exojax.database.exomol.api", "MdbExomol"),
    "MdbHitemp": ("exojax.database.hitemp.api", "MdbHitemp"),
    "MdbHitran": ("exojax.database.hitran.api", "MdbHitran"),
}


def _resolve(name: str):
    module_path, attr = _ALIASES[name]
    module = importlib.import_module(module_path)
    resolved = getattr(module, attr)
    globals()[name] = resolved
    return resolved


def __getattr__(name: str):
    """Lazy attribute loader that also raises a deprecation warning."""
    if name == "MdbExomol":
        warnings.warn(
            "exojax.database.api.MdbExomol is deprecated. "
            "Import it from exojax.database.exomol.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve(name)
    if name == "MdbHitemp":
        warnings.warn(
            "exojax.database.api.MdbHitemp is deprecated. "
            "Import it from exojax.database.hitemp.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve(name)
    if name == "MdbHitran":
        warnings.warn(
            "exojax.database.api.MdbHitran is deprecated. "
            "Import it from exojax.database.hitran.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
