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

import warnings

# Real implementations now live in the sub-packages
from .exomol.api import MdbExomol as _MdbExomol
from .hitemp.api import MdbHitemp as _MdbHitemp
from .hitran.api import MdbHitran as _MdbHitran

__all__ = ["MdbExomol", "MdbHitemp", "MdbHitran"]


def __getattr__(name: str):
    """Lazy attribute loader that also raises a deprecation warning."""
    if name == "MdbExomol":
        warnings.warn(
            "exojax.database.api.MdbExomol is deprecated. "
            "Import it from exojax.database.exomol.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _MdbExomol
    if name == "MdbHitemp":
        warnings.warn(
            "exojax.database.api.MdbHitemp is deprecated. "
            "Import it from exojax.database.hitemp.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _MdbHitemp
    if name == "MdbHitran":
        warnings.warn(
            "exojax.database.api.MdbHitran is deprecated. "
            "Import it from exojax.database.hitran.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _MdbHitran
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Eager bindings for static type checkers and IDEs
MdbExomol = _MdbExomol
MdbHitemp = _MdbHitemp
MdbHitran = _MdbHitran
