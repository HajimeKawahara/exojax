# exojax/database/contdb.py  (legacy shim kept for backward compatibility)
"""
Thin compatibility wrapper.

Please import the classes directly from their new modules:

    from exojax.database.cia.api import CdbCIA
    
This shim will be removed in a future **major** release.
"""
from __future__ import annotations

import warnings

# Real implementations now live in the sub-packages
from .cia.api import CdbCIA as _CdbCIA

__all__ = ["CdbCIA"]


def __getattr__(name: str):
    """Lazy attribute loader that also raises a deprecation warning."""
    if name == "CdbCIA":
        warnings.warn(
            "exojax.database.contdb.CdbCIA is deprecated. "
            "Import it from exojax.database.cia.api instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _CdbCIA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Eager bindings for static type checkers and IDEs
CdbCIA = _CdbCIA
