# src/exojax/database/__init__.py
"""
Public interface for database information.

Heavy sub-modules are imported lazily so that
`import exojax.database` stays lightweight.
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
    "MdbExomol": "exojax.database.exomol.api:MdbExomol",
    "MdbHitemp": "exojax.database.hitemp.api:MdbHitemp",
    "MdbHitran": "exojax.database.hitran.api:MdbHitran",
    "MdbHargreaves": "exojax.database.hargreaves.api:MdbHargreaves",
    "XdbExomolHR": "exojax.database.exomolhr.api:XdbExomolHR",
    "AdbVald": "exojax.database.vald.api:AdbVald",
    "AdbSepVald": "exojax.database.vald.api:AdbSepVald",
    "AdbKurucz": "exojax.database.kurucz.api:AdbKurucz",
    "PdbCloud": "exojax.database.pardb:PdbCloud",
    "CdbCIA": "exojax.database.cia.api:CdbCIA",
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
