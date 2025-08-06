# src/exojax/database/molinfo/__init__.py
"""
Public interface for molecular information.

Heavy sub-modules are imported lazily so that
`import exojax.database.molinfo` stays lightweight.
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
    "molmass": "exojax.database.molinfo.mass:molmass",
    "isotope_molmass": "exojax.database.molinfo.mass:isotope_molmass",
    "molmass_isotope": "exojax.database.molinfo.mass:molmass_isotope",
    "mean_molmass_manual": "exojax.database.molinfo.mass:mean_molmass_manual",
    "element_mass": "exojax.database.molinfo.mass:element_mass",
    "m_transition_state": "exojax.database.molinfo.qstate:m_transition_state",
    "branch_to_number": "exojax.database.molinfo.qstate:branch_to_number",
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
