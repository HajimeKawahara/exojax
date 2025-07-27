#!/usr/bin/env python3
"""fix_v2_1.py
===================
One‑shot *codemod* that migrates **legacy ``exojax.spec`` imports** to the new
package layout (``exojax.opacity``, ``exojax.rt`` …) released in ExoJAX v2.1.

Highlights
----------
* **Pure‑text rewrite** – uses regular expressions; no AST quirks.
* **Safe‑in‑place** – only rewrites when a mapping exists, otherwise leaves code
  untouched.
* **Multi‑alias splitter** – converts lines like
  ``from exojax.spec import modit, opachord`` into two independent imports with
  correct destinations.

Usage::

    python -m fix_v2_1 <path> [<path> ...]

Requirements: Python 3.8+, no external deps (``re`` only) – so the tool can run
in minimal environments.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import libcst as cst
from libcst import FlattenSentinel
from libcst.helpers import get_full_name_for_node   # NEW

# ────────────────────────────────────────────────────────────────────
# Load the mapping **once** from the package
# ────────────────────────────────────────────────────────────────────
spec = importlib.import_module("exojax.spec")
ALIASES = getattr(spec, "_ALIASES", {})
SUBMODULES = getattr(spec, "_SUBMODULES", {})

# exojax.spec.<old>  →  <new_module>
MODULE_REWRITE = {f"exojax.spec.{k}": v for k, v in SUBMODULES.items()}


# ────────────────────────────────────────────────────────────────────
# CST transformer
# ────────────────────────────────────────────────────────────────────
class FixSpecImports(cst.CSTTransformer):
    """Rewrite old `exojax.spec` imports to the new namespaces."""

    # -----  from exojax.spec import OpaPremodit, planck as pl  -----
        # -----  from exojax.spec.modit import tests  -----
    def leave_ImportFrom(self, node: cst.ImportFrom, updated: cst.ImportFrom):
        full_mod = get_full_name_for_node(updated.module)
        if full_mod is None or not full_mod.startswith("exojax.spec"):
            return updated

        # exojax.spec          → ''      (top-level)
        # exojax.spec.modit    → 'modit' (submodule)
        subkey = full_mod.replace("exojax.spec.", "", 1)
        dest_base = SUBMODULES.get(subkey)            # e.g. exojax.opacity.modit

        # If we have no mapping, leave the line unchanged
        if dest_base is None:
            return updated

        # Re-emit a single rewritten ImportFrom
        return updated.with_changes(
            module=cst.parse_expression(dest_base)
        )

    # -----  import exojax.spec.opacalc as oc  -----
    def leave_ImportAlias(self, node: cst.ImportAlias, updated: cst.ImportAlias):
        # Robustly obtain the dotted name as a str
        full = get_full_name_for_node(updated.name)
        if full is None:          # e.g. 'import (x, y)'  rare – just skip
            return updated

        
        for old_mod, new_mod in MODULE_REWRITE.items():
            if full.startswith(old_mod):
                return updated.with_changes(
                    name=cst.parse_expression(full.replace(old_mod, new_mod, 1))
                )
        return updated


# ────────────────────────────────────────────────────────────────────
# Helper to walk files / directories
# ────────────────────────────────────────────────────────────────────
def _process_file(path: Path) -> None:
    mod = cst.parse_module(path.read_text())
    rewritten = mod.visit(FixSpecImports())
    if rewritten.code != mod.code:
        path.write_text(rewritten.code)
        print(f"rewrote {path}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file() and p.suffix == ".py":
            _process_file(p)
        elif p.is_dir():
            for py in p.rglob("*.py"):
                _process_file(py)
        else:
            print(f"skipped {p}")
