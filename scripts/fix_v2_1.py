#!/usr/bin/env python3
"""ejxfix-spec2new
===================
One‑shot *codemod* that migrates **legacy ``exojax.spec`` imports** to the new
package layout (``exojax.opacity``, ``exojax.rt`` …) released in ExoJAX ≥ 0.13.

Highlights
----------
* **Pure‑text rewrite** – uses regular expressions; no AST quirks.
* **Safe‑in‑place** – only rewrites when a mapping exists, otherwise leaves code
  untouched.
* **Multi‑alias splitter** – converts lines like
  ``from exojax.spec import modit, opachord`` into two independent imports with
  correct destinations.

Usage::

    python -m ejxfix_spec2new <path> [<path> ...]

Requirements: Python 3.8+, no external deps (``re`` only) – so the tool can run
in minimal environments.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Mapping table: old sub‑module → new fully‑qualified path
# ---------------------------------------------------------------------------
ALIAS_MAP: Dict[str, str] = {
    # --- opacity layer ------------------------------------------------------
    "opacalc":        "exojax.opacity.opacalc",
    "initspec":       "exojax.opacity.initspec",
    "premodit":       "exojax.opacity.premodit.premodit",
    "modit":          "exojax.opacity.modit.modit",
    "lpf":            "exojax.opacity.lpf.lpf",
    "set_ditgrid":    "exojax.opacity._common.set_ditgrid",
    "optgrid":        "exojax.opacity.premodit.optgrid",
    "lbd":            "exojax.opacity.premodit.lbd",
    "lsd":            "exojax.opacity._common.lsd",
    "lbderror":       "exojax.opacity.premodit.lbderror",
    "dit":            "exojax.opacity.modit.dit",
    "ditkernel":      "exojax.opacity._common.ditkernel",
    "make_numatrix":  "exojax.opacity.lpf.make_numatrix",
    "opacont":        "exojax.opacity.opacont",
    "rayleigh":       "exojax.opacity.rayleigh",
    "generate_elower_grid_trange": "exojax.opacity.premodit.generate_elower_grid_trange",
    "profconv":       "exojax.opacity._common.profconv",
    "lpffilter":      "exojax.opacity._common.lpffilter",
    "chord":          "exojax.rt.chord",
    "atmrt":          "exojax.rt.atmrt",
    "opart":          "exojax.rt.opart",
    "layeropacity":   "exojax.rt.layeropacity",
    "planck":         "exojax.rt.planck",
    "rtlayer":        "exojax.rt.rtlayer",
    "rtransfer":      "exojax.rt.rtransfer",
    "toon":           "exojax.rt.toon",
    "twostream":      "exojax.rt.twostream",
    "mie":            "exojax.database.mie",
    "api":            "exojax.database.api",
    "atomll":        "exojax.database.atomll",
    "atomllapi":     "exojax.database.atomllapi",
    "contdb":        "exojax.database.contdb",
    "customapi":     "exojax.database.customapi",
    "dbmanager":     "exojax.database.dbmanager", 
    "exomol":        "exojax.database.exomol",
    "exomolhr":      "exojax.database.exomolhr",
    "hitran":        "exojax.database.hitran",
    "hitranapi":     "exojax.database.hitranapi",
    "hitrancia":     "exojax.database.hitrancia",
    "hminus":        "exojax.database.hminus",
    "moldb":         "exojax.database.moldb",
    "molinfo":       "exojax.database.molinfo",
    "multimol":      "exojax.database.multimol",
    "nonair":        "exojax.database.nonair",
    "pardb":         "exojax.database.pardb",
    "qstate":        "exojax.database.qstate",
    "limb_darkening":"exojax.postproc.limb_darkening",
    "response":      "exojax.postproc.response",
    "specop":        "exojax.postproc.specop",
    "spin_rotation": "exojax.postproc.spin_rotation",
}

# Pre‑compiled regex patterns -------------------------------------------------
RE_SPEC_FROM_SUB = re.compile(r"\bfrom\s+exojax\.spec\.(\w+)\b")
RE_SPEC_IMPORT_SUB = re.compile(r"\bimport\s+exojax\.spec\.(\w+)\b")
# Handles: `from exojax.spec import a, b, c`  (capturing the alias list)
RE_SPEC_IMPORT_GROUP = re.compile(
    r"^\s*from\s+exojax\.spec\s+import\s+([A-Za-z0-9_,\s]+)")


# ---------------------------------------------------------------------------
# 2. Core rewrite helpers
# ---------------------------------------------------------------------------

def _rewrite_from_sub(line: str) -> str:
    """Rewrite ¬from exojax.spec.<sub> import …¬ pattern."""

    def _sub(match: re.Match[str]) -> str:  # noqa: D401 – inner func
        sub = match.group(1)
        return f"from {ALIAS_MAP.get(sub, f'exojax.spec.{sub}')}"

    return RE_SPEC_FROM_SUB.sub(_sub, line)


def _rewrite_import_sub(line: str) -> str:
    """Rewrite ¬import exojax.spec.<sub> as xyz¬ pattern."""

    def _sub(match: re.Match[str]) -> str:
        sub = match.group(1)
        return f"import {ALIAS_MAP.get(sub, f'exojax.spec.{sub}')}"

    return RE_SPEC_IMPORT_SUB.sub(_sub, line)


def _rewrite_group_import(line: str) -> str | None:
    """Split and rewrite multi‑alias group import.

    Returns *None* if no rewrite was needed; otherwise the **entire replacement
    block** (may contain line‑breaks).
    """
    m = RE_SPEC_IMPORT_GROUP.match(line)
    if not m:
        return None

    aliases: List[str] = [a.strip() for a in m.group(1).split(",") if a.strip()]
    new_lines: List[str] = []
    keep: List[str] = []
    for name in aliases:
        if name in ALIAS_MAP:
            new_path = ALIAS_MAP[name]
            module_path, attr = new_path.rsplit(".", 1)
            new_lines.append(f"from {module_path} import {attr}\n")
        else:
            keep.append(name)

    if keep:
        new_lines.insert(0, f"from exojax.spec import {', '.join(keep)}\n")
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# 3. File‑level processing
# ---------------------------------------------------------------------------

def rewrite_content(text: str) -> str:
    """Return rewritten source (or original text if no changes)."""
    changed = False
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        # case 1: from exojax.spec.<sub> …
        new_line = _rewrite_from_sub(line)
        # case 2: import exojax.spec.<sub>
        new_line = _rewrite_import_sub(new_line)
        # case 3: grouped alias line
        group_replacement = _rewrite_group_import(new_line)
        if group_replacement is not None:
            new_line = group_replacement
        if new_line != line:
            changed = True
        out_lines.append(new_line)
    return "".join(out_lines) if changed else text


def rewrite_file(path: Path) -> None:
    """Rewrite one file in‑place if needed."""
    raw = path.read_text()
    new = rewrite_content(raw)
    if raw != new:
        path.write_text(new)
        print(f"[rewrite] {path}")


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Migrate exojax.spec imports to the new package layout")
    parser.add_argument("paths", nargs="+", help="Files or directories to process")
    args = parser.parse_args()

    for root in map(Path, args.paths):
        if root.is_dir():
            for py in root.rglob("*.py"):
                rewrite_file(py)
        elif root.suffix == ".py":
            rewrite_file(root)
        else:
            print(f"[skip] {root} (not .py)")


if __name__ == "__main__":
    main()
