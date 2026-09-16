"""Offline PyExoCross loader and MDBSnapshot comparison against RADIS.

Run in a dedicated process. Use --public-api to validate the production loader.
Only copies of bundled raw files are read. No downloads are allowed.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
from scipy.constants import c, h, k

from exojax.database.contracts import Lines
from exojax.database.core.broadening import line_strength_from_Einstein_coeff
from exojax.database.exomol.api import MdbExomol
from exojax.database.exomol.partition_function import qr_interp
from exojax.opacity import OpaDirect, OpaPremodit
from exojax.test.data import get_testdata_filename
from exojax.utils.constants import hcperk

from snapshot import load_snapshot


NURANGE = (4330.0, 4360.0)
TEMPERATURES = (500.0, 1000.0, 1500.0)
PRESSURES = (0.1, 1.0, 10.0)
REFERENCE_C2 = h * c / k * 100.0
CASES = (
    ("CO-H2", "CO", {}),
    ("CO-He", "CO", {"bkgdatm": "He"}),
    ("CO-default", "CO", {"broadf": False}),
    ("CO-filtered", "CO", {"crit": 1e-25, "elower_max": 2000.0}),
    ("H2O-H2", "H2O", {}),
    ("H2O-He", "H2O", {"bkgdatm": "He"}),
    ("H2O-filtered", "H2O", {"crit": 1e-25, "elower_max": 2000.0}),
)


def errors(actual, expected, *, spectrum=False):
    """Use unfloored line errors; spectra have a 1e-12 peak tail floor."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError("Mismatched shape or non-finite comparison values")
    scale = float(np.max(np.abs(expected)))
    if not scale:
        if spectrum:
            raise AssertionError("The reference spectrum is identically zero")
        np.testing.assert_array_equal(actual, expected)
        return {"max_relative": 0.0, "peak_relative": 0.0}
    delta = np.abs(actual - expected)
    floor = scale * 1e-12 if spectrum else np.finfo(float).tiny
    return {
        "max_relative": float(np.max(delta / np.maximum(np.abs(expected), floor))),
        "peak_relative": float(np.max(delta) / scale),
    }


def check(actual, expected, *, spectrum=False):
    result = errors(actual, expected, spectrum=spectrum)
    scale = float(np.max(np.abs(expected))) if spectrum else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=1e-9 if spectrum else 2e-12,
        atol=scale * 1e-12, equal_nan=False,
    )
    return result


def copy_raw_data(root, molecule):
    isotope = {"CO": "12C-16O", "H2O": "1H2-16O"}[molecule]
    source = Path(get_testdata_filename(molecule)) / isotope / "SAMPLE"
    target = root / molecule / isotope / "SAMPLE"
    target.mkdir(parents=True)
    hashes = {}
    for path in sorted(source.iterdir()):
        if path.name.endswith((".bz2", ".def", ".pf", ".broad")):
            shutil.copy2(path, target / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return target, hashes


def sorted_snapshot(snapshot, order):
    return replace(
        snapshot,
        lines=Lines(*(getattr(snapshot.lines, name)[order] for name in (
            "nu_lines", "elower", "line_strength_ref_original",
        ))),
        n_Texp=snapshot.n_Texp[order], alpha_ref=snapshot.alpha_ref[order],
    )


def direct_database(snapshot, einstein_a):
    """Supply the additional A array required by LPF, outside MDBSnapshot."""
    return SimpleNamespace(
        dbtype="exomol", molmass=snapshot.meta.molmass,
        nu_lines=snapshot.lines.nu_lines, elower=snapshot.lines.elower,
        logsij0=np.log(snapshot.lines.line_strength_ref_original), A=einstein_a,
        n_Texp=snapshot.n_Texp, alpha_ref=snapshot.alpha_ref,
        qr_interp=lambda T, Tref: qr_interp(
            T, Tref, snapshot.meta.T_gQT, snapshot.meta.gQT,
        ),
    )


def compare_case(path, options, nu_grid, *, public_api=False):
    # Block accidental downloads and avoid using a user's registered databank.
    with patch.object(MdbExomol, "download", side_effect=AssertionError("Offline check attempted a download")), \
         patch.object(MdbExomol, "is_registered", return_value=False):
        reference = MdbExomol(
            str(path), NURANGE, engine="pytables", broadf_download=False,
            local_databases=str(path.parents[2]),
            inherit_dataframe=True, gpu_transfer=False, **options,
        )
    if public_api:
        with patch(
            "exojax.database.exomol._pyexocross_download.urlopen",
            side_effect=AssertionError("Offline check attempted a download"),
        ):
            actual_mdb = MdbExomol(
                str(path), NURANGE, backend="pyexocross", broadf_download=False,
                inherit_dataframe=True, gpu_transfer=False, **options,
            )
        snapshot = actual_mdb.to_snapshot()
        selected = actual_mdb.df[actual_mdb.df_load_mask]
        diagnostics = {
            name: selected[name].to_numpy()
            for name in ("i_upper", "i_lower", "A", "gup", "jlower", "jupper")
        }
        diagnostics["transition_rows"] = None  # The public API retains range-selected rows.
        diagnostics["Sref_exojax"] = line_strength_from_Einstein_coeff(
            actual_mdb.A, actual_mdb.gpp, actual_mdb.nu_lines,
            actual_mdb.elower, actual_mdb.QT_interp_numpy(296.0),
        )
    else:
        snapshot, diagnostics = load_snapshot(
            path, NURANGE, reference_c2=REFERENCE_C2, chunk_size=37, **options,
        )
        actual_mdb = direct_database(snapshot, diagnostics["A"])
    frame = reference.df[reference.df_load_mask]
    ref_order = np.lexsort((frame.i_lower.values, frame.i_upper.values))
    px_order = np.lexsort((diagnostics["i_lower"], diagnostics["i_upper"]))
    for name in ("i_upper", "i_lower"):
        np.testing.assert_array_equal(diagnostics[name][px_order], frame[name].values[ref_order])
    px_snapshot = sorted_snapshot(snapshot, px_order)
    ref_snapshot = sorted_snapshot(reference.to_snapshot(), ref_order)

    line_checks = {}
    for name in ("nu_lines", "elower", "line_strength_ref_original"):
        line_checks[name] = check(getattr(px_snapshot.lines, name), getattr(ref_snapshot.lines, name))
    for name in ("n_Texp", "alpha_ref"):
        line_checks[name] = check(getattr(px_snapshot, name), getattr(ref_snapshot, name))
    for name, attribute in (("A", "A"), ("gup", "gpp"), ("jlower", "jlower"), ("jupper", "jupper")):
        line_checks[name] = check(diagnostics[name][px_order], np.asarray(getattr(reference, attribute))[ref_order])
    for name in ("molmass", "T_gQT", "gQT"):
        line_checks[name] = check(getattr(px_snapshot.meta, name), getattr(ref_snapshot.meta, name))
    q_temperatures = np.array((296.0, 500.5, 1000.25, 1499.75))
    line_checks["Q_interpolated"] = check(
        np.interp(q_temperatures, snapshot.meta.T_gQT, snapshot.meta.gQT),
        np.asarray(reference.QT_interp(q_temperatures)),
    )

    # Report the original ExoJAX converter separately; never hide its c2 difference.
    native_strength_error = errors(
        diagnostics["Sref_exojax"][px_order], ref_snapshot.lines.line_strength_ref_original,
    )
    px_direct = OpaDirect(actual_mdb, nu_grid)
    ref_direct = OpaDirect(reference, nu_grid)
    premodit_options = dict(diffmode=2, manual_params=(100.0, 1000.0, 1200.0))
    px_premodit = (
        OpaPremodit(actual_mdb, nu_grid, **premodit_options)
        if public_api else OpaPremodit.from_snapshot(snapshot, nu_grid, **premodit_options)
    )
    ref_premodit = OpaPremodit(reference, nu_grid, **premodit_options)
    opacity_checks = {}
    for name, actual, expected in (
        ("LPF", px_direct, ref_direct), ("PreMODIT", px_premodit, ref_premodit),
    ):
        worst = {"max_relative": 0.0, "peak_relative": 0.0}
        for temperature in TEMPERATURES:
            for pressure in PRESSURES:
                metrics = check(
                    np.asarray(actual.xsvector(temperature, pressure)),
                    np.asarray(expected.xsvector(temperature, pressure)), spectrum=True,
                )
                worst = {key: max(worst[key], metrics[key]) for key in worst}
        opacity_checks[name] = worst
    return {
        "selected_lines": len(snapshot.lines.nu_lines),
        "raw_transition_rows": diagnostics["transition_rows"],
        "options": options, "line_checks": line_checks,
        "native_exojax_reference_strength_error": native_strength_error,
        "opacity_checks": opacity_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Write the JSON validation report here")
    parser.add_argument("--case", choices=[item[0] for item in CASES], help="Run only one case")
    parser.add_argument("--public-api", action="store_true", help="Validate MdbExomol(backend='pyexocross') instead of the stage-1 adapter")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = {
        "versions": {name: version(name) for name in ("pyexocross", "radis", "jax", "jaxlib", "numpy", "pandas", "pyarrow", "scipy", "zarr")},
        "python": sys.version.split()[0], "jax_backend": jax.default_backend(),
        "public_api": args.public_api,
        "nurange_cm-1": NURANGE, "grid_points": 2048,
        "temperatures_K": TEMPERATURES, "pressures_bar": PRESSURES,
        "reference_c2_cm_K": REFERENCE_C2, "exojax_hcperk_cm_K": hcperk,
        "spectrum_rtol": 1e-9, "spectrum_atol_relative_to_peak": 1e-12,
        "raw_file_sha256": {}, "cases": {},
    }
    nu_grid = np.geomspace(*NURANGE, report["grid_points"])
    with TemporaryDirectory(prefix="exojax-pyexocross-") as temporary:
        root = Path(temporary)
        paths = {}
        for molecule in ("CO", "H2O"):
            paths[molecule], report["raw_file_sha256"][molecule] = copy_raw_data(root, molecule)
        for name, molecule, options in CASES:
            if args.case and args.case != name:
                continue
            with redirect_stdout(sys.stderr):
                result = compare_case(paths[molecule], options, nu_grid, public_api=args.public_api)
            report["cases"][name] = result
            print(f"PASS {name}: {result['selected_lines']} lines; {result['opacity_checks']}", flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
