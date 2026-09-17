"""Build the two small, immutable opacity datasets used by the starter tutorial.

Run from a checkout with ExoJAX installed. Source databases are build inputs;
only the generated dataset directories are uploaded to the documentation server.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
H2O_URL = (
    "https://www.exomol.com/db/H2O/1H2-16O/POKAZATEL/"
    "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
)
CO_URL = "https://www.exomol.com/db/CO/12C-16O/Li2015/"
H2O_SOURCE_SHA256 = "adf7aef2769ce4c652130d682e36d302e789df78da6ac96c2f55a5c204e47757"


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(name=path.name, size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def crop_h2o(source, destination):
    """Select native ExoMolOP nodes without rebinning or changing k coefficients."""
    import h5py

    with h5py.File(source, "r") as original:
        centers = original["bin_centers"][:]
        temperatures = original["t"][:]
        pressures = original["p"][:]
        if not all(np.all(np.diff(x) > 0) for x in (centers, temperatures, pressures)):
            raise ValueError("Expected ascending native ExoMolOP coordinates")
        indices = np.flatnonzero((centers >= 2000.0) & (centers <= 10000.0))
        if indices.size < 2 or centers[0] > 2000.0 or centers[-1] < 10000.0:
            raise ValueError("The source must cover the complete 1--5 micron interval")
        # Keep one guard band at either end for subsequent band interpolation.
        band = slice(max(0, indices[0] - 1), min(centers.size, indices[-1] + 2))
        selections = []
        for grid, low, high in ((temperatures, 500.0, 1500.0), (pressures, 1e-5, 10.0)):
            start = max(0, np.searchsorted(grid, low, side="right") - 1)
            stop = min(grid.size, np.searchsorted(grid, high, side="left") + 1)
            if grid[start] > low or grid[stop - 1] < high:
                raise ValueError("The source does not cover the tutorial T/P range")
            selections.append(slice(start, stop))
        temp, press = selections
        values = original["kcoeff"][press, temp, band, :]
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Source k coefficients must be finite and nonnegative")
        if original["kcoeff"].attrs.get("units") != "cm^2/molecule":
            raise ValueError("Expected ExoMolOP cross sections in cm^2/molecule")
        if original["p"].attrs.get("units") != "bar":
            raise ValueError("Expected ExoMolOP pressures in bar")
        arrays = dict(kcoeff=values, bin_centers=centers[band],
                      t=temperatures[temp], p=pressures[press])
        for key in ("samples", "weights", "mol_mass"):
            arrays[key] = original[key][:]
        if "bin_edges" in original:
            arrays["bin_edges"] = original["bin_edges"][band.start:band.stop + 1]
        with h5py.File(destination, "x") as output:
            for key, value in arrays.items():
                options = dict(compression="gzip", shuffle=True) if key == "kcoeff" else {}
                dataset = output.create_dataset(key, data=value, **options)
                if key in original:
                    dataset.attrs.update(original[key].attrs)
            for key in ("DOI", "Date_ID", "key_iso_ll", "method", "ngauss"):
                if key in original:
                    original.copy(key, output)
            output.create_dataset("mol_name", data=np.array([b"H2O"]))
            edges = arrays.get("bin_edges", arrays["bin_centers"])
            output.create_dataset("wnrange", data=edges[[0, -1]])
            output.create_dataset("wlrange", data=1e4 / edges[[-1, 0]])
            output.attrs.update(original.attrs)
            output.attrs["modification"] = "Native T/P/wavenumber subset for ExoJAX; no rebinning."
    return arrays


def build_h2o(source, directory):
    from exojax.opacity import OpaCKD

    source_record = file_record(source)
    if source_record["sha256"] != H2O_SOURCE_SHA256:
        raise ValueError("This v1 recipe requires the pinned POKAZATEL ExoMolOP source SHA256")
    arrays = crop_h2o(source, directory / "h2o_ckd.h5")
    opa = OpaCKD.from_external("exomolop", directory / "h2o_ckd.h5")
    expected = np.transpose(arrays["kcoeff"], (1, 0, 3, 2))
    expected = np.maximum(expected, np.finfo(expected.dtype).tiny)
    np.testing.assert_allclose(np.exp(opa.ckd_info.log_kggrid), expected, rtol=1e-12, atol=0)
    write_json(directory / "validation.json", {
        "native_coefficient_roundtrip_rtol": 1e-12,
        "native_coefficient_roundtrip_passed": True,
        "scope": "Subset/load preservation; upstream ExoMolOP accuracy is inherited."
    })
    return dict(
        molecule="H2O", isotopologue="1H2-16O", line_list="POKAZATEL",
        method="ExoMolOP CKD", molmass=float(opa.molmass),
        source={"url": H2O_URL, **source_record},
        citations=["https://doi.org/10.1093/mnras/sty1877",
                   "https://doi.org/10.1051/0004-6361/202038350"],
        modifications="Native node subset, lossless compression, explicit molecule name; no rebinning.",
        broadening="Upstream ExoMolOP H2/He solar mixture; see Chubb et al. (2021).",
        line_wings="Upstream ExoMolOP prescription, 500 Voigt half-widths.",
        wavelength_range_um=[1.0, 5.0], resolving_power=1000,
        temperature_range_K=[500.0, 1500.0], pressure_range_bar=[1e-5, 10.0],
        table_shape_P_T_band_g=list(arrays["kcoeff"].shape),
        dtype=str(arrays["kcoeff"].dtype),
    )


def build_co(source, directory):
    import jax
    import jax.numpy as jnp
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaDiffgrid, OpaPremodit, saveopa
    from exojax.opacity.diffgrid.diagnostics import (
        compare_diffgrid_with_teacher, diffgrid_interval_midpoint_temperatures,
    )
    from exojax.rt import ArtEmisPure
    from exojax.utils.grids import wavenumber_grid

    if source.parts[-3:] != ("CO", "12C-16O", "Li2015"):
        raise ValueError("Expected a CO/12C-16O/Li2015 source directory")
    if not (source / "12C-16O__H2.broad").is_file():
        raise ValueError("The CO starter recipe requires the ExoMol H2 broadening file")
    nu, _, resolution = wavenumber_grid(22920.0, 23000.0, 3500, unit="AA", xsmode="diffgrid")
    art = ArtEmisPure(nu_grid=nu, pressure_top=0.1, pressure_btm=10.0, nlayer=16,
                     rtsolver="fbased2st", nstream=2)
    mdb = MdbExomol(source, nurange=nu, gpu_transfer=False, bkgdatm="H2",
                    broadf_download=False, engine="pytables")
    teacher = OpaPremodit(mdb, nu, diffmode=2, auto_trange=(500.0, 1500.0),
                         broadening_resolution={"mode": "single", "value": None})
    opa = OpaDiffgrid(teacher, 1.0 / np.linspace(1/1500.0, 1/500.0, 17), art.pressure)
    source_files = sorted(path for path in source.iterdir()
                          if path.name.endswith((".def", ".pf", ".states.bz2", ".trans.bz2"))
                          or path.name == "12C-16O__H2.broad")
    meta = dict(
        molecule="CO", isotopologue="12C-16O", line_list="Li2015",
        method="PreMODIT / DiffGrid", molmass=float(opa.molmass),
        source={"url": CO_URL, "files": [
            dict(url=(CO_URL.rsplit("Li2015/", 1)[0] if path.suffix == ".broad" else CO_URL)
                 + path.name, **file_record(path)) for path in source_files]},
        citations=["https://doi.org/10.1088/0067-0049/216/1/15"],
        modifications="PreMODIT cross sections and inverse-temperature derivatives tabulated as DiffGrid.",
        broadening="H2, ExoMol broadening file, PreMODIT single representative broadening mode.",
        premodit_settings={"diffmode": 2, "auto_trange": [500.0, 1500.0],
                           "broadening_resolution": {"mode": "single", "value": None},
                           "cutwing": 1.0, "nstitch": 1, "line_strength_cutoff": 0.0},
        spectral_scope="Only lines inside the tabulated interval; outside-window line wings are omitted.",
        wavelength_range_um=[2.292, 2.300], grid_resolving_power=float(resolution),
        temperature_range_K=[500.0, 1500.0], pressure_range_bar=[0.1, 10.0],
        table_shape_layer_T_nu=list(opa.diffgrid_info.log_cross_section_grid.shape),
        dtype="float64",
    )
    diagnostics = []
    for temperature in diffgrid_interval_midpoint_temperatures(opa):
        result = compare_diffgrid_with_teacher(opa, teacher, jnp.full((16,), temperature))
        diagnostics.append(dict(temperature_K=float(temperature),
                                p99_absolute_log_error=result.absolute_log_cross_section_error_quantiles[0],
                                maximum_absolute_log_error=result.maximum_absolute_log_cross_section_error))

    def emission(calculator, temperature):
        profile = temperature * art.pressure ** 0.08
        xs = calculator.xsmatrix(profile) if calculator is opa else calculator.xsmatrix(profile, art.pressure)
        dtau = art.opacity_profile_xs(xs, jnp.full((16,), 1e-3), opa.molmass, 1e5)
        return art.run(dtau, profile)

    flux_checks = []
    for temperature in (900.0, 1000.0, 1100.0):
        value, derivative = jax.jvp(lambda t: emission(opa, t), (temperature,), (1.0,))
        reference, reference_derivative = jax.jvp(lambda t: emission(teacher, t), (temperature,), (1.0,))
        flux_error = float(jnp.max(jnp.abs(value-reference)) / jnp.max(jnp.abs(reference)))
        derivative_error = float(jnp.max(jnp.abs(derivative-reference_derivative)) /
                                 jnp.max(jnp.abs(reference_derivative)))
        if not np.isfinite([flux_error, derivative_error]).all() or flux_error > 1e-3 or derivative_error > 1e-2:
            raise ValueError(f"DiffGrid starter validation failed: {flux_error=}, {derivative_error=}")
        flux_checks.append(dict(T0_K=temperature, maximum_flux_error_over_peak=flux_error,
                                maximum_derivative_error_over_peak=derivative_error))
    saveopa(opa, directory / "co_diffgrid.npz", format="npz", extra_meta=meta)
    restored = OpaDiffgrid.from_saved_opa(directory / "co_diffgrid.npz")
    np.testing.assert_array_equal(restored.xsmatrix(jnp.full((16,), 1000.0)),
                                  opa.xsmatrix(jnp.full((16,), 1000.0)))
    write_json(directory / "validation.json", {
        "reference": "PreMODIT teacher with the same fixed line/broadening settings",
        "scope": "DiffGrid interpolation accuracy; not a test of line-wing or broadening approximations.",
        "inverse_temperature_midpoints": diagnostics,
        "emission_and_temperature_derivative": flux_checks,
        "thresholds": {"flux_error_over_peak": 1e-3, "derivative_error_over_peak": 1e-2},
        "saved_table_roundtrip_passed": True,
    })
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("h2o-ckd-v1", "co-diffgrid-v1"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h2o-source", type=Path)
    parser.add_argument("--co-source", type=Path)
    args = parser.parse_args()
    source = args.h2o_source if args.dataset == "h2o-ckd-v1" else args.co_source
    if source is None or not source.exists():
        parser.error("Supply the existing --h2o-source table or --co-source line-list directory")
    import jax
    jax.config.update("jax_enable_x64", True)
    import exojax

    destination = args.output_dir / args.dataset
    if destination.exists():
        parser.error(f"Dataset already exists: {destination}; published IDs are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = build_h2o if args.dataset == "h2o-ckd-v1" else build_co
    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix=".starter-") as temporary:
        directory = Path(temporary) / args.dataset
        directory.mkdir()
        metadata = builder(source.resolve(), directory)
        (directory / "LICENSE.txt").write_text(
            "ExoMol-derived opacity data, modified by ExoJAX contributors.\n"
            f"Licensed under Creative Commons Attribution-ShareAlike 4.0 International: {LICENSE_URL}\n"
            "Source data licence: https://exomol.com/data/licence/\n"
            "See manifest.json for source attribution, modifications, and scientific references.\n",
            encoding="utf-8",
        )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip())
        manifest = dict(schema_version=1, dataset=args.dataset,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        license="CC-BY-SA-4.0", license_url=LICENSE_URL,
                        producer={"exojax_version": exojax.__version__, "jax_version": jax.__version__,
                                  "numpy_version": np.__version__,
                                  "radis_version": version("radis") if args.dataset == "co-diffgrid-v1" else None,
                                  "git_commit": commit, "git_dirty": dirty,
                                  "build_script": file_record(Path(__file__))},
                        **metadata,
                        files=[file_record(path) for path in sorted(directory.iterdir())])
        write_json(directory / "manifest.json", manifest)
        directory.rename(destination)
    print(f"Prepared {destination}: {sum(item['size_bytes'] for item in manifest['files']) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
