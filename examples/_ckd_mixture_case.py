"""Internal saved CO/H2O emission case shared by validation and retrieval examples.

The bundled ExoMol SAMPLE lists contain real, truncated molecular line lists.
They support an offline comparison of algorithms, not an absolute opacity claim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaCKD, OpaDirect
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.core import compute_ckd_from_xsmatrix, gauss_legendre_grid
from exojax.opacity.ckd.mixing import mix_ckd_rorr, validate_ckd_mixture_tables
from exojax.rt import ArtEmisPure
from exojax.rt.layeropacity import layer_optical_depth, layer_optical_depth_ckd
from exojax.rt.planck import piB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "benchmark"))
from diffgrid_nuts_storage import collect_provenance, sha256, write_json, write_npz

METHODS = ("lbl", "premixed", "rorr", "same_g")
SPECIES = ("CO", "H2O")
PARAMETER_ORDER = ("log_vmr_co", "log_vmr_h2o", "log_temperature_scale")
LINE_FIELDS = ("nu_lines", "elower", "logsij0", "A", "n_Texp", "alpha_ref", "T_gQT", "gQT")


def model_code_sha256():
    """Identify shared numerical code, independently of the calling CLI."""
    paths = sorted(path for path in (ROOT / "src" / "exojax").rglob("*.py")
                   if path.name != "ExoJAX_version.py") + [Path(__file__)]
    entries = [(str(path.relative_to(ROOT)), sha256(path)) for path in paths]
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def runtime_environment():
    return {
        "backend": jax.default_backend(),
        "devices": [device.device_kind for device in jax.devices()],
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpy": np.__version__,
        "x64": bool(jax.config.jax_enable_x64),
    }


class _SavedExomol:
    """Data-only adapter for Direct, retaining natural and pressure broadening."""

    dbtype = "exomol"

    def __init__(self, arrays, index):
        for name in LINE_FIELDS:
            setattr(self, name, jnp.asarray(arrays[f"species_{index}_{name}"]))
        self.molmass = float(arrays["molecular_masses"][index])

    def qr_interp(self, temperature, reference):
        return jnp.interp(temperature, self.T_gQT, self.gQT) / jnp.interp(
            reference, self.T_gQT, self.gQT
        )


def _spectral_grid(samples_per_band):
    # Equal midpoint samples give exactly the same top-hat observation bins.
    edges = np.linspace(4330.0, 4362.0, 9)
    grid = edges[:-1, None] + (np.arange(samples_per_band) + 0.5) * (
        edges[1] - edges[0]
    ) / samples_per_band
    return grid.ravel(), np.stack((edges[:-1], edges[1:]), axis=1)


def _art(arrays, nu_grid):
    return ArtEmisPure(
        pressure_top=float(arrays["pressure"][0]),
        pressure_btm=float(arrays["pressure"][-1]),
        nlayer=len(arrays["pressure"]), nu_grid=nu_grid, nstream=4,
    )


def _table(arrays, index, ng):
    ggrid, weights = gauss_legendre_grid(ng)
    opa = OpaCKD.load_only()
    opa.Ng, opa.ready = ng, True
    opa.nu_bands, opa.band_edges = arrays["nu_bands"], arrays["band_edges"]
    opa.ckd_info = CKDTableInfo(
        log_kggrid=arrays[f"species_{index}_log_k_{ng}"],
        ggrid=ggrid, weights=weights,
        T_grid=arrays["temperature_grid"], P_grid=arrays["pressure"],
        nu_bands=opa.nu_bands, band_edges=opa.band_edges,
    )
    return opa


def _context(metadata, arrays, output_dir=None):
    arrays = {name: jnp.asarray(value) for name, value in arrays.items()}
    context = {"metadata": metadata, "arrays": arrays, "output_dir": output_dir}
    for name in ("bounds", "truth", "observed", "sigma", "nu_bands"):
        if name in arrays:
            context[name] = arrays[name]
    context["model_code_sha256"] = model_code_sha256()
    return context


def verify_case(output_dir):
    """Reject changed inputs or numerical source before creating a forward model."""
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Case preparation did not complete.")
    for name in ("case.json", "arrays.npz"):
        if sha256(directory / name) != manifest["artifacts"][name]:
            raise ValueError(f"Case artifact hash mismatch: {name}")
    if model_code_sha256() != manifest["model_code_sha256"]:
        raise ValueError("Case numerical source hash mismatch; prepare a new case.")
    return manifest


def load_context(output_dir):
    directory = Path(output_dir).resolve()
    verify_case(directory)
    metadata = json.loads((directory / "case.json").read_text())
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        context = _context(metadata, dict(archive), directory)
    context["case_sha256"] = sha256(directory / "manifest.json")
    return context


def make_forward(context, method, *, ng=None, refined=False):
    """Return normalized band flux(theta), ordered by PARAMETER_ORDER.

    LBL and premixed compute Direct opacity online. RORR and same-g interpolate
    saved individual tables online. Premixed sorts the actual summed LBL opacity
    in each layer; it still assumes vertical rank correlation after compression.
    """
    if method not in METHODS or (refined and method != "lbl"):
        raise ValueError("Unknown method or non-LBL reference refinement.")
    arrays = context["arrays"]
    ng = context["metadata"]["config"]["ng"] if ng is None else ng
    bands = arrays["nu_bands"]
    ggrid, weights = gauss_legendre_grid(ng)
    art = _art(arrays, bands)
    if method in ("lbl", "premixed"):
        nu_grid = arrays["nu_refined" if refined else "nu_grid"]
        opas = [OpaDirect(_SavedExomol(arrays, index), nu_grid) for index in range(2)]
        lbl_art = _art(arrays, nu_grid)
        samples_per_band = len(nu_grid) // len(bands)
    else:
        opas = [_table(arrays, index, ng) for index in range(2)]
        validate_ckd_mixture_tables(opas)

    def forward(theta):
        vmr = jnp.exp(theta[:2])
        temperature = arrays["base_temperature"] * jnp.exp(theta[2])
        mmw = 2.3 * (1.0 - vmr.sum()) + vmr @ arrays["molecular_masses"]
        if method in ("lbl", "premixed"):
            dtau = sum(
                layer_optical_depth(
                    arrays["dpressure"], opa.xsmatrix(temperature, arrays["pressure"]),
                    vmr[index], mmw, 1000.0,
                )
                for index, opa in enumerate(opas)
            )
            if method == "lbl":
                flux = lbl_art.run(dtau, temperature).reshape(
                    len(bands), samples_per_band
                ).mean(axis=1)
            else:
                shape = (len(temperature), len(bands), samples_per_band)
                log_k = compute_ckd_from_xsmatrix(
                    dtau.reshape(shape).reshape(-1, samples_per_band), ggrid
                ).reshape(len(temperature), len(bands), ng)
                flux = art.run_ckd(jnp.exp(log_k).swapaxes(1, 2), temperature, weights, bands)
        else:
            species = jnp.stack([
                layer_optical_depth_ckd(
                    arrays["dpressure"], opa.xstensor_ckd(temperature, arrays["pressure"]),
                    vmr[index], mmw, 1000.0,
                )
                for index, opa in enumerate(opas)
            ])
            mixed = mix_ckd_rorr(species, weights) if method == "rorr" else species.sum(axis=0)
            flux = art.run_ckd(mixed, temperature, weights, bands)
        return flux / arrays["flux_normalization"]

    return forward


def prepare_case(output_dir, *, samples_per_band=1024, ng=16, temperature_nodes=21,
                 validation_points=5, seed=0):
    """Copy bundled real line data, compute tables, and freeze the mock observation."""
    if not jax.config.jax_enable_x64:
        raise ValueError("Enable JAX x64 before preparing or evaluating this case.")
    if min(samples_per_band, ng, temperature_nodes) < 2 or validation_points < 1:
        raise ValueError("Grid sizes must be >= 2 and validation_points >= 1.")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(samples_per_band=samples_per_band, ng=ng,
                  temperature_nodes=temperature_nodes, validation_points=validation_points,
                  seed=seed, ng_values=sorted(set((max(2, ng // 2), ng, 2 * ng))))
    metadata = {
        "schema_version": 1, "config": config, "species_order": list(SPECIES),
        "parameter_order": list(PARAMETER_ORDER),
        "budgets": {"max_error_in_noise": 0.01, "max_q": 0.1,
                    "gradient_tolerance": 1e-3, "reference_fraction": 0.1,
                    "steps": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
        "physics": {"rt": "ArtEmisPure ibased, four streams, four layers",
                    "gravity_cm_s2": 1000.0, "background_mmw": 2.3,
                    "broadening": "bundled ExoMol H2 parameters, plus natural width",
                    "observation": "eight equal top-hat bins, normalized flux, sigma=0.01",
                    "continuum": "none", "profile": "Direct Voigt, no additional line cutoff"},
        "limitations": ["Truncated real SAMPLE line lists; missing lines and wings outside the sample are not assessed.",
                        "CKD uses Planck values at band centers and assumes vertical rank correlation.",
                        "Fixed H2 broadening does not respond to the retrieved composition.",
                        "Local validation points do not establish accuracy across the full prior."],
    }
    # Persist budgets and conditions before computing any spectrum or accuracy.
    write_json(directory / "case.json", metadata)
    nu_grid, band_edges = _spectral_grid(samples_per_band)
    nu_refined, _ = _spectral_grid(2 * samples_per_band)
    art = ArtEmisPure(pressure_top=0.03, pressure_btm=3.0, nlayer=4, nu_grid=nu_grid, nstream=4)
    bounds = np.array([[np.log(1e-4), np.log(3e-2)],
                       [np.log(1e-4), np.log(3e-2)], [np.log(0.85), np.log(1.15)]])
    truth = np.array([np.log(0.003), np.log(0.005), np.log(1.013)])
    bands = band_edges.mean(axis=1)
    arrays = dict(nu_grid=nu_grid, nu_refined=nu_refined, band_edges=band_edges,
                  nu_bands=bands, pressure=np.asarray(art.pressure), dpressure=np.asarray(art.dParr),
                  base_temperature=np.linspace(650.0, 1150.0, 4),
                  temperature_grid=np.linspace(500.0, 1400.0, temperature_nodes),
                  truth=truth, bounds=bounds, sigma=np.full(len(bands), 0.01),
                  flux_normalization=np.asarray(piB(1000.0, jnp.asarray(bands))))
    rng = np.random.default_rng(seed)
    points = bounds[:, 0] + (0.15 + 0.7 * rng.random((validation_points, 3))) * np.diff(bounds, axis=1)[:, 0]
    points[0] = truth
    arrays["validation_points"] = points
    source_paths, molecule_metadata, masses = [], [], []
    from exojax.database.exomol.api import MdbExomol
    from exojax.test.data import get_testdata_filename
    paths = ("CO/12C-16O/SAMPLE", "H2O/1H2-16O/SAMPLE")
    for index, path in enumerate(paths):
        source = Path(get_testdata_filename(path))
        source_paths.extend(sorted(file for file in source.iterdir() if file.is_file()))
        with tempfile.TemporaryDirectory(prefix="line-input-", dir=directory) as temporary:
            target = Path(temporary) / path
            shutil.copytree(source, target)
            mdb = MdbExomol(str(target), np.array([4329.0, 4363.0]),
                           crit=0.0, broadf_download=False, gpu_transfer=True)
            masses.append(float(mdb.molmass))
            for name in LINE_FIELDS:
                arrays[f"species_{index}_{name}"] = np.asarray(getattr(mdb, name))
            molecule_metadata.append({"species": SPECIES[index], "lines": len(mdb.nu_lines),
                                      "source": str(source.relative_to(ROOT)),
                                      "original_line_list": "Li2015" if index == 0 else "POKAZATEL"})
    arrays["molecular_masses"] = np.asarray(masses)
    metadata["molecules"] = molecule_metadata
    started = time.perf_counter()
    for index in range(2):
        opa = OpaDirect(_SavedExomol(arrays, index), nu_grid)
        evaluate = jax.jit(opa.xsmatrix)
        spectra = []
        for temperature in arrays["temperature_grid"]:
            spectra.append(np.asarray(evaluate(jnp.full(4, temperature), arrays["pressure"])))
        spectra = jnp.asarray(np.stack(spectra)).reshape(-1, samples_per_band)
        for count in config["ng_values"]:
            ggrid, _ = gauss_legendre_grid(count)
            log_k = compute_ckd_from_xsmatrix(spectra, ggrid)
            arrays[f"species_{index}_log_k_{count}"] = np.asarray(log_k).reshape(
                temperature_nodes, 4, len(bands), count
            ).swapaxes(2, 3)
    metadata["precompute_seconds_including_compile"] = time.perf_counter() - started
    context = _context(metadata, arrays)
    reference = np.asarray(jax.jit(make_forward(context, "lbl", refined=True))(jnp.asarray(truth)))
    arrays["noiseless_observed"] = reference
    arrays["observed"] = reference + np.random.default_rng(seed + 1).normal(size=len(bands)) * arrays["sigma"]
    metadata["environment"] = runtime_environment()
    metadata["provenance"] = collect_provenance(
        ROOT, [__file__, ROOT / "tests" / "benchmark" / "diffgrid_nuts_storage.py", *source_paths], config
    )
    write_npz(directory / "arrays.npz", **arrays)
    write_json(directory / "case.json", metadata)
    write_json(directory / "manifest.json", {
        "schema_version": 1, "status": "completed", "model_code_sha256": model_code_sha256(),
        "artifacts": {name: sha256(directory / name) for name in ("case.json", "arrays.npz")},
    })
    return load_context(directory)
