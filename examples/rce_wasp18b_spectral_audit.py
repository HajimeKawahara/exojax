"""Audit a saved converged WASP-18b spectrum without rerunning radiative transfer.

Gaussian response trials and blackbody integration checks do not establish
RCE convergence under changed physics, opacity grids, or spectral endpoints.
"""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import jax
import numpy as np
from scipy.integrate import quad

from _rce_broadband import band_planck_flux, make_band_quadrature
from _rce_observation import load_emission_candidate, pixel_flux_ratio
from _rce_wasp18b import SIGMA_SB, response_samples, spectral_error
from _rce_wasp18b_assets import (
    AssetManifest, SCHEMA_VERSION, _sha256, load_stellar_spectrum,
)
from exojax.rt.planck import fac_planck
from exojax.utils.constants import hcperk


def verified_inputs(run_path, manifest_path, observation_path, spectrum_path=None):
    """Verify the run/spectrum identity and reread only the required stellar assets.

    Opacity hashes recorded by the successful forward run must match the pinned
    manifest. Their large coefficient files are not reread by this audit.
    """
    run_path, manifest_path = Path(run_path).resolve(), Path(manifest_path).resolve()
    run = json.loads(run_path.read_text())
    if (run.get("schema_version") != "exojax.rce-wasp18b-run/1"
            or run.get("success") is not True or run.get("mode") != "rce"
            or run.get("state", {}).get("converged") is not True
            or run.get("state", {}).get("status") != 0):
        raise ValueError("Require a successful converged RCE run.")
    config = json.loads(manifest_path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported input manifest schema.")
    hashes = {key: item["sha256"].lower() for key, item in config["files"].items()}
    hashes["__manifest__"] = _sha256(manifest_path)
    if hashes != run.get("asset_sha256") or config != run.get("configuration"):
        raise ValueError("Run manifest or asset pins do not match.")
    observation = load_emission_candidate(observation_path)
    if (observation.sha256 != run.get("observation_sha256")
            or not np.array_equal(observation.row_indices, run.get("row_indices"))
            or not np.array_equal(observation.uncertainty, run.get("uncertainty"))):
        raise ValueError("Run observation hashes, rows, or uncertainties do not match.")
    saved = run["resolved_spectrum_file"]
    if saved.get("units") != "cm^-1; erg s^-1 cm^-2 (cm^-1)^-1; K":
        raise ValueError("Unexpected resolved spectrum units.")
    path = Path(spectrum_path or saved["path"]).resolve()
    if _sha256(path) != saved["sha256"]:
        raise ValueError("Resolved spectrum SHA256 mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        spectrum = {key: np.asarray(archive[key]) for key in (
            "nu_cm1", "planet_upward_flux_per_cm1", "temperature_K",
            "bottom_temperature_K", "parameters",
        )}
    for key in ("temperature_K", "bottom_temperature_K", "parameters"):
        if not np.array_equal(spectrum[key], run[key]):
            raise ValueError(f"Resolved spectrum {key} differs from the run.")
    for key, state_key in (("temperature_K", "temperature"),
                           ("bottom_temperature_K", "bottom_temperature")):
        if not np.array_equal(spectrum[key], run["state"][state_key]):
            raise ValueError(f"Resolved spectrum {key} differs from the converged state.")
    nu, flux = spectrum["nu_cm1"], spectrum["planet_upward_flux_per_cm1"]
    if (nu.ndim != 1 or nu.size < 2 or flux.shape != nu.shape
            or not np.all(np.isfinite(nu)) or np.any(nu <= 0) or np.any(np.diff(nu) <= 0)
            or not np.all(np.isfinite(flux)) or np.any(flux < 0)):
        raise ValueError("Require an ordered positive grid and finite nonnegative flux.")
    paths = {}
    spec = config["stellar"]
    for field in ("asset", "wavelength_asset", "flux_asset"):
        if field in spec:
            key = spec[field]
            paths[key] = (manifest_path.parent / Path(config["files"][key]["path"]).expanduser()).resolve()
            if _sha256(paths[key]) != hashes[key]:
                raise ValueError(f"Stellar asset SHA256 mismatch: {key}")
    manifest = AssetManifest(manifest_path, config, paths, hashes)
    stellar = load_stellar_spectrum(manifest)
    identity = {
        "run_sha256": _sha256(run_path), "manifest_sha256": hashes["__manifest__"],
        "spectrum_sha256": saved["sha256"], "observation_sha256": observation.sha256,
        "stellar_asset_sha256": {key: hashes[key] for key in paths},
        "opacity_verification": "Pins matched to the successful run; coefficient files not reread",
    }
    return run, observation, stellar, spectrum, identity


def response_audit(run, observation, stellar, spectrum, resolutions=(500., 700., 1000.),
                   wavelength_fwhm_nm=()):
    """Reapply the exact calibrated-pixel P1 operator to both planet and star."""
    selected = np.any(observation.W, axis=0)
    matrix = observation.W[:, selected]
    pixel_nu = 1e4 / observation.input_wavelength_um[selected]
    nu, flux = spectrum["nu_cm1"], spectrum["planet_upward_flux_per_cm1"]
    if pixel_nu.min() < nu[0] or pixel_nu.max() > nu[-1]:
        raise ValueError("Resolved spectrum does not cover the selected observation pixels.")
    stellar.validate_coverage(pixel_nu)
    area = float(observation.recipe["radius_ratio"]) ** 2

    def prediction(resolution):
        planet = response_samples(nu, flux, pixel_nu, resolution)
        star = response_samples(stellar.nu_cm1, stellar.surface_flux_per_cm1,
                                pixel_nu, resolution)
        if np.any(np.asarray(matrix @ star) <= 0):
            raise ValueError("Every bin requires a positive stellar flux sum.")
        return np.asarray(pixel_flux_ratio(matrix, planet, star, area))

    centers = prediction(None)
    recomputed = centers if run["response_resolution"] is None else prediction(run["response_resolution"])
    # CPU/GPU arithmetic may differ at roundoff; this is an identity check,
    # deliberately much tighter than the physical sensitivity budget.
    if not np.allclose(recomputed, run["prediction"], rtol=2e-12, atol=1e-15):
        raise ValueError("Saved spectrum does not reproduce the run prediction.")
    trials = []
    for resolution in resolutions:
        values = prediction(resolution)
        trials.append({"resolving_power": float(resolution), "prediction": values.tolist(),
                       "vs_exact_centers": spectral_error(values, centers, observation.uncertainty)})
    pairwise = [{"resolving_powers": [first["resolving_power"], second["resolving_power"]],
                 "difference": spectral_error(first["prediction"], second["prediction"],
                                              observation.uncertainty)}
                for first, second in combinations(trials, 2)]
    varying = []
    if len(wavelength_fwhm_nm):
        reference = prediction(700.)
        for width in wavelength_fwhm_nm:
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Wavelength FWHM must be finite and positive.")
            resolution = (1e4 / pixel_nu) / (width * 1e-3)
            values = prediction(resolution)
            varying.append({"wavelength_fwhm_nm": float(width),
                            "resolving_power_range": [float(resolution.min()), float(resolution.max())],
                            "prediction": values.tolist(),
                            "vs_constant_R700": spectral_error(values, reference, observation.uncertainty),
                            "vs_exact_centers": spectral_error(values, centers, observation.uncertainty)})
    return {
        "scope": "Fixed converged state; assumed Gaussian responses, not a measured SOSS LSF",
        "sigma_budget": {"max_sigma_strictly_below": .1, "weighted_norm_strictly_below": .3,
                         "weighted_norm_definition": "Euclidean norm of delta/sigma, not RMS"},
        "prediction_exact_centers": centers.tolist(),
        "saved_prediction_reproduction": spectral_error(recomputed, run["prediction"], observation.uncertainty),
        "gaussian_trials": trials,
        "gaussian_pairwise": pairwise,
        "wavelength_width_assumption": "R(lambda)=lambda/FWHM; local width of a Gaussian in wavenumber, not a calibrated wavelength LSF",
        "wavelength_width_trials": varying,
    }


def blackbody_audit(temperatures=(1500., 2500., 3500., 4500.),
                    ranges=((200., 50000.), (100., 50000.), (200., 100000.), (20., 100000.)),
                    orders=(1, 2, 4, 8, 16), band_resolution=1000.):
    """Compare shared band-Planck quadrature against an independent integral.

    Logarithmic bands follow the preparation grid rule. Dimensionless Planck
    integrals independently separate omitted tails from quadrature errors.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    if (temperatures.ndim != 1 or temperatures.size == 0
            or not np.all(np.isfinite(temperatures) & (temperatures > 0))
            or not np.isfinite(band_resolution) or band_resolution <= 0):
        raise ValueError("Use positive finite temperatures and band resolution.")
    total = fac_planck * (temperatures / hcperk)**4 * np.pi**4 / 15
    rows = []

    def kernel(x):
        return 0. if x == 0 else x**3 * np.exp(-x) / (-np.expm1(-x))

    for lower, upper in ranges:
        if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
            raise ValueError("Use ordered positive finite wavenumber endpoints.")
        count = math.ceil(math.log(upper / lower) * band_resolution)
        points = lower * np.exp(math.log(upper / lower) / count * np.arange(count + 1))
        points[0], points[-1] = lower, upper
        edges, widths = np.column_stack((points[:-1], points[1:])), np.diff(points)
        low_tail = np.array([quad(kernel, 0., hcperk * lower / t,
                                  epsabs=1e-12, epsrel=1e-12)[0] for t in temperatures]) * 15 / np.pi**4
        high_tail = np.array([quad(kernel, hcperk * upper / t, np.inf,
                                   epsabs=1e-12, epsrel=1e-12)[0] for t in temperatures]) * 15 / np.pi**4
        finite = total * (1 - low_tail - high_tail)
        quadrature = []
        for order in orders:
            nodes, weights = make_band_quadrature(edges, order)
            integral = np.asarray(band_planck_flux(temperatures, nodes, weights)) @ widths
            quadrature.append({"order": int(order), "flux_cgs": integral.tolist(),
                               "relative_error_vs_finite_integral": (integral / finite - 1).tolist(),
                               "relative_error_vs_stefan_boltzmann":
                                   (integral / (SIGMA_SB * temperatures**4) - 1).tolist()})
        rows.append({"nu_range_cm1": [lower, upper], "number_of_bands": count,
                     "omitted_low_nu_fraction": low_tail.tolist(),
                     "omitted_high_nu_fraction": high_tail.tolist(),
                     "quadrature": quadrature})
    return {
        "scope": "Analytic blackbody integration only; not an RCE endpoint or opacity convergence test",
        "temperature_K": temperatures.tolist(), "band_resolution": band_resolution,
        "planck_constants_relative_to_stefan_boltzmann":
            (total / (SIGMA_SB * temperatures**4) - 1).tolist(),
        "ranges": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, help="Optional relocated copy with the same pinned SHA256")
    parser.add_argument("--response-resolutions", type=float, nargs="+", default=[500., 700., 1000.],
                        help="Constant Gaussian resolving powers for pairwise sensitivity trials")
    parser.add_argument("--wavelength-fwhm-nm", type=float, nargs="+", default=[],
                        help="Optional local wavelength FWHM trials, using per-pixel R=lambda/FWHM")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    run, observation, stellar, spectrum, identity = verified_inputs(
        args.run, args.manifest, args.observation, args.spectrum)
    report = {"schema_version": "exojax.rce-wasp18b-spectral-audit/1", "success": True,
              "p7_physical_acceptance": "not_certified", "identity": identity,
              "response": response_audit(run, observation, stellar, spectrum, args.response_resolutions,
                                         args.wavelength_fwhm_nm),
              "blackbody": blackbody_audit()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
