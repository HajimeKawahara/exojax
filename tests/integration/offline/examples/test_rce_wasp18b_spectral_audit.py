"""Check saved-run identity, exact P1 responses, and independent Planck tails."""

import importlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.constants import c, h, k_B
from scipy.integrate import trapezoid

from tests.integration.offline.examples.test_rce_observation import product, write_product


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("rce_wasp18b_spectral_audit")
    jax.config.update("jax_enable_x64", previous)


def saved_run(audit, product):
    observation_path = write_product(*product)
    observation = audit.load_emission_candidate(observation_path)
    directory = observation_path.parent
    nu = np.linspace(2000., 11000., 9001)
    # An isolated line at a selected pixel makes the response test nontrivial.
    planet = 5. * (1 + 8 * np.exp(-((nu - 5000.) / 2.)**2))
    star = 1000. + .01 * nu
    stellar_path = directory / "stellar.npz"
    np.savez(stellar_path, nu_cm1=nu, surface_flux_per_cm1=star)
    config = {
        "schema_version": audit.SCHEMA_VERSION,
        "files": {"star": {"path": stellar_path.name, "sha256": audit._sha256(stellar_path),
                            "source": "synthetic test spectrum"}},
        "stellar": {"format": "npz", "asset": "star", "version": "test/1",
                    "flux_unit": "erg s-1 cm-2 (cm-1)-1"},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(config))
    spectrum_path = directory / "spectrum.npz"
    temperature, bottom, parameters = np.array([2300., 2700.]), 3000., np.array([0., .6, -.5])
    np.savez(spectrum_path, nu_cm1=nu, planet_upward_flux_per_cm1=planet,
             temperature_K=temperature, bottom_temperature_K=bottom, parameters=parameters)
    selected = np.any(observation.W, axis=0)
    pixel_nu = 1e4 / observation.input_wavelength_um[selected]
    matrix = observation.W[:, selected]
    prediction = .01 * (matrix @ np.interp(pixel_nu, nu, planet)) / (matrix @ np.interp(pixel_nu, nu, star))
    run = {
        "schema_version": "exojax.rce-wasp18b-run/1", "success": True, "mode": "rce",
        "state": {"converged": True, "status": 0, "temperature": temperature.tolist(),
                  "bottom_temperature": bottom},
        "configuration": config,
        "asset_sha256": {"star": audit._sha256(stellar_path), "__manifest__": audit._sha256(manifest_path)},
        "observation_sha256": observation.sha256, "row_indices": observation.row_indices.tolist(),
        "uncertainty": observation.uncertainty.tolist(), "temperature_K": temperature.tolist(),
        "bottom_temperature_K": bottom, "parameters": parameters.tolist(),
        "prediction": prediction.tolist(), "response_resolution": None,
        "resolved_spectrum_file": {"path": str(spectrum_path), "sha256": audit._sha256(spectrum_path),
                                   "units": "cm^-1; erg s^-1 cm^-2 (cm^-1)^-1; K"},
    }
    run_path = directory / "run.json"
    run_path.write_text(json.dumps(run))
    return run_path, manifest_path, observation_path, spectrum_path


def test_verified_spectrum_reproduces_p1_and_resolves_line_response(audit, product):
    paths = saved_run(audit, product)
    run, observation, stellar, spectrum, identity = audit.verified_inputs(*paths)
    result = audit.response_audit(run, observation, stellar, spectrum)
    np.testing.assert_allclose(result["prediction_exact_centers"], run["prediction"], rtol=3e-16)
    assert result["saved_prediction_reproduction"]["max_sigma"] < 1e-12
    assert identity["spectrum_sha256"] == audit._sha256(paths[-1])
    assert [r["resolving_power"] for r in result["gaussian_trials"]] == [500., 700., 1000.]
    trials = result["gaussian_trials"]
    assert trials[0]["prediction"][1] < trials[1]["prediction"][1] < trials[2]["prediction"][1]
    assert trials[0]["vs_exact_centers"]["max_sigma"] > 0
    pairwise = result["gaussian_pairwise"]
    assert [row["resolving_powers"] for row in pairwise] == [[500., 700.], [500., 1000.], [700., 1000.]]
    difference = (np.asarray(trials[0]["prediction"]) - trials[2]["prediction"]) / observation.uncertainty
    np.testing.assert_allclose(pairwise[1]["difference"]["weighted_norm"], np.linalg.norm(difference))
    run["prediction"][0] *= 1.01
    with pytest.raises(ValueError, match="reproduce"):
        audit.response_audit(run, observation, stellar, spectrum)


@pytest.mark.parametrize("kind", ["failed", "fixed", "manifest", "observation", "spectrum", "state", "star"])
def test_inconsistent_saved_inputs_are_rejected(audit, product, kind):
    paths = saved_run(audit, product)
    run = json.loads(paths[0].read_text())
    if kind == "failed":
        run["state"]["converged"] = False
    elif kind == "fixed":
        run["mode"] = "fixed_temperature"
    elif kind == "manifest":
        run["asset_sha256"]["__manifest__"] = "0" * 64
    elif kind == "observation":
        run["uncertainty"][0] *= 2
    elif kind == "spectrum":
        paths[-1].write_bytes(paths[-1].read_bytes() + b"changed")
    elif kind == "state":
        run["state"]["temperature"][0] += 1
    else:
        star = paths[0].parent / "stellar.npz"
        star.write_bytes(star.read_bytes() + b"changed")
    paths[0].write_text(json.dumps(run))
    with pytest.raises(ValueError):
        audit.verified_inputs(*paths)


def test_blackbody_units_tail_budget_and_order_are_independent(audit):
    # Broad bands deliberately make a one-point quadrature inaccurate.
    report = audit.blackbody_audit(temperatures=[2500.],
                                   ranges=[(200., 50000.), (20., 100000.)],
                                   orders=(1, 16), band_resolution=2.)
    baseline, extended = report["ranges"]
    first, accurate = baseline["quadrature"]
    assert abs(first["relative_error_vs_finite_integral"][0]) > 1e-4
    assert abs(accurate["relative_error_vs_finite_integral"][0]) < 1e-12
    assert extended["omitted_low_nu_fraction"][0] < baseline["omitted_low_nu_fraction"][0] / 900
    assert extended["omitted_high_nu_fraction"][0] < baseline["omitted_high_nu_fraction"][0]
    # Independent astropy radiance integral verifies pi, c and per-cm^-1 units.
    nu = np.geomspace(200., 50000., 100001) / u.cm
    # Match the library's rounded hc/k constant, whose bolometric offset from
    # modern SI constants is explicitly reported separately by the audit.
    effective_temperature = 2500. * (h.cgs * c.cgs / k_B.cgs).value / audit.hcperk
    radiance = BlackBody(effective_temperature * u.K)(nu.to(u.Hz, equivalencies=u.spectral()))
    flux_per_nu = (np.pi * u.sr * radiance * (29979245800. * u.cm / u.s)).to_value(u.erg / u.s / u.cm)
    flux_per_nu *= audit.fac_planck / (2 * np.pi * h.cgs * c.cgs**2).value
    integral = trapezoid(flux_per_nu, nu.value)
    np.testing.assert_allclose(accurate["flux_cgs"][0], integral, rtol=2e-8)
    assert "not an RCE" in report["scope"]


def test_cli_writes_reviewable_report_atomically(audit, product, monkeypatch):
    paths = saved_run(audit, product)
    # Quadrature is independently tested above; keep the CLI orchestration tiny.
    monkeypatch.setattr(audit, "blackbody_audit", lambda: {"scope": "tested separately"})
    output = paths[0].parent / "audit" / "report.json"
    assert audit.main(["--run", str(paths[0]), "--manifest", str(paths[1]),
                       "--observation", str(paths[2]), "--output", str(output),
                       "--response-resolutions", "400", "700", "1600",
                       "--wavelength-fwhm-nm", "1.8", "2.0"]) == 0
    report = json.loads(output.read_text())
    assert report["success"] and report["p7_physical_acceptance"] == "not_certified"
    assert [row["resolving_power"] for row in report["response"]["gaussian_trials"]] == [400., 700., 1600.]
    widths = report["response"]["wavelength_width_trials"]
    assert [row["wavelength_fwhm_nm"] for row in widths] == [1.8, 2.0]
    np.testing.assert_allclose(widths[1]["resolving_power_range"], [500., 1500.])
    assert "vs_constant_R700" in widths[1]
    assert not output.with_name(output.name + ".tmp").exists()


def test_vector_response_on_nonuniform_grid_is_normalized_and_differentiable(audit):
    from scipy.integrate import quad
    from scipy.special import ndtr

    nu = np.array([990., 999.7, 999.97, 1000., 1000.02, 1000.11, 1001., 1010.])
    line = np.array([0., 0., -.6, -.9, -.8, 0., 0., 0.])
    centers, resolutions = np.array([1000.03, 999.8]), np.array([650., 1500.])
    expected = []
    for center, resolution in zip(centers, resolutions):
        sigma = center / (resolution * np.sqrt(8 * np.log(2)))
        lower, upper = center - 4 * sigma, center + 4 * sigma
        points = nu[(nu > lower) & (nu < upper)]
        expected.append(quad(lambda n: np.interp(n, nu, line)
                             * np.exp(-.5 * ((n - center) / sigma)**2)
                             / (np.sqrt(2 * np.pi) * sigma), lower, upper,
                             points=points, epsabs=1e-12)[0] / (ndtr(4.) - ndtr(-4.)))
    evaluate = jax.jit(lambda amplitude: audit.response_samples(
        nu, 1. + amplitude * line, centers, resolutions))
    np.testing.assert_allclose(evaluate(0.), 1., rtol=2e-15)
    np.testing.assert_allclose(evaluate(1.), 1. + np.asarray(expected), rtol=2e-13)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(1.), expected, rtol=2e-12)
    np.testing.assert_array_equal(audit.response_samples(nu, 1. + line, centers, 700.),
                                  audit.response_samples(nu, 1. + line, centers, [700., 700.]))
    for invalid in ([700.], [[700., 800.]], [700., 0.], [np.nan, 700.], [700., np.inf]):
        with pytest.raises(ValueError, match="response|Response"):
            audit.response_samples(nu, 1. + line, centers, invalid)


def test_vector_coverage_checks_the_widest_interior_pixel(audit):
    column_module = importlib.import_module("_rce_wasp18b")
    assets = importlib.import_module("_rce_wasp18b_assets")
    column = column_module.Wasp18bColumn.__new__(column_module.Wasp18bColumn)
    column.pixel_nu = np.array([1000., 1100., 1200.])
    column.nu = np.array([900., 1300.])
    column.stellar = assets.StellarSpectrum(column.nu, np.ones(2), {})
    column.check_response_coverage([700., 600., 500.])
    with pytest.raises(ValueError, match="cover selected pixels"):
        column.check_response_coverage([1000., 3., 1000.])
    with pytest.raises(ValueError, match="match pixel_nu"):
        column.check_response_coverage([700., 600.])
    column.stellar = assets.StellarSpectrum(np.array([999., 1201.]), np.ones(2), {})
    with pytest.raises(ValueError, match="Stellar spectrum does not cover"):
        column.check_response_coverage([700., 600., 500.])
