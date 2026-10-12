"""Couple real thermochemistry and opacity APIs using small artificial tables.

These offline fixtures verify assembly; they do not validate WASP-18b opacity
completeness or convergence against measured observations.
"""

from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity import OpaCKD, OpaDiffgrid
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.core import gauss_legendre_grid
from exojax.rt.planck import piBarr
from exojax.utils.constants import bar_cgs, kB, m_u


class _Teacher:
    method = "analytic-test"
    ready = True
    nu_grid = np.linspace(100.0, 50000.0, 129)

    def __init__(self, energy):
        self.energy = energy

    def xsmatrix(self, temperature, pressure):
        shape = 1.0 + (self.nu_grid / 10000.0 - self.energy / 1000.0)**2
        return (1e-25 * jnp.exp(-self.energy / temperature)[:, None]
                * jnp.asarray(pressure)[:, None]**0.1 * shape)


def _ckd(energy):
    edges = np.geomspace(100.0, 50000.0, 13)
    edges = jnp.asarray(np.column_stack((edges[:-1], edges[1:])))
    centers = jnp.mean(edges, axis=1)
    g, weights = gauss_legendre_grid(4)
    temperature = jnp.array([1500.0, 2500.0, 3500.0, 4500.0])
    pressure = jnp.array([1e-4, 10.0])
    log_k = (jnp.log(1e-25) - energy / temperature[:, None, None, None]
             + 0.1 * jnp.log(pressure)[None, :, None, None]
             + 0.2 * g[None, None, :, None]
             + jnp.log(1.0 + (centers / 10000.0 - energy / 1000.0)**2))
    opa = OpaCKD.load_only()
    opa.Ng, opa.nu_bands, opa.band_edges = len(g), centers, edges
    opa.ckd_info = CKDTableInfo(log_k, g, weights, temperature, pressure, centers, edges)
    opa.ready = True
    return opa


@pytest.fixture(scope="module")
def setup():
    pytest.importorskip("exogibbs.thermo.standard")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
            module = importlib.import_module("_rce_wasp18b")
            assets = importlib.import_module("_rce_wasp18b_assets")
            boundaries = np.array([0.001, 0.1, 10.0])
            pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
            ckd = {name: _ckd(energy) for name, energy in (("H2O1", 800.0), ("C1O1", 1400.0))}
            lbl = {name: OpaDiffgrid(_Teacher(energy), np.array([1500.0, 2500.0, 3500.0, 4500.0]), pressure)
                   for name, energy in (("H2O1", 800.0), ("C1O1", 1400.0))}
            cia = {pair: SimpleNamespace(
                nucia=jnp.array([100.0, 25000.0, 50000.0]),
                tcia=jnp.array([1500.0, 4500.0]), logac=jnp.full((2, 3), coefficient),
            ) for pair, coefficient in ((('H2', 'H2'), -46.0), (('H2', 'He1'), -47.0))}
            nu = np.linspace(50.0, 55000.0, 2049)
            stellar = assets.StellarSpectrum(nu, np.asarray(piBarr(jnp.array([6400.0]), nu)[0]),
                                             {"source": "offline blackbody fixture"})
            observation = SimpleNamespace(
                recipe={"radius_ratio": 0.097, "a_over_rstar": 3.5},
                W=np.array([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]),
                input_wavelength_um=np.array([0.8, 1.0, 1.5, 2.0, 20.0]),
                uncertainty=np.array([1e-5, 2e-5]),
            )
            arguments = dict(
                observation=observation, stellar=stellar, ckd=ckd, lbl=lbl, cia=cia,
                boundaries=boundaries, settings=module.ColumnSettings(19000.0, 200.0),
                isotope_fractions={"H2O1": 0.997, "C1O1": 0.986},
                temperature_initial=np.array([2200.0, 2800.0]), bottom_temperature_initial=3200.0,
            )
            column = module.Wasp18bColumn(**arguments)
            yield SimpleNamespace(module=module, column=column, arguments=arguments)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_full_chemistry_ckd_lbl_spectrum_and_jvp(setup):
    column = setup.column
    temperature = jnp.array([2200.0, 2800.0])
    bottom = 3200.0
    parameters = jnp.array([0.0, 0.6, np.log(0.5)])

    def prediction(parameters):
        chemistry = column.chemistry(jnp.append(temperature, bottom), parameters[0], parameters[1])
        flux = column.net_flux(temperature, bottom, parameters, chemistry)
        spectrum = column.spectrum(temperature, bottom, parameters, chemistry)
        return chemistry.diagnostics.valid, flux, spectrum

    calculate = jax.jit(prediction)
    valid, flux, spectrum = calculate(parameters)
    assert len(column.chemistry.species) == 538
    assert {'H2', 'He1', 'H2O1', 'C1O1', 'H1', 'e1-'}.issubset(column.chemistry.species)
    assert np.all(valid)
    assert flux.shape == (3,) and spectrum.shape == (2,)
    assert np.all(np.isfinite(flux)) and np.all(np.isfinite(spectrum) & (spectrum > 0))
    direction = jnp.array([0.3, 0.1, 0.2])
    derivative = jax.jit(lambda p: jax.jvp(
        lambda values: prediction(values)[1:], (p,), (direction,),
    )[1])(parameters)
    assert np.all(np.isfinite(derivative[0])) and np.all(np.isfinite(derivative[1]))
    assert np.linalg.norm(derivative[1]) > 1e-9
    for step in (2e-4, 1e-4):
        upper, lower = calculate(parameters + step * direction), calculate(parameters - step * direction)
        assert np.all(upper[0]) and np.all(lower[0])
        for actual, high, low in zip(derivative, upper[1:], lower[1:]):
            np.testing.assert_allclose(actual, (high - low) / (2 * step), rtol=3e-4, atol=1e-9)


def test_conservation_convergence_setting_requires_boolean(setup):
    settings = setup.arguments["settings"]
    assert settings.chemistry_conservation_convergence is False
    for value in (True, False):
        assert replace(settings, chemistry_conservation_convergence=value).chemistry_conservation_convergence == value
    for value in (1, "true", None, np.bool_(True)):
        with pytest.raises(ValueError, match="chemistry_conservation_convergence must be boolean"):
            replace(settings, chemistry_conservation_convergence=value)


def test_column_passes_conservation_optin_without_changing_tolerances(setup, monkeypatch):
    observed = []
    original = setup.module.prepare_chemistry

    def prepare(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(setup.module, "prepare_chemistry", prepare)
    for value in (True, False):
        settings = replace(setup.arguments["settings"], chemistry_conservation_convergence=value)
        column = setup.module.Wasp18bColumn(**dict(setup.arguments, settings=settings))
        assert column.settings.chemistry_conservation_convergence == value
        assert observed[-1]["conservation_convergence"] == value
        assert observed[-1]["conservation_rtol"] == 3e-6
        assert observed[-1]["epsilon_crit"] == 3e-14


def test_temperature_and_chemistry_invalidity_prevents_clipped_predictions(setup):
    column = setup.column
    temperature, bottom = jnp.array([2200.0, 2800.0]), 3200.0
    parameters = jnp.array([0.0, 0.6, np.log(0.5)])
    chemistry = column.chemistry(jnp.append(temperature, bottom), *parameters[:2])
    # Even a stale valid chemistry result must not permit table clamping.
    for invalid_temperature, invalid_bottom in ((temperature.at[0].set(1499.0), bottom),
                                               (temperature, 4501.0)):
        assert not column.temperature_valid(invalid_temperature, invalid_bottom, parameters)
        result = jax.jit(lambda t, b: column.spectrum(t, b, parameters, chemistry))(
            invalid_temperature, invalid_bottom,
        )
        assert np.all(np.isnan(result))
    invalid_chemistry = chemistry._replace(
        diagnostics=chemistry.diagnostics._replace(valid=jnp.zeros(3, dtype=bool))
    )
    flux = column.net_flux(temperature, bottom, parameters, invalid_chemistry)
    assert np.all(np.isnan(flux))


def test_explicit_resolved_tables_preserve_spectrum_and_are_runtime_inputs(setup):
    column = setup.column
    temperature, bottom = jnp.array([2200.0, 2800.0]), 3200.0
    parameters = jnp.array([0.0, 0.6, np.log(0.5)])
    chemistry = column.chemistry(jnp.append(temperature, bottom), *parameters[:2])
    tables = column.resolved_opacity_tables()
    compiled = jax.jit(lambda data: column.spectrum(
        temperature, bottom, parameters, chemistry, resolved_tables=data,
    ))
    expected = column.spectrum(temperature, bottom, parameters, chemistry)
    np.testing.assert_allclose(compiled(tables), expected, rtol=1e-12)
    changed = {name: (grid, values + np.log(2.0), derivative, floor)
               for name, (grid, values, derivative, floor) in tables.items()}
    assert np.max(np.abs(np.asarray(compiled(changed) - expected))) > 1e-10
    jaxpr = jax.make_jaxpr(compiled)(tables)
    large_arrays = [array for entry in tables.values() for array in entry if array.ndim == 3]
    assert not any(const is array for const in jaxpr.consts for array in large_arrays)


def test_continuum_uses_each_layer_mass_and_cia_number_densities(setup):
    from exojax.database.hminus import log_hminus_continuum

    module = setup.module
    nu = jnp.array([200.0, 2000.0, 10000.0, 60000.0])
    temperature = jnp.array([2000.0, 3000.0])
    boundaries = jnp.array([0.001, 0.1, 10.0])
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    species = ('H2', 'He1', 'H1', 'e1-')
    fractions = jnp.array([[0.8, 0.15, 0.05, 0.0], [0.6, 0.1, 0.3, 0.0], [0.5, 0.1, 0.4, 0.0]])
    mmw = jnp.array([2.3, 1.8, 1.5])
    chemistry = SimpleNamespace(x=fractions, mmw=mmw, number_density_e=jnp.zeros(3),
                                number_density_h=jnp.array([1e12, 3e13, 5e13]))
    result = module.continuum_optical_depth(nu, temperature, pressure, boundaries, chemistry,
                                         species, 19000.0, setup.arguments['cia'])
    density = np.asarray(pressure) * bar_cgs / (kB * np.asarray(temperature))
    column_number = np.diff(boundaries) * bar_cgs / (19000.0 * m_u * np.asarray(mmw[:-1]))
    expected = column_number * density * (1e-46 * np.asarray(fractions[:-1, 0])**2
                                         + 1e-47 * np.asarray(fractions[:-1, 0] * fractions[:-1, 1]))
    np.testing.assert_allclose(result[:, :3], np.broadcast_to(expected[:, None], (2, 3)), rtol=1e-13)
    np.testing.assert_array_equal(result[:, 3], 0.0)
    # No artificial floor is introduced for an absent electron population.
    no_cia = module.continuum_optical_depth(nu, temperature, pressure, boundaries, chemistry,
                                         species, 19000.0, {})
    np.testing.assert_array_equal(no_cia, 0.0)
    electrons = jnp.array([1e8, 1e10, 1e11])
    ionized = SimpleNamespace(**dict(vars(chemistry), number_density_e=electrons))
    hminus = module.continuum_optical_depth(nu, temperature, pressure, boundaries, ionized,
                                          species, 19000.0, {})
    extinction = 10.0 ** log_hminus_continuum(
        nu, temperature, electrons[:-1], ionized.number_density_h[:-1],
    )
    distance = column_number / density
    np.testing.assert_allclose(hminus, extinction * distance[:, None], rtol=1e-13)

    def from_electrons(values):
        state = SimpleNamespace(**dict(vars(chemistry), number_density_e=values))
        return module.continuum_optical_depth(nu, temperature, pressure, boundaries, state,
                                             species, 19000.0, {})

    _, derivative = jax.jvp(from_electrons, (jnp.zeros(3),), (electrons,))
    np.testing.assert_allclose(derivative, hminus, rtol=1e-13)


def test_column_rejects_table_and_chemistry_domain_mismatches(setup):
    arguments = setup.arguments
    with pytest.raises(ValueError, match="identical nonempty species"):
        setup.module.Wasp18bColumn(**dict(arguments, lbl={'H2O1': arguments['lbl']['H2O1']}))
    narrow = _ckd(800.0)
    narrow.ckd_info = replace(narrow.ckd_info, T_grid=jnp.array([600.0, 800.0, 1000.0, 1400.0]))
    with pytest.raises(ValueError, match="temperature domains do not overlap"):
        setup.module.Wasp18bColumn(**dict(arguments, ckd=dict(arguments['ckd'], H2O1=narrow)))
    with pytest.raises(ValueError, match="pressure.*match|does not match"):
        setup.module.Wasp18bColumn(**dict(arguments, boundaries=np.array([0.001, 0.2, 10.0])))


def test_response_coverage_shared_constant_ratio_and_observation_budget(setup):
    column, module = setup.column, setup.module
    column.check_response_coverage(100.0)
    with pytest.raises(ValueError, match="response"):
        column.check_response_coverage(0.0)
    with pytest.raises(ValueError, match="response"):
        column.check_response_coverage(1.0)
    stellar = module.response_samples(column.nu, column.star_resolved, column.pixel_nu, 100.0)
    planet = module.response_samples(column.nu, 0.17 * column.star_resolved, column.pixel_nu, 100.0)
    from _rce_observation import pixel_flux_ratio
    np.testing.assert_allclose(pixel_flux_ratio(column.W, planet, stellar, column.area_ratio),
                               0.17 * column.area_ratio)
    baseline, sigma = np.array([0.001, 0.002]), np.array([1e-5, 2e-5])
    assert module.spectral_error(baseline + sigma * 0.09, baseline, sigma)['passed']
    assert not module.spectral_error(baseline + sigma * 0.11, baseline, sigma)['passed']
    with pytest.raises(ValueError, match="positive uncertainties"):
        module.spectral_error(baseline, baseline, np.zeros(2))


def test_cia_exact_zeros_have_finite_temperature_derivatives(setup):
    module = setup.module
    temperature = jnp.array([1500.0, 3000.0])
    boundaries = jnp.array([0.001, 0.1, 10.0])
    pressure = jnp.sqrt(boundaries[:-1] * boundaries[1:])
    nu = jnp.array([50.0, 100.0, 150.0, 200.0, 300.0, 350.0])
    species = ('H2', 'He1')
    fractions = jnp.array([[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]])
    mmw = jnp.array([2.3, 1.8, 1.5])
    chemistry = SimpleNamespace(x=fractions, mmw=mmw, number_density_e=jnp.zeros(3),
                                number_density_h=jnp.zeros(3))
    coefficients = np.array([[0.0, 0.0, 1e-46], [0.0, 2e-46, 3e-46]])
    with np.errstate(divide='ignore'):
        logac = jnp.asarray(np.log10(coefficients))
    table = SimpleNamespace(nucia=jnp.array([100.0, 200.0, 300.0]),
                            tcia=jnp.array([1500.0, 4500.0]), logac=logac)

    def evaluate(values):
        return module.continuum_optical_depth(nu, values, pressure, boundaries, chemistry,
                                             species, 19000.0, {('H2', 'H2'): table})

    result, derivative = jax.jit(lambda values: jax.jvp(
        evaluate, (values,), (jnp.ones_like(values),),
    ))(temperature)
    assert np.all(np.isfinite(result)) and np.all(np.isfinite(derivative))
    interpolated = np.stack([np.interp(nu, table.nucia, values, left=0.0, right=0.0)
                             for values in coefficients])
    slope = (interpolated[1] - interpolated[0]) / 3000.0
    ac = interpolated[0] + (np.asarray(temperature)[:, None] - 1500.0) * slope
    factor = (np.diff(boundaries) * bar_cgs / (19000.0 * m_u * np.asarray(mmw[:-1]))
              * np.asarray(pressure) * bar_cgs / kB * np.asarray(fractions[:-1, 0])**2)
    expected = factor[:, None] * ac / np.asarray(temperature)[:, None]
    expected_derivative = factor[:, None] * (
        slope / np.asarray(temperature)[:, None] - ac / np.asarray(temperature)[:, None]**2
    )
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-14, atol=1e-18)
    np.testing.assert_array_equal(result[:, [0, 1, 5]], 0.0)
    np.testing.assert_array_equal(derivative[:, [0, 1, 5]], 0.0)


def test_response_integrates_a_narrow_native_line_and_preserves_flux_gradient(setup):
    from scipy.integrate import quad
    from scipy.special import ndtr

    nu = np.array([980.0, 999.981, 999.999, 1000.001, 1000.005, 1000.08, 1020.0])
    shape = np.array([0.0, 0.0, -0.9, -0.99, -0.9, 0.0, 0.0])
    centers, resolution = np.array([1000.003, 1000.25]), 1000.0
    expected, expected_gradient = [], []
    for center in centers:
        sigma = center / (resolution * np.sqrt(8 * np.log(2)))
        lower, upper = center - 4 * sigma, center + 4 * sigma
        points = nu[(nu > lower) & (nu < upper)]
        norm = ndtr(4.0) - ndtr(-4.0)

        def integral(values):
            return quad(lambda value: np.interp(value, nu, values)
                        * np.exp(-0.5 * ((value - center) / sigma)**2)
                        / (np.sqrt(2 * np.pi) * sigma),
                        lower, upper, points=points, epsabs=1e-12)[0] / norm

        expected.append(integral(1.0 + shape))
        expected_gradient.append(integral(shape))

    evaluate = jax.jit(lambda amplitude: setup.module.response_samples(
        nu, 1.0 + amplitude * jnp.asarray(shape), centers, resolution,
    ))
    np.testing.assert_allclose(evaluate(1.0), expected, rtol=1e-12, atol=1e-13)
    assert np.max(1.0 - np.asarray(expected)) > 0.01
    derivative = jax.jit(jax.jacfwd(evaluate))(1.0)
    np.testing.assert_allclose(derivative, expected_gradient, rtol=1e-12, atol=1e-13)


def test_failed_forward_keeps_solver_failure_when_comparison_requested(setup, monkeypatch, tmp_path):
    runner = importlib.import_module("rce_wasp18b_forward")
    observation = SimpleNamespace(sha256="observation", row_indices=[0, 1],
                                  uncertainty=setup.column.observation.uncertainty)
    config = dict(parameters=[0., .6, np.log(.5)], prior_lower=[-1., .2, np.log(.1)],
                  prior_upper=[1., 1.2, 0.], stellar={"asset": "star"}, cia={},
                  opacity=dict(mapping={}, omitted_absorbers=[], scattering="omitted",
                               broadening_limitations=[]))
    manifest = SimpleNamespace(config=config, sha256={"star": "stellar"})
    monkeypatch.setattr(runner, "load_emission_candidate", lambda _: observation)
    monkeypatch.setattr(runner, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "prepare_column", lambda *args, **kwargs: setup.column)
    monkeypatch.setattr(runner, "provenance", lambda: {"test_fixture": True})
    monkeypatch.setattr(runner, "run_column", lambda *args, **kwargs: dict(
        success=False, failure="rce_not_converged", state=dict(converged=False, status=9)))
    output = tmp_path / "failure.json"
    # A failed solve must not even open a reference spectrum for comparison.
    status = runner.main(["--manifest", "unused", "--observation", "unused",
                          "--output", str(output), "--compare", str(tmp_path / "missing.json")])
    report = json.loads(output.read_text())
    assert status == 1 and not report["success"]
    assert report["failure"] == "rce_not_converged"
    assert report["state"]["status"] == 9
    assert "comparison_skipped" in report and "comparison" not in report
    assert "prediction" not in report


def test_forward_runner_fixed_temperature_report_and_comparison(setup):
    runner = importlib.import_module('rce_wasp18b_forward')
    column = setup.column
    report = runner.run_column(column, [0.0, 0.6, np.log(0.5)],
                               fixed_temperature=True, response_resolution=100.0)
    assert report['success'] and report['mode'] == 'fixed_temperature'
    assert np.all(report['chemistry']['valid'])
    assert np.all(np.isfinite(report['prediction']))
    assert 0.0 <= report['bottom_incident_fraction'] <= 1.0
    assert 'ckd_interpolation_vs_resolved' in report and 'response_vs_centers' in report
    assert np.all(np.isfinite(report['blackbody_integral']['relative_error']))
    for name in ('spectrum_compile_seconds', 'first_spectrum_seconds', 'warm_spectrum_seconds'):
        assert np.isfinite(report[name]) and report[name] >= 0.0
    report.update(observation_sha256={'fixture': 'synthetic'}, row_indices=[0, 1],
                  uncertainty=column.observation.uncertainty,
                  physical_inputs={'gravity_cgs': column.settings.gravity_cgs},
                  provenance=runner.provenance(),
                  thermodynamics=column.entropy.thermodynamics.metadata)
    serialized = json.loads(json.dumps(runner._serializable(report), allow_nan=False))
    assert serialized['provenance']['x64'] and serialized['provenance']['code_sha256']
    assert runner.compare_runs(serialized, serialized)['passed']
    changed = json.loads(json.dumps(serialized))
    changed['prediction'][0] += 0.09 * changed['uncertainty'][0]
    result = runner.compare_runs(changed, serialized)
    assert result['kind'] == 'numerical' and result['passed']
    changed['physical_inputs']['gravity_cgs'] *= 1.1
    with pytest.raises(ValueError, match='identical physical inputs'):
        runner.compare_runs(changed, serialized)
    assert runner.compare_runs(changed, serialized, kind='physics')['kind'] == 'physics'
    changed['row_indices'] = [1, 2]
    with pytest.raises(ValueError, match='row_indices'):
        runner.compare_runs(changed, serialized, kind='physics')


def test_rce_runner_converges_and_preserves_a_domain_limited_failure(setup):
    from exojax.atm.rce_device import RceStatus

    runner = importlib.import_module('rce_wasp18b_forward')
    column = setup.column
    # These are explicit toy transfer controls, not a narrowed scientific prior.
    # Keep the stronger-irradiation failure alongside the converged branch.
    success = runner.run_column(column, [0.0, 0.6, np.log(0.1)])
    assert success['success'] and success['mode'] == 'rce'
    state = success['state']
    assert state['converged'] and state['physics_valid'] and state['domain_valid']
    assert int(state['status']) == RceStatus.CONVERGED
    np.testing.assert_allclose(state['radiative_flux'] + state['convective_flux'],
                               column.internal_flux, rtol=0, atol=0.1)
    assert np.max(state['stability_residual']) <= 1e-8
    assert np.all(state['convective_flux'] >= -0.1)
    assert np.all(np.isfinite(success['prediction']))

    failure = runner.run_column(column, [0.0, 0.6, np.log(0.5)])
    assert not failure['success'] and failure['failure'] == 'rce_not_converged'
    assert 'prediction' not in failure
    last = failure['state']
    assert int(last['status']) == RceStatus.DOMAIN_STEP_FAILED
    assert not last['converged'] and last['domain_valid'] and last['physics_valid']
    assert np.all(np.isfinite(last['temperature']))
    assert np.all(np.isfinite(last['flux_residual']))
    assert last['bottom_temperature'] <= column.temperature_range[1]


def test_forward_runner_records_explicit_mask_continuation(setup):
    runner = importlib.import_module('rce_wasp18b_forward')
    report = runner.run_column(setup.column, [0., .6, np.log(.1)],
                               mask_continuation=True)
    assert report['success'] and report['state']['converged']
    history = report['mask_continuation']
    assert history['stop_reason'] == 'converged'
    assert history['stages'][-1]['state']['converged']
    np.testing.assert_array_equal(history['stages'][0]['initial_temperature_K'],
                                  setup.column.solver_arguments[2])
    assert report['warm_solve_seconds'] is None
    assert np.all(np.isfinite(report['prediction']))
