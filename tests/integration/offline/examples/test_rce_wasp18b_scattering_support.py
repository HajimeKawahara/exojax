"""Finite scattering support is explicit and shared by all transfer paths."""

from copy import copy, deepcopy
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b import setup
from tests.integration.offline.examples.test_rce_wasp18b_comparison_inputs import inputs, run
from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


@pytest.mark.parametrize("policy", [None, "clip", "ZERO", 0, [], {}])
def test_invalid_policy_is_rejected_even_if_scattering_is_disabled(modules, policy):
    with pytest.raises(ValueError, match="outside_range"):
        modules.column.ColumnSettings(19000., 200., stellar_scattering_outside_range=policy)


@pytest.mark.parametrize("nu", [[0.], [-1.], [np.nan], [np.inf], []])
@pytest.mark.parametrize("policy", ["error", "zero"])
def test_invalid_coordinates_are_never_zero_filled(modules, nu, policy):
    scattering = importlib.import_module("_rce_wasp18b_scattering")
    with pytest.raises(ValueError, match="positive finite"):
        scattering.scattering_cross_sections(nu, outside_range=policy)


def test_supported_values_are_bitwise_identical_and_endpoints_are_inclusive(modules):
    scattering = importlib.import_module("_rce_wasp18b_scattering")
    inside = np.geomspace(200., 50000., 513)
    nu = np.concatenate(([100., np.nextafter(200., 0.)], inside,
                         [np.nextafter(50000., np.inf), 100000.]))
    expected = scattering.scattering_cross_sections(inside)
    actual = scattering.scattering_cross_sections(nu, outside_range="zero")
    for name in expected:
        np.testing.assert_array_equal(actual[name][2:-2], expected[name])
        np.testing.assert_array_equal(actual[name][[0, 1, -2, -1]], 0.)
        assert actual[name][2] > 0 and actual[name][-3] > 0
    with pytest.raises(ValueError, match="200 <="):
        scattering.scattering_cross_sections(nu)
    traced = jax.jit(lambda values: scattering._cross_sections(values, jnp, outside_range="zero"))(nu)
    native = jax.jit(lambda values: scattering._cross_sections(values, jnp))(inside)
    for name in expected:
        np.testing.assert_array_equal(traced[name][2:-2], native[name])
        np.testing.assert_array_equal(np.asarray(traced[name])[[0, 1, -2, -1]], 0.)
        np.testing.assert_allclose(traced[name], actual[name], rtol=5e-15, atol=0.)


def test_band_audit_and_traced_depth_use_the_same_support_and_chemistry_jvp(modules, tmp_path):
    scattering = importlib.import_module("_rce_wasp18b_scattering")
    audit = importlib.import_module("_rce_wasp18b_audit")
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range="zero")
    column.band_nodes = jnp.array([[100., 199.], [200., 4000.], [49999., 50000.], [50001., 100000.]])
    column.band_weights = jnp.array([.4, .6])
    chemistry = chemistry._replace(x=chemistry.x.at[:, 4].set(.1).at[:, 5].set(.01))

    def evaluate(scale):
        state = chemistry._replace(x=chemistry.x * scale)
        depth = scattering.scattering_layer_depth(column.band_nodes, state, column.boundaries,
                    column.settings.gravity_cgs, column.chemistry.species, outside_range="zero")
        return jnp.sum(depth * column.band_weights, axis=-1)

    value, tangent = jax.jvp(jax.jit(evaluate), (1.,), (1.,))
    expected = sum(audit.scattering_optical_depth(column, chemistry).values())
    np.testing.assert_allclose(value, expected, rtol=5e-15, atol=0.)
    np.testing.assert_array_equal(value[:, [0, -1]], 0.)
    np.testing.assert_allclose(tangent, value, rtol=5e-15, atol=0.)


def test_column_requires_explicit_zero_for_extended_grid(setup):
    arguments = dict(setup.arguments)
    arguments["settings"] = replace(arguments["settings"], stellar_scattering=True)
    with pytest.raises(ValueError, match="200 <="):
        setup.module.Wasp18bColumn(**arguments)
    arguments["settings"] = replace(arguments["settings"], stellar_scattering_outside_range="zero")
    column = setup.module.Wasp18bColumn(**arguments)
    assert float(column.band_edges[0, 0]) == 100.
    assert column.settings.stellar_scattering_outside_range == "zero"


def test_column_transfer_has_no_scattering_outside_support(modules, tmp_path):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    column.nu = jnp.array([100., 200., 4000., 50000., 100000.])
    column.star_resolved = jnp.ones(5)
    column.optical_depth = lambda *args, **kwargs: jnp.ones((2, 5)) * .5
    absorption_column = copy(column)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range="zero")
    chemistry = chemistry._replace(x=chemistry.x.at[:, 4].set(.1).at[:, 5].set(.01))
    args = (jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.7]), chemistry)
    expected = jax.jit(lambda *values: absorption_column.fluxes(*values, resolved=True))(*args)
    actual = jax.jit(lambda *values: column.fluxes(*values, resolved=True))(*args)
    for left, right in zip(actual, expected):
        # The two transfer closures can differ by one rounding unit even at
        # zero scattering; the cross-section zero itself is exact above.
        np.testing.assert_allclose(np.asarray(left)[:, [0, -1]], np.asarray(right)[:, [0, -1]],
                                   rtol=2e-14, atol=0.)
    assert np.all(np.asarray(actual.upward[0, 1:-1]) > np.asarray(expected.upward[0, 1:-1]))


@pytest.mark.parametrize("resolved", [False, True])
def test_column_tiny_opacity_outside_support_has_finite_jacobian(modules, tmp_path, resolved):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    column.nu = jnp.array([100., 200., 4000., 50000., 100000.])
    column.star_resolved = column.star_bands = jnp.ones(5)
    column.band_nodes, column.band_weights = column.nu[:, None], jnp.ones(1)
    column.band_widths, column.g_weights = jnp.ones(5), jnp.array([.3, .7])

    def optical_depth(temperature, chemistry, *, resolved=False, resolved_tables=None):
        depth = (temperature[:, None] / 2500.
                 * jnp.array([[1e-200, .5, .5, .5, 1e-290]]))
        return depth if resolved else jnp.broadcast_to(depth[:, None, :], (2, 2, 5))

    column.optical_depth = optical_depth
    absorption_column = copy(column)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range="zero")

    def evaluate(model, vector):
        temperature, bottom, parameters = vector[:2], vector[2], vector[3:]
        hydrogen = .2 + .01 * jnp.append(temperature, bottom) / 3000.
        state = chemistry._replace(x=chemistry.x.at[:, 4].set(hydrogen))
        return jnp.stack(model.fluxes(temperature, bottom, parameters, state, resolved=resolved))

    point = jnp.array([2300., 2700., 3000., 0., .6, -.7])
    function = jax.jit(lambda vector: evaluate(column, vector))
    reference = jax.jit(lambda vector: evaluate(absorption_column, vector))
    value, expected = function(point), reference(point)
    forward = jax.jacfwd(function)(point)
    reverse = jax.jacrev(function)(point)
    expected_jacobian = jax.jacfwd(reference)(point)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(forward))
    assert np.all(np.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(value[..., [0, -1]], expected[..., [0, -1]], rtol=2e-14, atol=0.)
    np.testing.assert_allclose(forward[..., [0, -1], :], expected_jacobian[..., [0, -1], :],
                               rtol=2e-12, atol=0.)


def test_numeric_comparison_preserves_legacy_default_but_rejects_changed_support(modules):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest, column = inputs(modules)
    original = forward.physical_inputs(manifest, column, 700.)
    legacy = deepcopy(original)
    del legacy["stellar_scattering_outside_range"]
    assert forward.compare_runs(run(original), run(legacy))["passed"]
    assert "stellar_scattering_outside_range" not in legacy
    column.settings = replace(column.settings, stellar_scattering_outside_range="zero")
    current = forward.physical_inputs(manifest, column, 700.)
    assert current["stellar_scattering_outside_range"] == "zero"
    with pytest.raises(ValueError, match="identical physical inputs"):
        forward.compare_runs(run(current), run(legacy))
    assert forward.compare_runs(run(current), run(current))["passed"]
    assert forward.compare_runs(run(current), run(legacy), kind="physics")["kind"] == "physics"


def test_cli_reads_manifest_policy_and_records_it_without_mutating_configuration(modules, monkeypatch, tmp_path):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest, column = inputs(modules)
    manifest.config.update(parameters=[0., .6, -.7], prior_lower=[-1., .2, -2.],
                           prior_upper=[1., 1.2, 0.], column={"settings": dict(
                               gravity_cgs=18000., internal_temperature=200.,
                               stellar_scattering=True, stellar_scattering_outside_range="zero")})
    manifest.config["opacity"]["scattering"] = "stellar_two_stream"
    original = deepcopy(manifest.config)
    column.settings = modules.column.ColumnSettings(**manifest.config["column"]["settings"])
    column.temperature_valid = lambda *args: True
    column.temperature_range = (1500., 4500.)
    column.solver_arguments = (None, None, np.array([2500., 2500.]), 2500.)
    column.chemistry = SimpleNamespace(species=["fixture"])
    observation = SimpleNamespace(sha256={}, row_indices=[0], uncertainty=np.array([1e-5]))
    monkeypatch.setattr(forward, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(forward, "load_emission_candidate", lambda _: observation)
    monkeypatch.setattr(forward, "provenance", lambda: {"test_fixture": True})

    def prepare(actual, obs, **kwargs):
        assert actual is manifest and obs is observation
        assert modules.column.ColumnSettings(**actual.config["column"]["settings"]) == column.settings
        return column

    monkeypatch.setattr(forward, "prepare_column", prepare)
    output = tmp_path / "check.json"
    assert forward.main(["--manifest", "unused", "--observation", "unused",
                         "--output", str(output), "--check-inputs"]) == 0
    report = json.loads(output.read_text())
    assert report["settings"]["stellar_scattering_outside_range"] == "zero"
    assert report["physical_inputs"]["stellar_scattering_outside_range"] == "zero"
    assert report["configuration"] == original == manifest.config


def test_failed_run_still_declares_finite_model_scope(modules, tmp_path):
    forward = importlib.import_module("rce_wasp18b_forward")
    _, column, _ = fixture_manifest(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range="zero")
    column.solver_arguments = (column.pressure, column.boundaries, jnp.array([0., 0.]), 0.)
    report = forward.run_column(column, jnp.array([0., .6, -.7]), fixed_temperature=True)
    assert not report["success"]
    assert report["stellar_scattering"]["outside_range"] == "zero"
    assert report["stellar_scattering"]["coefficient_support_cm1"] == [200., 50000.]
    assert report["stellar_scattering"]["outside_support_physical_error_bounded"] is False
