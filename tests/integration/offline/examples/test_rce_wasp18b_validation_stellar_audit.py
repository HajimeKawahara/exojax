"""Both runners audit direct plus diffuse stellar leakage on a solved fixture.

The solver is scripted; these tests do not perform a physical RCE solve.
"""

from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import modules
from tests.integration.offline.examples.test_rce_wasp18b_validation import runner_column


def previous_forward():
    path = (Path(__file__).resolve().parents[4]
            / "results/rce_wasp18b_validation_stellar_audit_previous/rce_wasp18b_forward.py")
    spec = importlib.util.spec_from_file_location("forward_before_stellar_audit_extraction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", ["error", "zero"])
@pytest.mark.parametrize("scatterers", [False, True])
def test_validation_matches_original_forward_and_preserves_cached_solver(modules, tmp_path, policy, scatterers):
    validation = importlib.import_module("rce_wasp18b_validation")
    forward = importlib.import_module("rce_wasp18b_forward")
    _, column = runner_column(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range=policy)
    if not scatterers:
        state = column.chemistry.result
        column.chemistry.result = state._replace(x=state.x.at[:, 2:].set(0.))
    parameters = [0., .6, np.log(.5)]
    original = previous_forward().run_column(column, parameters, fixed_temperature=True,
                                             response_resolution=300.)
    current = forward.run_column(column, parameters, fixed_temperature=True, response_resolution=300.)
    evaluator = validation.PreparedValidationRun(column, parameters, 300.)
    solve, absorption = evaluator.solve, evaluator.stellar_absorption
    actual = evaluator.run(parameters)
    repeated = evaluator.run(parameters)
    assert actual["success"] and repeated["success"] and current["success"]
    assert evaluator.solve is solve and evaluator.stellar_absorption is absorption
    # The extraction preserves the original forward report exactly.
    assert validation.serializable(current["stellar_scattering"]) == validation.serializable(original["stellar_scattering"])
    assert actual["stellar_scattering"] == validation.serializable(current["stellar_scattering"])
    assert repeated["stellar_scattering"] == actual["stellar_scattering"]
    np.testing.assert_allclose(actual["prediction"], current["prediction"], rtol=2e-12, atol=0.)
    metadata = actual["stellar_scattering"]
    assert metadata["outside_range"] == policy
    assert metadata["outside_support_physical_error_bounded"] is False
    report = metadata["stellar_flux_audit"]
    flux = report["boundary_flux_cgs"]
    total = flux["downward"][-1] + flux["incident"][-1]
    assert report["bottom_reaching_fraction"] == pytest.approx(total / flux["incident"][0])
    assert report["bottom_total_stellar_over_internal_flux"] == pytest.approx(total / column.internal_flux)
    assert len(flux["net"]) == 3 and len(report["layer_heating_cgs"]) == 2
    if scatterers:
        assert flux["downward"][-1] > 0.
        assert report["bottom_reaching_fraction"] > actual["bottom_incident_fraction"]
    else:
        np.testing.assert_array_equal(flux["downward"], 0.)
        # Separate combined-flux and stellar-only JIT kernels can differ by one ulp.
        np.testing.assert_allclose(report["bottom_reaching_fraction"],
                                   actual["bottom_incident_fraction"], rtol=2e-15, atol=0.)


def test_shared_audit_respects_zero_support_at_both_endpoints(modules, tmp_path):
    audit = importlib.import_module("_rce_wasp18b_audit")
    _, column = runner_column(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=True,
                              stellar_scattering_outside_range="zero")
    column.band_nodes, column.band_weights = jnp.array([[100.], [100000.]]), jnp.ones(1)
    column.star_bands, column.band_widths = jnp.ones(2), jnp.ones(2)
    arguments = (column, jnp.array([2300., 2700.]), jnp.array([0., .6, -.7]), column.chemistry.result)
    depth = jnp.full((2, 2, 2), .25)
    report = audit.stellar_flux_audit(*arguments, absorption_depth=depth)
    flux = report["boundary_flux_cgs"]
    np.testing.assert_array_equal(flux["downward"], 0.)
    np.testing.assert_array_equal(flux["upward"], 0.)
    assert report["bottom_reaching_fraction"] == pytest.approx(np.exp(-.5 / column.settings.mu0))
    column.settings = replace(column.settings, stellar_scattering_outside_range="error")
    with pytest.raises(ValueError, match="200 <="):
        audit.stellar_flux_audit(*arguments, absorption_depth=depth)


def test_scattering_off_has_no_additional_report_or_absorption_compilation(modules, tmp_path):
    validation = importlib.import_module("rce_wasp18b_validation")
    _, column = runner_column(modules, tmp_path)
    parameters = [0., .6, np.log(.5)]
    evaluator = validation.PreparedValidationRun(column, parameters, 300.)
    assert evaluator.stellar_absorption is None
    report = evaluator.run(parameters)
    assert report["success"] and "stellar_scattering" not in report
