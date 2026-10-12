"""Exercise streaming runner orchestration with a scripted solved-state fixture.

The radiative transfer and archived opacity paths are real; the supplied
solver state and chemistry isolate runner caching/checkpoints from the RCE
solver and thermochemical tests maintained separately.
"""

from copy import copy
import hashlib
import importlib
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


class State(NamedTuple):
    temperature: object
    bottom_temperature: object
    flux_residual: object
    stability_residual: object
    converged: object
    status: object


class FixedChemistry:
    def __init__(self, species, result):
        self.species, self.result = species, result

    def __call__(self, nodes, metal_scale, carbon_to_oxygen):
        return self.result


class ConstantCKD:
    nu_bands = jnp.array([3750., 4250.])

    def xstensor_ckd(self, temperature, pressure):
        return jnp.full((temperature.size, 2, 2), 1e-24)


def runner_column(modules, tmp_path):
    from _rce_broadband import band_stellar_flux, make_band_quadrature

    manifest, column, chemistry = fixture_manifest(modules, tmp_path)
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    column.chemistry = FixedChemistry(column.chemistry.species, chemistry)
    column.ckd = {name: ConstantCKD() for name in column.lbl}
    column.g_weights = jnp.array([.5, .5])
    column.band_edges = jnp.array([[3500., 4000.], [4000., 4500.]])
    column.band_nodes, column.band_weights = make_band_quadrature(column.band_edges, 8)
    column.band_widths = jnp.array([500., 500.])
    column.star_bands = jnp.asarray(band_stellar_flux(
        column.stellar.nu_cm1, column.stellar.surface_flux_per_cm1, column.band_edges))
    column.internal_flux = modules.column.SIGMA_SB * 200**4
    column.observation = SimpleNamespace(uncertainty=np.full(3, 1e-5))
    temperature, bottom = jnp.array([2300., 2700.]), 3000.
    column.solver_arguments = (column.pressure, column.boundaries, temperature, bottom)
    column.solver_options = {}
    column.solve = lambda parameters: State(
        temperature + 0 * parameters[0], jnp.asarray(bottom), jnp.zeros(3),
        -jnp.ones(2), jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32))
    streamed = copy(column)
    streamed.lbl, streamed.resolved_stream = metadata, plan
    return column, streamed


def test_streaming_validation_reuses_solver_and_checkpoints(modules, tmp_path):
    validation = importlib.import_module("rce_wasp18b_validation")
    forward = importlib.import_module("rce_wasp18b_forward")
    full, streamed = runner_column(modules, tmp_path)
    parameters = [0., .6, np.log(.5)]
    # Exercise the actual forward runner's streaming branch and its diagnostics.
    reference = forward.run_column(full, parameters, fixed_temperature=True, response_resolution=300.)
    saved = tmp_path / "resolved_spectrum.npz"
    actual = forward.run_column(streamed, parameters, fixed_temperature=True,
                                response_resolution=300., resolved_output=saved)
    assert actual["success"] and actual["resolved_streaming"]["all_chunk_coefficients_verified"]
    assert "ckd_interpolation_vs_resolved" in actual and "response_vs_centers" in actual
    np.testing.assert_allclose(actual["prediction"], reference["prediction"], rtol=2e-12, atol=0)
    assert actual["spectrum_compile_seconds"] is None
    assert actual["resolved_spectrum_file"]["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    with np.load(saved, allow_pickle=False) as archive:
        expected_flux = full.fluxes(
            archive["temperature_K"], archive["bottom_temperature_K"], archive["parameters"],
            full.chemistry.result, resolved=True, resolved_tables=full.resolved_opacity_tables(),
        ).upward[0]
        np.testing.assert_array_equal(archive["nu_cm1"], full.nu)
        np.testing.assert_allclose(archive["planet_upward_flux_per_cm1"], expected_flux,
                                   rtol=2e-12, atol=0)
        np.testing.assert_array_equal(archive["temperature_K"], actual["temperature_K"])
        np.testing.assert_array_equal(archive["parameters"], parameters)

    evaluator = validation.PreparedValidationRun(streamed, parameters, 300.)
    assert evaluator.resolved_tables is None
    solve = evaluator.solve
    checkpoints = []
    first = evaluator.run(parameters, on_state=checkpoints.append)
    chemistry, radiation = evaluator.compiled_chemistry, evaluator.compiled_radiation
    second = evaluator.run([.1, .6, np.log(.4)], on_state=checkpoints.append)
    assert first["success"] and second["success"]
    np.testing.assert_allclose(first["prediction"], reference["prediction"], rtol=2e-12, atol=0)
    assert evaluator.solve is solve and evaluator.compiled_chemistry is chemistry
    assert evaluator.compiled_radiation is radiation
    assert "radiation_compile_seconds" not in second
    assert len(checkpoints) == 2 and all("prediction" not in state for state in checkpoints)
    assert first["resolved_streaming"]["all_chunk_coefficients_verified"]
    report = {"cases": [{}, {}], "runs": [first, second]}
    validation.checkpoint(tmp_path / "report.json", report)
    assert report["summary"]["successful"] == 2
    assert not (tmp_path / "report.json.tmp").exists()


def test_variant_keeps_stream_for_unchanged_pressure(modules, tmp_path, monkeypatch):
    validation = importlib.import_module("rce_wasp18b_validation")
    _, base = runner_column(modules, tmp_path)

    def construct(*args, **kwargs):
        result = copy(base)
        result.settings = args[6]
        return result

    monkeypatch.setattr(validation, "Wasp18bColumn", construct)
    case = {"initial_temperature_K": 2600., "settings": {"nstream": 16}}
    variant = validation.variant_column(base, case)
    assert variant is not base and variant.resolved_stream is base.resolved_stream
    assert variant.settings.nstream == 16 and base.settings.nstream == 8
    with pytest.raises(ValueError, match="retain stored opacity pressure centers"):
        validation.variant_column(base, dict(case, pressure_boundaries_bar=[.001, .1, 9.]))


@pytest.mark.parametrize("suite", ["vertical", "depth"])
def test_pressure_suites_accept_streaming_option(modules, tmp_path, monkeypatch, suite):
    validation = importlib.import_module("rce_wasp18b_validation")
    output = tmp_path / "missing" / "report.json"
    def stop_at_manifest(path):
        raise ValueError("reached manifest loading")
    monkeypatch.setattr(validation, "load_manifest", stop_at_manifest)
    status = validation.main(["--suite", suite, "--stream-resolved", "--manifest", "unused.json",
                              "--observation", "unused.ecsv", "--output", str(output)])
    assert status == 1
    assert "reached manifest loading" in output.read_text()
