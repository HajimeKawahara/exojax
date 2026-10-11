"""Validation keeps continuation failures and evaluates only final roots."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce_device import ColumnEvaluation, make_device_rce_solver


class _Diagnostics(NamedTuple):
    valid: object


class _Chemistry(NamedTuple):
    diagnostics: _Diagnostics


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples"))
    return importlib.import_module("rce_wasp18b_validation")


def _column(streaming):
    from _rce_broadband import Fluxes

    def evaluate(t, tb, parameters):
        return ColumnEvaluation(
            jnp.stack([100 + t[0] - (2000 + parameters[0]),
                       100 + .02 * (tb - 5000)]),
            jnp.atleast_1d((tb - 3000) / 3000), jnp.array(True), jnp.int32(0),
        )

    def valid(t, tb, parameters):
        nodes = jnp.append(t, tb)
        return jnp.all((nodes >= 1500) & (nodes <= 4500)) & (parameters[0] >= 0)

    def fluxes(t, tb, parameters, chemistry):
        upward = jnp.full((2, 1, 1), 110. / 9000.)
        downward = jnp.zeros_like(upward)
        incident = jnp.full_like(upward, 10. / 9000.)
        return Fluxes(upward, downward, incident, upward - incident)

    arguments = ([1.], [.1, 10.], [2000.], 2500., 100., evaluate)
    options = dict(flux_atol=.1, flux_rtol=0., stability_atol=1e-8)
    column = SimpleNamespace(
        solver_arguments=arguments, solver_options=options, temperature_valid=valid,
        internal_flux=100., pressure=jnp.array([1.]), temperature_range=(1500., 4500.),
        check_response_coverage=lambda resolution: None,
        solve=make_device_rce_solver(*arguments, valid_temperature=valid, **options),
        chemistry=lambda nodes, metallicity, ratio: _Chemistry(_Diagnostics(jnp.ones(2, dtype=bool))),
        resolved_opacity_tables=lambda: {},
        spectrum=lambda t, tb, p, c, **kwargs: jnp.array([t[0] / tb]),
        fluxes=fluxes, g_weights=jnp.ones(1), band_widths=jnp.array([9000.]),
        band_nodes=jnp.array([[1000., 10000.]]), band_weights=jnp.array([.5, .5]),
        stream_calls=[],
    )
    if streaming:
        def top_flux(column, t, tb, parameters, chemistry):
            column.stream_calls.append(True)
            return jnp.array([t[0] / tb])

        column.resolved_stream = SimpleNamespace(
            provenance={"test_fixture": True}, top_flux=top_flux,
            prediction=lambda column, flux, resolution: flux,
        )
    return column


@pytest.mark.parametrize("streaming", [False, True])
def test_optin_keeps_all_stages_and_predictions_require_convergence(validation, streaming):
    parameters = jnp.array([0., .6, -.7])
    column = _column(streaming)
    default = validation.PreparedValidationRun(column, parameters).run(parameters)
    assert not default["success"] and "prediction" not in default
    assert "mask_continuation" not in default

    prepared = validation.PreparedValidationRun(column, parameters, mask_continuation=True)
    snapshots = []
    result = prepared.run(parameters, on_state=snapshots.append)
    assert result["success"] and result["state"]["converged"]
    history = result["mask_continuation"]["stages"]
    assert result["mask_continuation"]["stop_reason"] == "converged"
    assert len(history) > 1 and not history[0]["state"]["converged"]
    assert len(snapshots) == len(history) + 1
    assert snapshots[0]["mask_continuation"]["stop_reason"] == "running"
    assert len(snapshots[0]["mask_continuation"]["stages"]) == 1
    assert snapshots[-1]["state"]["converged"]
    np.testing.assert_allclose(result["prediction"], [2. / 3.], atol=1e-9)

    failed = prepared.run(jnp.array([-1., .6, -.7]))
    assert not failed["success"] and "prediction" not in failed
    assert failed["mask_continuation"]["stop_reason"] == "invalid_state"
    assert len(failed["mask_continuation"]["stages"]) == 1
    assert failed["mask_continuation"]["stages"][0]["initial_temperature_K"] == [2000.]
    if streaming:
        assert len(column.stream_calls) == 1
        assert result["resolved_streaming"]["test_fixture"]


@pytest.mark.parametrize("suite", ["vertical", "depth"])
@pytest.mark.parametrize("streaming", [False, True])
def test_native_pressure_variants_align_seed_and_mask_without_mutating_base(
        validation, monkeypatch, suite, streaming):
    from _rce_wasp18b import ColumnSettings
    from _rce_wasp18b_streaming import MetadataOnlyDiffgrid, StreamingResolvedPlan
    from exojax.opacity import OpaDiffgrid

    boundaries = np.geomspace(1e-6, 100., 33)
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = 2000. * (pressure / boundaries[-1])**.01
    mask = np.arange(32) % 3 == 0
    nu = jnp.array([2000., 3000., 4000.])
    teacher = SimpleNamespace(
        nu_grid=nu, xsmatrix=lambda t, p: (1e-24 * p[:, None]
                                           * (t[:, None] / 2000.) * (nu[None, :] / 2000.)),
    )
    opacity = OpaDiffgrid(teacher, [1500., 2500., 4500.], pressure)
    if streaming:
        opacity = MetadataOnlyDiffgrid(dict(
            nu_grid=nu, pressure_grid=np.asarray(opacity.diffgrid_info.pressure_grid),
            temperature_grid=opacity.diffgrid_info.temperature_grid,
            inverse_temperature_grid=opacity.diffgrid_info.inverse_temperature_grid,
            log_cross_section_floor=opacity.diffgrid_info.log_cross_section_floor,
        ))
    base = SimpleNamespace(
        boundaries=boundaries, pressure=pressure, lbl={"test": opacity},
        solver_arguments=(pressure, boundaries, temperature, 2000.),
        solver_options={"convective_mask_initial": mask, "flux_atol": .1},
        settings=ColumnSettings(1000., 200.), isotope_fractions={"test": 1.},
        observation=None, stellar=None, ckd=None, cia=None,
    )
    if streaming:
        base.resolved_stream = StreamingResolvedPlan(
            ("test",), (), np.asarray(nu), (), {"original": True}, pressure)

    def construct(*args, **kwargs):
        chosen = np.asarray(args[5])
        return SimpleNamespace(
            lbl=args[3], boundaries=chosen, pressure=np.sqrt(chosen[:-1] * chosen[1:]),
            settings=args[6], solver_options=kwargs["solver_options"],
            solver_arguments=(None, chosen, kwargs["temperature_initial"],
                              kwargs["bottom_temperature_initial"]),
        )

    monkeypatch.setattr(validation, "Wasp18bColumn", construct)
    for case in validation.native_pressure_grids(boundaries, suite):
        case.update(initial_temperature_K=None, settings={})
        indices = np.asarray(case["source_pressure_indices"])
        variant = validation.variant_column(base, case)
        np.testing.assert_allclose(variant.pressure, pressure[indices], rtol=1e-12)
        np.testing.assert_array_equal(variant.solver_arguments[2], temperature[indices])
        np.testing.assert_array_equal(variant.solver_options["convective_mask_initial"], mask[indices])
        np.testing.assert_allclose(variant.solver_arguments[3],
                                   2000. * (variant.boundaries[-1] / boundaries[-1])**.01,
                                   rtol=2e-15)
        chosen_info = variant.lbl["test"].diffgrid_info
        np.testing.assert_array_equal(chosen_info.pressure_grid, pressure[indices])
        if streaming:
            np.testing.assert_array_equal(variant.resolved_stream.pressure_indices, indices)
        else:
            original_info = opacity.diffgrid_info
            np.testing.assert_array_equal(chosen_info.log_cross_section_grid,
                                           original_info.log_cross_section_grid[indices])
            np.testing.assert_array_equal(chosen_info.log_cross_section_derivative_grid,
                                           original_info.log_cross_section_derivative_grid[indices])
        if len(indices) == 32:
            assert variant is base
        else:
            assert variant.solver_options is not base.solver_options
    np.testing.assert_array_equal(base.solver_options["convective_mask_initial"], mask)
    np.testing.assert_array_equal(base.lbl["test"].diffgrid_info.pressure_grid, pressure)
    if streaming:
        np.testing.assert_array_equal(base.resolved_stream.pressure_indices, np.arange(32))
        assert base.resolved_stream.provenance == {"original": True}
