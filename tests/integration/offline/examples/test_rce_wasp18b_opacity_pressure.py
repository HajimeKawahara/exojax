"""Independent opacity-node refinements keep RT geometry and seeds fixed."""

from copy import copy
from dataclasses import replace
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.ckd.contracts import CKDTableInfo
from tests.integration.offline.examples.test_rce_wasp18b_streaming import modules


def native_column(modules, monkeypatch):
    validation = importlib.import_module("rce_wasp18b_validation")
    boundaries = np.geomspace(1e-5, 100., 33)
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = np.linspace(1500., 4500., 16)
    # Nonlinear log P makes held-out node restoration a measurable change.
    values = (-55. + .02*np.log(pressure)[None, :, None, None]**2
              + temperature[:, None, None, None]/10000. + np.zeros((16, 32, 2, 2)))
    ckd = OpaCKD.__new__(OpaCKD)
    ckd.ready = True
    ckd.ckd_info = CKDTableInfo(jnp.asarray(values), jnp.array([.25, .75]),
        jnp.array([.5, .5]), jnp.asarray(temperature), jnp.asarray(pressure),
        jnp.array([2500., 3500.]), jnp.array([[2000., 3000.], [3000., 4000.]]))
    metadata = modules.stream.MetadataOnlyDiffgrid(dict(nu_grid=np.array([2000., 3000.]),
        pressure_grid=pressure, temperature_grid=temperature[::-1],
        inverse_temperature_grid=1/temperature[::-1], log_cross_section_floor=np.log(1e-99)))
    seed = np.linspace(2000., 3000., 32)
    mask = np.arange(32) % 3 == 0
    base = SimpleNamespace(boundaries=boundaries, pressure=pressure, lbl={"test": metadata},
        ckd={"test": ckd}, solver_arguments=(pressure, boundaries, seed, 3200.),
        solver_options={"convective_mask_initial": mask},
        settings=modules.column.ColumnSettings(19000., 200.), isotope_fractions={"test": 1.},
        observation=None, stellar=None, cia=None)
    base.resolved_stream = modules.stream.StreamingResolvedPlan(
        ("test",), (), np.array([2000., 3000.]), (), {"file_sha256": {"original": "pin"}},
        pressure, temperature[::-1])
    def construct(*args, **kwargs):
        result = copy(base)
        result.ckd, result.lbl = args[2:4]
        result.solver_arguments = (pressure, args[5], kwargs["temperature_initial"],
                                   kwargs["bottom_temperature_initial"])
        result.solver_options = kwargs["solver_options"]
        return result
    monkeypatch.setattr(validation, "Wasp18bColumn", construct)
    config = dict(parameters=[0., .6, np.log(.5)], prior_lower=[-1., .2, np.log(.1)],
                  prior_upper=[1., 1.2, 0.], column={"pressure_boundaries_bar": boundaries})
    return validation, base, config


def test_opacity_pressure_cases_native_slices_and_held_out_values(modules, monkeypatch):
    validation, base, config = native_column(modules, monkeypatch)
    ckd = base.ckd["test"]
    pressure, temperature = map(np.asarray, (ckd.ckd_info.P_grid, ckd.ckd_info.T_grid))
    values = np.asarray(ckd.ckd_info.log_kggrid)
    seed, mask = base.solver_arguments[2], base.solver_options["convective_mask_initial"]
    cases = validation.make_cases(config, "opacity_pressure")
    assert [len(case["opacity_pressure_indices"]) for case in cases] == [9, 17, 32]
    assert set(cases[0]["opacity_pressure_indices"]) < set(cases[1]["opacity_pressure_indices"])
    variants = [validation.variant_column(base, case) for case in cases]
    assert variants[-1] is base
    assert not np.array_equal(variants[0].ckd["test"].ckd_info.P_grid,
                              variants[1].ckd["test"].ckd_info.P_grid)
    for variant, case in zip(variants, cases):
        indices = np.asarray(case["opacity_pressure_indices"])
        info = variant.ckd["test"].ckd_info
        np.testing.assert_array_equal(info.P_grid, pressure[indices])
        np.testing.assert_array_equal(info.T_grid, temperature)
        np.testing.assert_array_equal(info.log_kggrid, values[:, indices])
        np.testing.assert_array_equal(variant.pressure, pressure)
        np.testing.assert_array_equal(variant.solver_arguments[2], seed)
        np.testing.assert_array_equal(variant.solver_options["convective_mask_initial"], mask)
        assert variant.solver_arguments[3] == 3200.
        actual = np.asarray(variant.ckd["test"].xstensor_ckd(
            jnp.full(32, temperature[3]), jnp.asarray(pressure)))
        expected = np.exp(np.interp(np.log(pressure), np.log(pressure[indices]), values[3, indices, 0, 0]))
        np.testing.assert_allclose(actual[:, 0, 0], expected, rtol=2e-14)
        np.testing.assert_allclose(actual[indices, 0, 0], np.exp(values[3, indices, 0, 0]), rtol=2e-14)
        variant.lbl["test"].check_pressure_grid(pressure)
        if variant is not base:
            assert variant.resolved_stream.provenance["file_sha256"] == {"original": "pin"}
            assert variant.resolved_stream.provenance["pressure_resampling"]["source_archive_indices"] == indices.tolist()
    np.testing.assert_array_equal(ckd.ckd_info.log_kggrid, values)
    assert base.resolved_stream.target_pressure is None
    for bad in ([1, 31], [0, 30], [0, 4, 4, 31], [0., 31.]):
        with pytest.raises(ValueError, match="both opacity endpoints"):
            validation.variant_column(base, dict(cases[0], opacity_pressure_indices=bad))
    bad_ckd = copy(ckd)
    bad_ckd.ckd_info = replace(ckd.ckd_info, P_grid=ckd.ckd_info.P_grid.at[3].multiply(1.01))
    changed = copy(base)
    changed.ckd = {"test": bad_ckd}
    with pytest.raises(ValueError, match="match exactly"):
        validation.variant_column(changed, cases[0])
    changed.resolved_stream = None
    with pytest.raises(ValueError, match="--stream-resolved"):
        validation.variant_column(changed, cases[0])


@pytest.mark.parametrize("suite", ["opacity_pressure", "opacity_temperature"])
def test_opacity_pressure_suite_requires_streaming_before_loading(modules, tmp_path, suite):
    validation = importlib.import_module("rce_wasp18b_validation")
    path = tmp_path / "rejected.json"
    assert validation.main(["--suite", suite, "--manifest", "absent.json",
                            "--observation", "absent.ecsv", "--output", str(path)]) == 1
    assert "requires --stream-resolved" in path.read_text()


def test_temperature_cases_match_physical_knots_and_keep_domain(modules, monkeypatch):
    validation, base, config = native_column(modules, monkeypatch)
    original = base.ckd["test"].ckd_info
    original_lbl = base.lbl["test"].diffgrid_info
    cases = validation.make_cases(config, "opacity_temperature")
    assert [len(case["opacity_temperature_indices"]) for case in cases] == [5, 9, 16]
    for case in cases:
        variant = validation.variant_column(base, case)
        indices = np.asarray(case["opacity_temperature_indices"])
        reverse = 15 - indices[::-1]
        ckd, lbl = variant.ckd["test"].ckd_info, variant.lbl["test"].diffgrid_info
        np.testing.assert_array_equal(ckd.T_grid, original.T_grid[reverse])
        np.testing.assert_array_equal(lbl.temperature_grid, original_lbl.temperature_grid[indices])
        np.testing.assert_array_equal(ckd.T_grid, lbl.temperature_grid[::-1])
        np.testing.assert_array_equal(np.asarray(ckd.T_grid)[[0, -1]], np.asarray(original.T_grid)[[0, -1]])
        np.testing.assert_array_equal(ckd.P_grid, original.P_grid)
        np.testing.assert_array_equal(ckd.log_kggrid, original.log_kggrid[reverse])
        np.testing.assert_array_equal(lbl.inverse_temperature_grid, original_lbl.inverse_temperature_grid[indices])
        np.testing.assert_array_equal(variant.boundaries, base.boundaries)
        np.testing.assert_array_equal(variant.solver_arguments[2], base.solver_arguments[2])
        np.testing.assert_array_equal(variant.solver_options["convective_mask_initial"],
                                      base.solver_options["convective_mask_initial"])
        assert variant.solver_arguments[3] == base.solver_arguments[3]
        if len(indices) == 16:
            assert variant is base
        else:
            assert variant.resolved_stream.provenance["temperature_subset"]["source_archive_indices"] == indices.tolist()
    assert base.resolved_stream.temperature_indices is None
    for bad in ([1, 15], [0, 14], [0, 8, 8, 15], [0., 15.], np.array([0, 15, 8, 15], dtype=np.uint64)):
        with pytest.raises(ValueError, match="temperature endpoints"):
            validation.variant_column(base, dict(cases[0], opacity_temperature_indices=bad))
    changed = copy(base)
    changed.ckd = {"test": copy(base.ckd["test"])}
    changed.ckd["test"].ckd_info = replace(original, T_grid=original.T_grid.at[3].add(.01))
    with pytest.raises(ValueError, match="match exactly"):
        validation.variant_column(changed, cases[0])
