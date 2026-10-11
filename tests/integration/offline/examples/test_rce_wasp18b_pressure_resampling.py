"""Validate interior layer refinement of the pinned opacity interpolation model."""

from copy import copy
from dataclasses import replace
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.diffgrid.core import cross_section_matrix, interpolate_log_cross_section
from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


def test_pressure_mapping_native_identity_invalids_and_hermite_jvp(modules):
    source = np.geomspace(.001, 100., 5)
    target = np.sort(np.r_[source, np.sqrt(source[:-1] * source[1:])])
    rng = np.random.default_rng(11)
    values = jnp.asarray(-55 + rng.normal(size=(5, 4, 7)))
    slopes = jnp.asarray(rng.normal(size=(5, 4, 7)) * 3000)
    mapping = modules.stream.pressure_interpolation(source, source)
    for array in (values, slopes):
        np.testing.assert_array_equal(modules.stream.resample_pressure_values(array, mapping), array)
    for invalid in ([.0001, 1.], [.1, 101.], [1.], [1., 1.], [1., .1], [np.nan, 1.], [-1., 1.]):
        with pytest.raises(ValueError):
            modules.stream.pressure_interpolation(source, invalid)
    mapping = modules.stream.pressure_interpolation(source, target)
    lower, upper, weight = mapping
    interpolated = [modules.stream.resample_pressure_values(array, mapping) for array in (values, slopes)]
    inverse = jnp.array([1/4500., 1/3500., 1/2500., 1/1500.])
    temperature = jnp.linspace(1643., 4307., len(target))
    def actual(t):
        return interpolate_log_cross_section(t, inverse, *interpolated)
    def expected(t):
        low = interpolate_log_cross_section(t, inverse, values[lower], slopes[lower])
        high = interpolate_log_cross_section(t, inverse, values[upper], slopes[upper])
        return low + weight[:, None] * (high - low)
    for actual_value, expected_value in zip(jax.jvp(actual, (temperature,), (jnp.ones_like(temperature),)),
                                            jax.jvp(expected, (temperature,), (jnp.ones_like(temperature),))):
        np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=2e-14)
    # A floor crossing must be clipped after pressure/Hermite interpolation.
    floor_values = jnp.array([[[-80., -60.]] * 4, [[-40., -60.]] * 4])
    floor_mapping = modules.stream.pressure_interpolation([1., 100.], [10., 100.])
    f = modules.stream.resample_pressure_values(floor_values, floor_mapping)
    xs = cross_section_matrix(jnp.array([3000., 3000.]), inverse, f, jnp.zeros_like(f), jnp.array(-65.))
    np.testing.assert_allclose(xs[0], np.exp([-60., -60.]), rtol=2e-14)


@pytest.mark.parametrize("source_indices", [None, [0, 2, 3]])
def test_resampled_stream_matches_full_transfer_and_global_p1(modules, tmp_path, source_indices):
    manifest, column, chemistry = fixture_manifest(modules, tmp_path, boundaries=np.geomspace(.0001, 10., 5))
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    original = np.asarray(column.boundaries)
    boundaries = np.sort(np.r_[original, np.sqrt(original[1:-2] * original[2:-1])])
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    selected = np.arange(len(column.pressure)) if source_indices is None else np.asarray(source_indices)
    refined = plan.pressure_subset(selected).pressure_resample(pressure)
    full = copy(column)
    full.pressure, full.boundaries, full.lbl = jnp.asarray(pressure), jnp.asarray(boundaries), {}
    for name, opacity in column.lbl.items():
        info = opacity.diffgrid_info
        # Independent host interpolation of each coefficient along log P.
        def interpolate(array):
            flat = np.asarray(array)[selected].reshape((len(selected), -1))
            result = np.stack([np.interp(np.log(pressure), np.log(np.asarray(column.pressure)[selected]), row) for row in flat.T], axis=-1)
            return jnp.asarray(result.reshape((len(pressure),) + array.shape[1:]))
        own = copy(opacity)
        own._apply_diffgrid_info(replace(info, pressure_grid=jnp.asarray(pressure),
            log_cross_section_grid=interpolate(info.log_cross_section_grid),
            log_cross_section_derivative_grid=interpolate(info.log_cross_section_derivative_grid)))
        full.lbl[name] = own
    streamed = copy(full)
    streamed.lbl = {name: opacity.pressure_subset(selected).pressure_resample(pressure)
                    for name, opacity in metadata.items()}
    for opacity in streamed.lbl.values():
        opacity.check_pressure_grid(pressure)
        assert not hasattr(opacity.diffgrid_info, "log_cross_section_grid")
    chemistry = jax.tree.map(lambda value: jnp.repeat(value[:1], len(boundaries), axis=0), chemistry)
    arguments = (jnp.linspace(1800., 3300., len(pressure)), 3500., jnp.array([0., .6, np.log(.5)]), chemistry)
    expected = full.fluxes(*arguments, resolved=True).upward[0]
    actual = refined.top_flux(streamed, *arguments)
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    for resolution in (None, 300., np.array([200., 300., 400., 600.])):
        np.testing.assert_allclose(refined.prediction(streamed, actual, resolution),
                                   full.spectrum(*arguments, resolving_power=resolution), rtol=2e-12)
    assert refined.provenance["file_sha256"] == plan.provenance["file_sha256"]
    assert refined.provenance["pressure_resampling"]["source_archive_indices"] == selected.tolist()
    assert refined.provenance["all_chunk_coefficients_verified"]
    assert "pressure_resampling" not in plan.provenance and plan.target_pressure is None
    np.testing.assert_array_equal(refined.pressure_subset([0, 2, 5]).target_pressure, pressure[[0, 2, 5]])
    with pytest.raises(ValueError, match="pressure selection"):
        plan.top_flux(streamed, *arguments)
    refined.chunks[0]["H2O1"].path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        refined.top_flux(streamed, *arguments)


def test_inner_grids_seeds_masks_and_common_boundaries(modules, monkeypatch):
    validation = importlib.import_module("rce_wasp18b_validation")
    boundaries = np.geomspace(1e-5, 100., 33)
    source = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = 2500. * (source / 1.)**.015
    mask = np.arange(32) % 3 == 0
    descriptor = modules.stream.MetadataOnlyDiffgrid(dict(nu_grid=np.array([2000., 3000.]),
        pressure_grid=source, temperature_grid=np.array([4500., 1500.]),
        inverse_temperature_grid=np.array([1/4500., 1/1500.]), log_cross_section_floor=np.log(1e-99)))
    base = SimpleNamespace(boundaries=boundaries, pressure=source, lbl={"test": descriptor},
        solver_arguments=(source, boundaries, temperature, 3100.),
        solver_options={"convective_mask_initial": mask, "flux_atol": .1},
        settings=modules.column.ColumnSettings(19000., 200.), isotope_fractions={"test": 1.},
        observation=None, stellar=None, ckd={"unchanged": object()}, cia=None)
    base.resolved_stream = modules.stream.StreamingResolvedPlan(("test",), (), np.array([2000.,3000.]), (), {}, source)
    def construct(*args, **kwargs):
        b = np.asarray(args[5])
        return SimpleNamespace(lbl=args[3], ckd=args[2], boundaries=b, pressure=np.sqrt(b[:-1]*b[1:]),
            solver_options=kwargs["solver_options"], settings=args[6],
            solver_arguments=(None, b, kwargs["temperature_initial"], kwargs["bottom_temperature_initial"]))
    monkeypatch.setattr(validation, "Wasp18bColumn", construct)
    cases = validation.inner_refinement_grids(boundaries)
    assert [len(case["pressure_centers_bar"]) for case in cases] == [32, 62, 122]
    variants = []
    for case in cases:
        case.update(initial_temperature_K=None, settings={})
        variant = validation.variant_column(base, case)
        variants.append(variant)
        assert variant.ckd is base.ckd
        assert np.isin(boundaries, variant.boundaries).all()
        np.testing.assert_array_equal(variant.pressure[[0,-1]], source[[0,-1]])
        assert variant.pressure.min() >= source.min() and variant.pressure.max() <= source.max()
        np.testing.assert_allclose(variant.solver_arguments[2], 2500.*variant.pressure**.015, rtol=2e-15)
        assert variant.solver_arguments[3] == 3100.
        old_nodes, nodes = np.append(source, boundaries[-1]), np.append(variant.pressure, boundaries[-1])
        midpoint = .5*(np.log(nodes[:-1])+np.log(nodes[1:]))
        index = np.searchsorted(np.log(old_nodes), midpoint, side="right")-1
        np.testing.assert_array_equal(variant.solver_options["convective_mask_initial"], mask[index])
        assert variant.solver_options["convective_mask_initial"][-1] == mask[-1]
        if len(variant.pressure) != 32:
            assert variant.resolved_stream is not base.resolved_stream
    assert variants[0] is base and base.resolved_stream.target_pressure is None
    np.testing.assert_array_equal(base.solver_options["convective_mask_initial"], mask)
    def report(variant, case, offset):
        flux = np.log(variant.boundaries)+offset
        return dict(success=True, prediction=[.01+offset*1e-6], uncertainty=[.001], parameters=[0.,.6,-.7],
            mode="rce", observation_sha256={}, row_indices=[0], physical_inputs={},
            pressure_bar=variant.pressure, pressure_boundaries_bar=variant.boundaries,
            temperature_K=variant.solver_arguments[2], case=case,
            broadband_flux={key:flux for key in ("upward","downward","incident","net")},
            state={"convective_flux":-flux})
    compared = validation.compare_inner_runs(report(variants[2],cases[2],1.), report(variants[1],cases[1],0.))
    for key in ("upward", "downward", "incident", "net"):
        np.testing.assert_allclose(compared["common_boundary_flux"]["change_cgs"][key],1.)
    np.testing.assert_array_equal(compared["common_boundary_flux"]["change_cgs"]["total_net"], np.zeros(33))
    assert len(compared["common_boundary_flux"]["pressure_bar"]) == 33
    base.resolved_stream = None
    with pytest.raises(ValueError, match="--stream-resolved"):
        validation.variant_column(base, cases[1])


def test_inner_suite_requires_streaming_before_loading(modules, tmp_path):
    validation = importlib.import_module("rce_wasp18b_validation")
    path = tmp_path / "rejected.json"
    assert validation.main(["--suite", "inner_refinement", "--manifest", "absent.json",
                            "--observation", "absent.ecsv", "--output", str(path)]) == 1
    assert "requires --stream-resolved" in path.read_text()
