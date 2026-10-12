"""Check common spectral intervals, seam integration and verified boundary RT."""

from copy import copy
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    filename = Path(__file__).with_name("test_rce_wasp18b_streaming.py")
    spec = importlib.util.spec_from_file_location("flux_audit_streaming_fixture", filename)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    yield SimpleNamespace(audit=importlib.import_module("rce_wasp18b_flux_audit"),
        stream=importlib.import_module("_rce_wasp18b_streaming"),
        column=importlib.import_module("_rce_wasp18b"),
        assets=importlib.import_module("_rce_wasp18b_assets"), fixture=fixtures.fixture_manifest,
        teacher=fixtures.Teacher)
    jax.config.update("jax_enable_x64", previous)


def test_common_interval_excludes_partial_bands(modules):
    edges = np.array([[3490., 3600.], [3600., 3800.], [3800., 4200.],
                      [4200., 4400.], [4400., 4550.]])
    indices, interval = modules.audit.common_band_interval(edges, np.linspace(3500., 4500., 65))
    np.testing.assert_array_equal(indices, [1, 2, 3])
    assert interval == (3600., 4400.)
    with pytest.raises(ValueError, match="No complete"):
        modules.audit.common_band_interval(edges, [4401., 4549.])
    edges[2, 0] += 1.
    with pytest.raises(ValueError, match="contiguous"):
        modules.audit.common_band_interval(edges, [3500., 4500.])


def test_trapezoid_outer_clipping_and_shared_endpoint(modules):
    nu = np.array([1., 2., 4., 7.])
    values = np.stack((nu * 2 + 3, nu * -5 + 1))
    lower, upper = 1.3, 6.2
    expected = np.array([upper**2 + 3*upper - lower**2 - 3*lower,
                         -2.5*upper**2 + upper + 2.5*lower**2 - lower])
    whole = modules.audit.integrate_interval(nu, values, lower, upper)
    split = (modules.audit.integrate_interval(nu[:3], values[:, :3], lower, 4.)
             + modules.audit.integrate_interval(nu[2:], values[:, 2:], 4., upper))
    np.testing.assert_allclose(whole, expected, rtol=1e-15)
    np.testing.assert_allclose(split, whole, rtol=1e-15)
    with pytest.raises(ValueError, match="inside"):
        modules.audit.integrate_interval(nu, values, .9, upper)


def test_streamed_all_boundary_fluxes_match_full_reference(modules, tmp_path):
    manifest, column, chemistry = modules.fixture(modules, tmp_path)
    descriptors, plan = modules.stream.load_streaming_opacities(manifest)
    streamed = copy(column)
    streamed.lbl, streamed.resolved_stream = descriptors, plan
    temperature, bottom = jnp.array([2300., 2700.]), 3000.
    parameters = jnp.array([0., .6, np.log(.5)])
    interval = (3555., 4455.)
    full = column.fluxes(temperature, bottom, parameters, chemistry, resolved=True)
    actual, details = modules.audit.resolved_boundary_integrals(
        streamed, temperature, bottom, parameters, chemistry, interval)
    for name, values in full._asdict().items():
        expected = modules.audit.integrate_interval(column.nu, values, *interval)
        np.testing.assert_allclose(actual[name], expected, rtol=3e-12, atol=1e-7)
    assert details["chunk_count_used"] == 2
    # The fixture deliberately changes the second archive's duplicate endpoint.
    assert details["maximum_raw_seam_flux_density_difference_cgs"]["upward"] > 0
    assert plan.provenance["all_chunk_coefficients_verified"]
    np.testing.assert_array_equal(streamed.nu, column.nu)
    comparison = modules.audit.compare_boundary_integrals(actual, actual)
    assert all(row["max_absolute_difference_over_interval_incident"] == 0.
               for row in comparison["fluxes"].values())


def test_streaming_rejects_changed_coefficient_archive(modules, tmp_path):
    manifest, column, chemistry = modules.fixture(modules, tmp_path)
    column.lbl, column.resolved_stream = modules.stream.load_streaming_opacities(manifest)
    column.resolved_stream.chunks[0]["H2O1"].path.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        modules.audit.resolved_boundary_integrals(column, jnp.array([2300., 2700.]),
            3000., jnp.array([0., .6, np.log(.5)]), chemistry, (3555., 4455.))
    assert not column.resolved_stream.provenance["all_chunk_coefficients_verified"]


def test_selected_ckd_flux_matches_full_spectral_selection(modules, tmp_path):
    from _rce_broadband import make_band_quadrature, band_stellar_flux, integrate_broadband_fluxes, Fluxes
    from exojax.opacity import OpaCKD

    _, column, chemistry = modules.fixture(modules, tmp_path)
    edges = jnp.array([[3500., 3750.], [3750., 4000.], [4000., 4250.], [4250., 4500.]])
    column.ckd = {}
    for name, scale in zip(column.lbl, (1., 1.5)):
        opa = OpaCKD(modules.teacher(np.asarray(column.nu), scale), Ng=4, band_width=250.)
        opa.band_edges, opa.nu_bands = edges, jnp.mean(edges, axis=1)
        opa.precompute_tables(np.array([1500., 2500., 3500., 4500.]), np.asarray(column.pressure))
        column.ckd[name] = opa
    column.band_edges, column.g_weights = edges, opa.ckd_info.weights
    column.band_nodes, column.band_weights = make_band_quadrature(edges)
    column.band_widths = jnp.diff(edges, axis=1).ravel()
    column.star_bands = band_stellar_flux(column.stellar.nu_cm1, column.stellar.surface_flux_per_cm1, edges)
    temperature, bottom, parameters = jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, np.log(.5)])
    indices = np.array([1, 2])
    full = column.fluxes(temperature, bottom, parameters, chemistry)
    expected = integrate_broadband_fluxes(Fluxes(*(value[..., indices] for value in full)),
                                         column.g_weights, column.band_widths[indices])
    local = modules.audit.select_ckd_bands(column, indices)
    actual = integrate_broadband_fluxes(local.fluxes(temperature, bottom, parameters, chemistry),
                                       local.g_weights, local.band_widths)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=3e-12, atol=1e-7)
    assert column.band_edges.shape == (4, 2)
    assert all(value.ckd_info.log_kggrid.shape[-1] == 4 for value in column.ckd.values())
