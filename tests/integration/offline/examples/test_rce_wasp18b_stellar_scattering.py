"""Exercise the opt-in stellar transfer through the actual P7 column paths."""

from copy import copy
from dataclasses import replace
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_stellar_scattering_setting_requires_bool(modules, value):
    assert not modules.column.ColumnSettings(19000., 200.).stellar_scattering
    with pytest.raises(ValueError, match="stellar_scattering must be boolean"):
        modules.column.ColumnSettings(19000., 200., stellar_scattering=value)


def test_zero_scattering_preserves_all_thermal_and_stellar_boundaries(modules, tmp_path):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    # Retain molecular absorption and thermal emission, but remove scatterers.
    chemistry = chemistry._replace(x=chemistry.x.at[:, 2:].set(0.))
    arguments = (jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.7]), chemistry)
    reference = column.fluxes(*arguments, resolved=True)
    column.settings = replace(column.settings, stellar_scattering=True)
    actual = column.fluxes(*arguments, resolved=True)
    for left, right in zip(actual, reference):
        np.testing.assert_array_equal(left, right)


def test_constant_spectral_bands_and_resolved_transfer_agree(modules, tmp_path):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=True)
    # Every band represents the same single frequency for its source and
    # continuum; opacity is constant in g. This isolates CKD broadcasting.
    column.band_nodes, column.band_weights = column.nu[:, None], jnp.ones(1)
    column.band_widths, column.g_weights = jnp.ones(column.nu.size), jnp.array([.3, .7])
    column.star_bands = column.star_resolved
    depth = jnp.ones((2, column.nu.size)) * jnp.array([.1, 2.])[:, None]
    column.optical_depth = lambda t, c, resolved=False, resolved_tables=None: (
        depth if resolved else jnp.broadcast_to(depth[:, None, :], (2, 2, column.nu.size)))
    arguments = (jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.7]), chemistry)
    resolved = column.fluxes(*arguments, resolved=True)
    bands = column.fluxes(*arguments)
    for band, line in zip(bands, resolved):
        np.testing.assert_allclose(band, np.broadcast_to(np.asarray(line)[:, None, :], band.shape),
                                   rtol=3e-14, atol=1e-12)


def test_streamed_stellar_scattering_matches_full_and_observation(modules, tmp_path):
    manifest, column, chemistry = fixture_manifest(modules, tmp_path)
    arguments = (jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.7]), chemistry)
    absorption = column.fluxes(*arguments, resolved=True).upward[0]
    column.settings = replace(column.settings, stellar_scattering=True)
    expected = column.fluxes(*arguments, resolved=True).upward[0]
    assert np.all(np.asarray(expected) > np.asarray(absorption))
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    streamed = copy(column)
    streamed.lbl = metadata
    actual = plan.top_flux(streamed, *arguments)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.)
    np.testing.assert_allclose(plan.prediction(streamed, actual, 300.),
                               column.spectrum(*arguments, resolving_power=300.), rtol=2e-12)
    assert plan.provenance["all_chunk_coefficients_verified"]


def test_temperature_chemistry_and_irradiation_jvp_matches_finite_difference(modules, tmp_path):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    absorption_column = copy(column)
    column.settings = replace(column.settings, stellar_scattering=True)

    def evaluate(vector):
        temperature, bottom, parameters = vector[:2], vector[2], vector[3:]
        # Smooth controlled chemical dependence checks that traced scattering
        # retains composition derivatives, independently of equilibrium AD.
        hydrogen = .2 + .01 * jnp.append(temperature, bottom) / 3000. + .02 * parameters[0]
        x = chemistry.x.at[:, 4].set(hydrogen).at[:, 2].set(.8 - hydrogen)
        state = chemistry._replace(x=x)
        full = column.fluxes(temperature, bottom, parameters, state, resolved=True)
        reference = absorption_column.fluxes(temperature, bottom, parameters, state, resolved=True)
        return jnp.stack((full.net, full.upward - reference.upward))

    point = jnp.array([2233., 2788., 3111., 0., .6, -.7])
    direction = jnp.array([.8, -.4, .2, .03, -.02, .1])
    function = jax.jit(evaluate)
    _, actual = jax.jvp(function, (point,), (direction,))
    step = .01
    expected = (function(point + step * direction) - function(point - step * direction)) / (2 * step)
    scale = np.max(np.abs(actual), axis=(1, 2), keepdims=True)
    assert np.all(scale > 0.) and np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=0., atol=3e-6)


@pytest.mark.parametrize("enabled,label", [(True, "omitted"), (False, "stellar_two_stream")])
def test_manifest_requires_matching_scattering_declaration(modules, enabled, label):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest = SimpleNamespace(config=dict(
        opacity=dict(mapping={}, ckd={}, lbl={}, scattering=label,
                     omitted_absorbers=[], broadening_limitations=[]),
        column=dict(settings=dict(stellar_scattering=enabled)),
    ))
    with pytest.raises(ValueError, match="requires opacity.scattering"):
        forward.prepare_column(manifest, None)


@pytest.mark.parametrize("scatterers", [False, True])
def test_runner_reports_total_direct_and_diffuse_stellar_leakage(modules, tmp_path, scatterers):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest, column, state = fixture_manifest(modules, tmp_path)
    if not scatterers:
        state = state._replace(x=state.x.at[:, 2:].set(0.))
    species = column.chemistry.species

    class FixedChemistry:
        def __call__(self, *args):
            return state

    chemistry = FixedChemistry()
    chemistry.species = species
    column.chemistry = chemistry
    column.settings = replace(column.settings, stellar_scattering=True)
    column.solver_arguments = (column.pressure, column.boundaries, jnp.array([2300., 2700.]), 3000.)
    column.internal_flux = modules.column.SIGMA_SB * column.settings.internal_temperature**4
    column.observation = SimpleNamespace(uncertainty=jnp.full(3, 1e-5))
    column.band_nodes, column.band_weights = column.nu[:, None], jnp.ones(1)
    column.band_widths, column.g_weights = jnp.ones(column.nu.size), jnp.array([.3, .7])
    column.star_bands = column.star_resolved
    column.ckd = {name: SimpleNamespace(nu_bands=column.nu,
        xstensor_ckd=lambda t, p, own=opacity: jnp.repeat(own.xsmatrix(t)[:, None, :], 2, axis=1))
        for name, opacity in column.lbl.items()}
    _, column.resolved_stream = modules.stream.load_streaming_opacities(manifest)
    report = forward.run_column(column, jnp.array([0., .6, -.7]), fixed_temperature=True)
    assert report["success"]
    stellar = report["stellar_scattering"]["stellar_flux_audit"]
    flux = stellar["boundary_flux_cgs"]
    expected = flux["downward"][-1] + flux["incident"][-1]
    assert stellar["bottom_reaching_fraction"] == pytest.approx(expected / flux["incident"][0])
    assert stellar["bottom_total_stellar_over_internal_flux"] == pytest.approx(expected / column.internal_flux)
    if scatterers:
        assert stellar["bottom_reaching_fraction"] > float(report["bottom_incident_fraction"])
    else:
        assert stellar["bottom_reaching_fraction"] == float(report["bottom_incident_fraction"])
    assert report["stellar_scattering"]["thermal_photon_scattering_included"] is False
