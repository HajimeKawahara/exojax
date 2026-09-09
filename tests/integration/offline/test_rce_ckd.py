"""Offline RCE integration with temperature-dependent synthetic CKD tables."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.atm.rce import reconstruct_boundary_temperature, solve_rce
from exojax.opacity import OpaCKD
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.core import gauss_legendre_grid
from exojax.rt.flux import (
    integrate_ckd_flux,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.layeropacity import layer_optical_depth, layer_optical_depth_ckd
from exojax.rt.planck import piBarr


def _synthetic_ckd(band_edges):
    """Represent a smooth, perfectly correlated spectrum without line data."""
    ggrid, weights = gauss_legendre_grid(8)
    temperature_grid = jnp.array([250.0, 2000.0])
    pressure_grid = jnp.array([1.0e-3, 100.0])
    log_k = (
        jnp.log(2.0e-24)
        + (temperature_grid[:, None, None, None] - 700.0) / 1400.0
        + jnp.log(pressure_grid)[None, :, None, None]
        + 2.0 * ggrid[None, None, :, None]
        + jnp.zeros(len(band_edges))
    )
    opa = OpaCKD.load_only()
    opa.Ng = len(ggrid)
    opa.band_edges = jnp.asarray(band_edges)
    opa.nu_bands = jnp.mean(opa.band_edges, axis=1)
    opa.ckd_info = CKDTableInfo(
        log_kggrid=log_k,
        ggrid=ggrid,
        weights=weights,
        T_grid=temperature_grid,
        P_grid=pressure_grid,
        nu_bands=opa.nu_bands,
        band_edges=opa.band_edges,
    )
    opa.ready = True
    return opa


def test_rce_with_temperature_dependent_ckd():
    edges = np.linspace(1.0, 12000.0, 25)
    opa = _synthetic_ckd(np.column_stack((edges[:-1], edges[1:])))
    boundaries = np.geomspace(1.0e-2, 10.0, 9)
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = jnp.linspace(450.0, 850.0, len(pressure))
    bottom_temperature = 950.0
    mus, angular_weights = gauss_legendre_grid(4)
    band_widths = jnp.diff(opa.band_edges, axis=1).ravel()
    internal_flux = 2.0e6
    tmin, tmax = np.asarray(opa.ckd_info.T_grid)[[0, -1]]
    pmin, pmax = np.asarray(opa.ckd_info.P_grid)[[0, -1]]

    def optical_depth(values):
        return layer_optical_depth_ckd(
            np.diff(boundaries), opa.xstensor_ckd(values, pressure),
            1.0e-3, 18.0, 1000.0,
        )

    def flux_from_depth(values, bottom, dtau):
        boundary_temperature = reconstruct_boundary_temperature(
            pressure, boundaries, values, bottom
        )
        source = piBarr(boundary_temperature, opa.nu_bands)[:, None, :]
        upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, source, mus, angular_weights,
            source_center=piBarr(values, opa.nu_bands)[:, None, :],
            upper_fraction=(pressure - boundaries[:-1]) / np.diff(boundaries),
        )
        return integrate_ckd_flux(
            upward - downward, opa.ckd_info.weights, band_widths
        )

    def radiative_flux(values, bottom):
        return flux_from_depth(values, bottom, optical_depth(values))

    def valid_state(values, bottom):
        # Only centers are used for opacity lookup; the bottom emits Planck flux.
        return bool(
            np.all((values >= tmin) & (values <= tmax))
            and np.all((pressure >= pmin) & (pressure <= pmax))
        )

    direction = jnp.ones_like(temperature)
    _, derivative = jax.jvp(
        lambda values: radiative_flux(values, bottom_temperature),
        (temperature,), (direction,),
    )
    step = 1.0e-2
    finite_difference = (
        radiative_flux(temperature + step, bottom_temperature)
        - radiative_flux(temperature - step, bottom_temperature)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-7)
    frozen_depth = optical_depth(temperature)
    _, frozen_derivative = jax.jvp(
        lambda values: flux_from_depth(values, bottom_temperature, frozen_depth),
        (temperature,), (direction,),
    )
    assert np.linalg.norm(derivative - frozen_derivative) > 0.01 * np.linalg.norm(
        derivative
    )

    result = solve_rce(
        pressure, boundaries, temperature, bottom_temperature, internal_flux,
        radiative_flux, valid_state=valid_state, flux_atol=0.1, flux_rtol=1.0e-7,
    )
    assert result.converged, result.status
    assert result.domain_valid
    assert np.any(result.convective_mask)
    assert np.all(result.convective_flux >= -0.3)
    np.testing.assert_allclose(result.flux_residual, 0.0, atol=0.3)
    assert np.max(result.gradient_residual) <= 1.0e-6

    assert valid_state(temperature, 2500.0)
    for invalid_temperature in (200.0, 2200.0):
        with pytest.raises(ValueError, match="valid_state"):
            solve_rce(
                pressure, boundaries, np.full(len(pressure), invalid_temperature),
                bottom_temperature, internal_flux, radiative_flux,
                valid_state=valid_state,
            )
    pressure *= 1.0e-3
    boundaries *= 1.0e-3
    with pytest.raises(ValueError, match="valid_state"):
        solve_rce(
            pressure, boundaries, temperature, bottom_temperature, internal_flux,
            radiative_flux, valid_state=valid_state,
        )


def test_ckd_interface_flux_and_heating_match_synthetic_lbl():
    # Narrow bands isolate k quadrature from the band-center Planck approximation.
    opa = _synthetic_ckd(np.array([[999.95, 1000.05], [1999.95, 2000.05]]))
    boundaries = np.array([0.01, 0.1, 1.0, 10.0])
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    temperature = jnp.array([500.0, 650.0, 900.0])
    boundary_temperature = reconstruct_boundary_temperature(
        pressure, boundaries, temperature, 1100.0
    )
    mus, angular_weights = gauss_legendre_grid(4)
    widths = jnp.diff(opa.band_edges, axis=1).ravel()
    dtau = layer_optical_depth_ckd(
        np.diff(boundaries), opa.xstensor_ckd(temperature, pressure),
        1.0e-3, 18.0, 1000.0,
    )
    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, piBarr(boundary_temperature, opa.nu_bands)[:, None, :],
        mus, angular_weights,
        source_center=piBarr(temperature, opa.nu_bands)[:, None, :],
        upper_fraction=(pressure - boundaries[:-1]) / np.diff(boundaries),
    )
    flux_ckd = integrate_ckd_flux(upward - downward, opa.ckd_info.weights, widths)

    # Resolve the original spectrum, whose opacity order is shared by all layers.
    nfrequency = 1024
    fraction = (jnp.arange(nfrequency) + 0.5) / nfrequency
    nu_grid = (opa.band_edges[:, :1] + widths[:, None] * fraction).ravel()
    spectral_fraction = jnp.tile(fraction, len(widths))
    cross_section = (
        2.0e-24 * pressure[:, None]
        * jnp.exp((temperature[:, None] - 700.0) / 1400.0 + 2.0 * spectral_fraction)
    )
    dtau_lbl = layer_optical_depth(
        np.diff(boundaries), cross_section, 1.0e-3, 18.0, 1000.0
    )
    upward_lbl, downward_lbl = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau_lbl, piBarr(boundary_temperature, nu_grid), mus, angular_weights,
        source_center=piBarr(temperature, nu_grid),
        upper_fraction=(pressure - boundaries[:-1]) / np.diff(boundaries),
    )
    flux_lbl = jnp.sum(
        (upward_lbl - downward_lbl) * jnp.repeat(widths / nfrequency, nfrequency),
        axis=1,
    )
    np.testing.assert_allclose(flux_ckd, flux_lbl, rtol=3.0e-5)
    # Layer heating is proportional to the boundary flux difference.
    np.testing.assert_allclose(np.diff(flux_ckd), np.diff(flux_lbl), rtol=3.0e-5)
