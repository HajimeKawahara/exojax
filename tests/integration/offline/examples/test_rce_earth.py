"""Check spectral flux/composition coupling without downloading line data."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.ckd.core import gauss_legendre_grid


@pytest.fixture
def column(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    example = importlib.import_module("rce_earth")
    g, weights = gauss_legendre_grid(4)
    edges = np.linspace(20.0, 30000.0, 81)
    bands = jnp.asarray(np.column_stack((edges[:-1], edges[1:])))
    nu = jnp.mean(bands, axis=1)

    def cross_section(temperature, pressure):
        return (
            1.0e-23 * jnp.exp((temperature[:, None, None] - 260.0) / 600.0)
            * (pressure[:, None, None] + 0.01) * jnp.exp(3.0 * g[None, :, None])
            * (0.05 + jnp.exp(-((nu[None, None, :] - 1500.0) / 900.0)**2))
        )

    opacity = SimpleNamespace(
        band_edges=bands, xstensor_ckd=cross_section,
        ckd_info=SimpleNamespace(weights=weights, T_grid=jnp.array([160.0, 340.0]),
                                 P_grid=jnp.array([1.0e-4, 1.0])),
    )
    no_continuum = lambda t, p, x, nu: jnp.zeros((t.size, nu.size))
    return example.EarthColumn(opacity, no_continuum, nlayer=6, albedo=0.30)


def test_global_solar_flux_and_black_surface_boundary(column):
    temperature = jnp.linspace(210.0, 280.0, 6)
    up, down, stellar = column.spectral_fluxes(temperature, 288.0)
    integration = lambda f: jnp.sum(f * column.g_weights[:, None] * column.widths)
    assert float(integration(stellar[0])) / 1000.0 == pytest.approx(238.175)
    assert np.all(np.asarray(stellar[1:] <= stellar[:-1]))
    assert np.all(np.asarray(down[0]) == 0.0)
    from exojax.rt.planck import piB
    np.testing.assert_allclose(
        up[-1], jnp.broadcast_to(piB(288.0, column.nu), up[-1].shape),
        rtol=1.0e-12, atol=1.0e-12,
    )
    assert float(integration(stellar[-1])) < float(integration(stellar[0]))


def test_ocean_water_feedback_enters_radiative_jacobian(column, monkeypatch):
    # This warm column receives its water from the cooler ocean, so its
    # abundance responds to the ocean temperature throughout the atmosphere.
    temperature = jnp.full(6, 300.0)
    bottom = 288.0
    derivative = jax.jacfwd(column.radiative_flux, argnums=1)(temperature, bottom)
    step = 1.0e-3
    finite_difference = (
        column.radiative_flux(temperature, bottom + step)
        - column.radiative_flux(temperature, bottom - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-7, atol=1.0e-6)
    frozen_depth = column.optical_depth(temperature, bottom)
    monkeypatch.setattr(column, "optical_depth", lambda t, tb: frozen_depth)
    frozen_derivative = jax.jacfwd(column.radiative_flux, argnums=1)(temperature, bottom)
    assert np.linalg.norm(derivative - frozen_derivative) > 0.1 * np.linalg.norm(derivative)


def test_valid_state_rejects_table_extrapolation_and_nonliquid_ocean(column):
    temperature = np.full(6, 250.0)
    assert column.valid_state(temperature, 288.0)
    assert not column.valid_state(temperature, 270.0)
    assert not column.valid_state(np.full(6, 150.0), 288.0)
    assert not column.valid_state(np.full(6, 350.0), 288.0)
    assert not column.valid_state(temperature, 330.0)


@pytest.mark.parametrize("incomplete", ["thermal_only", "gap"])
def test_rejects_missing_spectral_coverage(column, incomplete):
    example = importlib.import_module("rce_earth")
    edges = np.asarray(column.opacity.band_edges).copy()
    if incomplete == "thermal_only":
        edges = edges[:10]
    else:
        edges[10, 0] += 1.0
    column.opacity.band_edges = jnp.asarray(edges)
    with pytest.raises(ValueError, match="continuously cover"):
        example.EarthColumn(column.opacity, column.continuum)
