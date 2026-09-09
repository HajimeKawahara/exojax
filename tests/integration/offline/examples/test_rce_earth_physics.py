"""Thermodynamic checks for the N2--H2O RCE example without opacity downloads."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def physics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("rce_earth_physics")


def test_saturation_obeys_clausius_clapeyron(physics):
    assert float(physics.saturation_vapor_pressure(273.16)) == pytest.approx(611.657)
    temperature = jnp.array([180.0, 250.0, 288.0, 320.0])
    log_derivative = jax.vmap(
        jax.grad(lambda value: jnp.log(physics.saturation_vapor_pressure(value)))
    )(temperature)
    np.testing.assert_allclose(
        log_derivative,
        physics.latent_heat(temperature) / (physics.R_VAPOR * temperature**2),
        rtol=1.0e-12,
    )


def test_pseudoadiabat_dry_and_steam_limits(physics):
    temperature = jnp.array([200.0, 250.0, 288.0, 320.0])
    np.testing.assert_allclose(
        physics.pseudoadiabatic_gradient(temperature, 0.0), 2.0 / 7.0,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        physics.pseudoadiabatic_gradient(temperature, 1.0e10),
        physics.R_VAPOR * temperature / physics.latent_heat(temperature),
        rtol=1.0e-9,
    )
    vapor_fraction = physics.saturation_vapor_pressure(288.0) / 1.0e5
    ratio = physics.EPSILON * vapor_fraction / (1.0 - vapor_fraction)
    assert 0.13 < float(physics.pseudoadiabatic_gradient(288.0, ratio)) < 0.15


@pytest.fixture
def cold_trap_column():
    # The warm low-pressure top has e_sat > P; water remains cold-trap limited.
    pressure = jnp.array([1.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.5])
    temperature = jnp.array([240.0, 210.0, 200.0, 240.0, 275.0])
    return pressure, temperature, 288.0, 1.0


def test_ocean_saturation_and_cold_trap(physics, cold_trap_column):
    pressure, temperature, bottom_temperature, bottom_pressure = cold_trap_column
    vmr = np.asarray(physics.water_vmr(*cold_trap_column))
    saturation = np.asarray(
        physics.saturation_vapor_pressure(jnp.append(temperature, bottom_temperature))
        / (1.0e5 * jnp.append(pressure, bottom_pressure))
    )
    assert saturation[0] > 1.0
    assert np.all((vmr > 0.0) & (vmr < 1.0))
    assert np.all(np.diff(vmr) >= 0.0)
    assert np.all(vmr <= saturation)
    np.testing.assert_allclose(vmr[:3], saturation[2], rtol=1.0e-12)
    np.testing.assert_allclose(vmr[3:], saturation[3:], rtol=1.0e-12)


def test_cold_trap_uses_dry_and_lower_column_uses_moist_gradient(
    physics, cold_trap_column
):
    gradient = np.asarray(physics.convective_gradient(*cold_trap_column))
    assert gradient.shape == (5,)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient[:2], 2.0 / 7.0, rtol=1.0e-3)
    assert np.all(gradient[2:] < gradient[1])
    assert gradient[-1] < 0.17


def test_moist_closure_is_jittable_and_has_finite_derivatives(
    physics, cold_trap_column
):
    pressure, temperature, bottom_temperature, bottom_pressure = cold_trap_column

    def closure(node_temperature):
        values = physics.water_vmr(
            pressure, node_temperature[:-1], node_temperature[-1], bottom_pressure
        )
        gradient = physics.convective_gradient(
            pressure, node_temperature[:-1], node_temperature[-1], bottom_pressure
        )
        return jnp.concatenate((values, gradient))

    nodes = jnp.append(temperature, bottom_temperature)
    assert np.all(np.isfinite(jax.jit(closure)(nodes)))
    derivative = np.asarray(jax.jit(jax.jacfwd(closure))(nodes))
    assert np.all(np.isfinite(derivative))
    # Water above the cold trap depends on its temperature, not local warming.
    assert derivative[0, 0] == 0.0
    assert derivative[0, 2] > 0.0
    # The ocean's vapor supply responds to the solved surface temperature.
    assert derivative[len(nodes) - 1, -1] > 0.0
