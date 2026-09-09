"""Offline physical checks for the tutorial's continuum adapter."""

import importlib.util
import io
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import netcdf_file


_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/rce_earth_continuum.py"
_SPEC = importlib.util.spec_from_file_location("rce_earth_continuum", _EXAMPLE)
continuum = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(continuum)


@pytest.fixture
def continuum_table(tmp_path):
    """Synthetic coefficients, independent of the externally licensed data."""
    path = tmp_path / "continuum.nc"
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("nu", 4)
        arrays = {
            "wavenumbers": [0.0, 500.0, 1000.0, 20000.0],
            "self_absco_ref": [2e-25, 4e-25, 6e-25, 8e-25],
            "for_absco_ref": [1e-27, 2e-27, 3e-27, 4e-27],
            "self_texp": [2.0, 3.0, 4.0, 5.0],
        }
        for name, values in arrays.items():
            dataset.createVariable(name, "d", ("nu",))[:] = values
        for name, value in [("ref_temp", 296.0), ("ref_press", 1013.0)]:
            dataset.createVariable(name, "d", ())[...] = value
    return path


def test_reference_state_radiation_factor_and_spectral_support(continuum_table):
    sigma = jax.jit(continuum.load_continuum(continuum_table))
    nu = jnp.array([-1.0, 0.0, 500.0, 750.0, 1000.0, 20001.0])
    actual = sigma(jnp.array([296.0]), jnp.array([1.013]), jnp.array([0.02]), nu)
    native_nu = np.array([500.0, 1000.0])
    stimulated_emission = (1.0 - np.exp(-1.4387752 * native_nu / 296.0)) / (
        1.0 + np.exp(-1.4387752 * native_nu / 296.0)
    )
    expected_native = (
        np.array([4e-25, 6e-25]) * 0.02 + np.array([2e-27, 3e-27]) * 0.98
    ) * native_nu * stimulated_emission
    expected = [
        0.0, 0.0, expected_native[0], expected_native.mean(), expected_native[1], 0.0
    ]
    np.testing.assert_allclose(actual[0], expected, rtol=1e-13, atol=0.0)


def test_partner_density_and_self_temperature_dependence(continuum_table):
    sigma = continuum.load_continuum(continuum_table)
    temperature = jnp.array([200.0, 296.0, 320.0])
    pressure = jnp.ones(3)
    frequency = jnp.array([500.0])
    for water, coefficient, exponent in [(0.0, 2e-27, 0.0), (1.0, 4e-25, 3.0)]:
        actual = sigma(temperature, pressure, jnp.full(3, water), frequency)[:, 0]
        expected = coefficient * (296.0 / temperature) ** (1.0 + exponent) / 1.013
        expected *= 500.0 * jnp.tanh(1.4387752 * 500.0 / (2.0 * temperature))
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
    water = jnp.full(3, 0.02)
    original = sigma(temperature, pressure, water, frequency)
    np.testing.assert_allclose(
        sigma(temperature, 2.0 * pressure, water, frequency), 2.0 * original
    )
    # Multiplying by the water column adds the second power of density.
    np.testing.assert_allclose(
        2.0 * sigma(temperature, 2.0 * pressure, water, frequency), 4.0 * original
    )


def test_temperature_jacobian_is_finite_and_matches_finite_difference(continuum_table):
    sigma = continuum.load_continuum(continuum_table)

    def evaluate(temperature):
        return 1e23 * sigma(
            temperature, jnp.array([0.1, 1.0]),
            jnp.array([0.01, 0.02]), jnp.array([0.0, 750.0])
        )

    temperatures = jnp.array([230.0, 300.0])
    derivative = np.asarray(jax.jacfwd(evaluate)(temperatures))
    step = 1e-3
    finite_difference = np.stack(
        [
            (
                evaluate(temperatures + step * direction)
                - evaluate(temperatures - step * direction)
            ) / (2.0 * step)
            for direction in np.eye(2)
        ], axis=-1
    )
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-12)


def test_download_rejects_unverified_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        continuum, "urlopen", lambda *args, **kwargs: io.BytesIO(b"incorrect data")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        continuum.download_data(tmp_path)
    assert not list(tmp_path.iterdir())
