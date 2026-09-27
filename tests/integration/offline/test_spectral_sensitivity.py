"""Offline physical checks for the CO information-content tutorial."""

import importlib.util
from pathlib import Path
import socket

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_EXAMPLE = Path(__file__).resolve().parents[3] / "examples/spectral_sensitivity.py"
_SPEC = importlib.util.spec_from_file_location("spectral_sensitivity", _EXAMPLE)
tutorial = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tutorial)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("The bundled CO tutorial must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


def test_bundled_cia_retains_temperature_dependence():
    path = _EXAMPLE.with_name("spectral_sensitivity_data") / "H2-H2_2011_4320-4370.cia"
    nu = np.array([4330.5, 4345.5, 4361.5])
    database = tutorial.CdbCIA(str(path), nu)
    # Cover the tutorial's nominal atmospheric temperatures without clamping.
    assert float(database.tcia.min()) < 470.0
    assert float(database.tcia.max()) > 1910.0
    opacity = tutorial.OpaCIA(database, nu)
    temperatures = jnp.array([487.0, 1207.0, 1893.0])
    derivative = jax.vmap(jax.jacfwd(opacity.logacia_vector))(temperatures)
    coefficients = opacity.logacia_matrix(temperatures)
    assert np.all(np.isfinite(coefficients))
    assert np.all(np.isfinite(derivative))
    assert np.all(np.linalg.norm(derivative, axis=1) > 0.0)


def test_observed_co_spectrum_jacobian_matches_finite_difference():
    # Use fewer quadrature samples for a fast integration check; this does not
    # test the spectral convergence of the tutorial's default 4096-cell grid.
    observed_spectrum, nu = tutorial.make_observed_spectrum(samples_per_bin=2)
    forward = jax.jit(observed_spectrum)
    theta = jnp.array([1200.0, -2.3, 0.1])
    flux = np.asarray(forward(theta))
    jacobian = np.asarray(jax.jit(jax.jacfwd(observed_spectrum))(theta))
    assert flux.shape == nu.shape == (256,)
    assert jacobian.shape == (256, 3)
    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0.0)
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.linalg.norm(jacobian, axis=0) > 0.0)

    steps = np.array([1.0e-2, 1.0e-5, 1.0e-6])
    differences = np.stack(
        [
            (forward(theta + direction * step) - forward(theta - direction * step))
            / (2.0 * step)
            for direction, step in zip(np.eye(3), steps)
        ], axis=-1,
    )
    # A derivative can cross zero within the band, so use each parameter's
    # spectral sensitivity scale when comparing the numerical derivatives.
    scale = np.max(np.abs(jacobian), axis=0)
    np.testing.assert_allclose(
        jacobian / scale, differences / scale, rtol=2.0e-6, atol=2.0e-6
    )
