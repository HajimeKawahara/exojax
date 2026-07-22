"""Tests for layer-dependent LPF detuning and profile evaluation."""

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.broadening import gamma_natural
from exojax.opacity.lpf.lpf import voigt_profile_tensor
from exojax.opacity.lpf.make_numatrix import doppler_shifted_line_detuning
from exojax.opacity.lpf.make_numatrix import make_numatrix0


NU0 = 15232.997694444446
A_H_ALPHA = 4.4074272e7
MASS_H = 1.008


def _profile_tensor(nu_grid, temperatures, velocities):
    nu_lines_np = np.array([NU0], dtype=np.float64)
    numatrix0 = make_numatrix0(nu_grid, nu_lines_np)
    nu_lines = jnp.asarray(nu_lines_np)
    detuning = doppler_shifted_line_detuning(numatrix0, nu_lines, velocities)
    sigma = jnp.stack(
        [doppler_sigma(nu_lines, temperature, MASS_H) for temperature in temperatures]
    )
    gamma = jnp.broadcast_to(gamma_natural(jnp.array([A_H_ALPHA])), sigma.shape)
    return voigt_profile_tensor(detuning, sigma, gamma)


def test_positive_velocity_redshifts_line_center():
    nu_grid = np.linspace(NU0 - 3.0, NU0 + 3.0, 6001, dtype=np.float64)
    profiles = _profile_tensor(nu_grid, [6000.0, 6000.0], jnp.array([0.0, 30.0]))
    peaks = jnp.argmax(profiles[:, 0, :], axis=1)
    peak_nu = nu_grid[np.asarray(peaks)]

    assert peak_nu[1] < peak_nu[0]
    expected = NU0 / (1.0 + 30.0 / 299792.458)
    assert float(peak_nu[1]) == pytest.approx(expected, abs=0.002)


def test_voigt_profile_tensor_is_normalized_on_wide_grid():
    nu_grid = np.linspace(NU0 - 100.0, NU0 + 100.0, 200001, dtype=np.float64)
    profile = _profile_tensor(nu_grid, [6000.0], jnp.array([0.0]))
    integral = jnp.trapezoid(profile[0, 0], jnp.asarray(nu_grid))

    assert float(integral) == pytest.approx(1.0, rel=2.0e-6)


def test_rest_numatrix_preserves_sub_float32_wavenumber_detuning():
    nu_lines = np.array([NU0], dtype=np.float64)
    nu_grid = np.array([NU0 - 1.0e-4, NU0 + 1.0e-4], dtype=np.float64)
    assert np.float32(nu_grid[0]) == np.float32(nu_lines[0])

    numatrix0 = make_numatrix0(nu_grid, nu_lines)
    detuning = doppler_shifted_line_detuning(
        numatrix0, jnp.asarray(nu_lines), jnp.array([0.0])
    )

    np.testing.assert_allclose(
        np.asarray(detuning[0, 0]),
        np.array([-1.0e-4, 1.0e-4]),
        rtol=0.0,
        atol=5.0e-8,
    )


def test_velocity_line_axis_must_match_numatrix():
    with pytest.raises(ValueError, match="line axis"):
        doppler_shifted_line_detuning(
            jnp.ones((2, 3)),
            jnp.array([NU0, NU0 + 1.0]),
            jnp.ones((3, 1)),
        )
