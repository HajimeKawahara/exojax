"""Hydrogen line selection and deferred device transfer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database import AdbHydrogen
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.line_coefficients import line_coefficients


def _assert_selected(adb, upper, transferred):
    np.testing.assert_array_equal(adb.n_upper, upper)
    np.testing.assert_array_equal(adb.n_lower, np.full(len(upper), 2))
    np.testing.assert_array_equal(adb._gupper, 2 * np.asarray(upper)**2)
    assert adb.nu_lines.shape == adb._A.shape == adb.Sij0.shape == (len(upper),)
    array_type = jax.Array if transferred else np.ndarray
    for name in ("atomicmass", "solarA", "ionE"):
        values = getattr(adb, name)
        assert isinstance(values, array_type)
        assert values.shape == (len(upper),)
    assert adb.line_masses is adb.atomicmass
    if transferred:
        np.testing.assert_array_equal(adb.dev_nu_lines, adb.nu_lines)
        np.testing.assert_array_equal(adb.A, adb._A)
        np.testing.assert_array_equal(adb.gupper, adb._gupper)
        np.testing.assert_allclose(adb.logsij0, np.log(adb.Sij0))
    else:
        assert not hasattr(adb, "dev_nu_lines")


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_repeated_selection_keeps_host_and_device_arrays_aligned(gpu_transfer):
    adb = AdbHydrogen(gpu_transfer=gpu_transfer)
    _assert_selected(adb, np.arange(3, 11), gpu_transfer)
    adb.masking(np.isin(adb.n_upper, [3, 5, 7]))
    _assert_selected(adb, [3, 5, 7], gpu_transfer)
    if not gpu_transfer:
        adb.generate_jnp_arrays()
        _assert_selected(adb, [3, 5, 7], True)
    adb.masking(adb.n_upper == 5)
    _assert_selected(adb, [5], True)


@pytest.mark.parametrize("gpu_transfer", [False, True])
@pytest.mark.parametrize("selection", ["constructor", "mask"])
def test_empty_selection_keeps_arrays_aligned(gpu_transfer, selection):
    if selection == "constructor":
        with pytest.warns(UserWarning, match="No hydrogen lines"):
            adb = AdbHydrogen(nurange=[1.0, 2.0], gpu_transfer=gpu_transfer)
    else:
        adb = AdbHydrogen(gpu_transfer=gpu_transfer)
        with pytest.warns(UserWarning, match="No hydrogen lines"):
            adb.masking(np.zeros(len(adb.nu_lines), dtype=bool))
    _assert_selected(adb, [], gpu_transfer)
    if not gpu_transfer:
        adb.generate_jnp_arrays()
        _assert_selected(adb, [], True)


@pytest.mark.parametrize("gpu_transfer", [False, True])
@pytest.mark.parametrize("upper", [3, 4])
def test_masked_opacity_matches_constructor_selection(gpu_transfer, upper):
    masked = AdbHydrogen(gpu_transfer=gpu_transfer)
    masked.masking(masked.n_upper <= 6)
    masked.masking(masked.n_upper == upper)
    if not gpu_transfer:
        masked.generate_jnp_arrays()
    direct = AdbHydrogen(n_upper_min=upper, n_upper_max=upper)
    grid = direct.nu_lines[0] + np.linspace(-5.0, 5.0, 101)
    temperatures = jnp.array([7000.0, 11000.0])
    pressures = jnp.array([0.0, 1.0])
    velocities = jnp.array([-12.0, 7.0])
    lower = jnp.array([[1e6], [2e6]])
    upper_population = lower * 0.1

    results = []
    for adb in (masked, direct):
        opa = OpaDirect(adb, grid)
        profiles = jax.vmap(opa.line_profiles)(temperatures, velocities)
        extinction, emissivity = line_coefficients(
            profiles, adb.nu_lines, adb.A, adb.glower, adb.gupper,
            lower, upper_population,
        )
        results.append((
            profiles,
            opa.xsvector(temperatures[0], pressures[0]),
            jax.jit(opa.xsmatrix)(temperatures, pressures),
            extinction,
            emissivity,
        ))
    for actual, expected in zip(*results):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
