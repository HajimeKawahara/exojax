"""ExoAtom's direct-opacity contract, independent of database downloads."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import voigt_profile

from exojax.database.core.broadening import doppler_sigma
from exojax.opacity import OpaAlkali, OpaDirect
from exojax.utils.constants import ccgs, hcperk


GRID = np.linspace(999.5, 1002.5, 301)


def _database(elower=(200.0, 500.0), reference_temperature=296.0, element=11, center=1000.0):
    energy = np.asarray(elower)
    count = len(energy)
    centers = center + 2.0 * np.arange(count)
    einstein_a = np.full(count, 1e7)
    upper_weights = 2.0 + 2.0 * np.arange(count)
    temperatures = np.array([100.0, 296.0, 1000.0, 3000.0, 10000.0])
    partitions = np.array([2.0, 2.1, 3.5, 8.0, 30.0])
    reference_partition = np.interp(reference_temperature, temperatures, partitions)
    log_strength = (
        np.log(einstein_a) + np.log(upper_weights)
        - np.log(8.0 * np.pi * ccgs) - 2.0 * np.log(centers)
        - np.log(reference_partition) - hcperk * energy / reference_temperature
        + np.log(-np.expm1(-hcperk * centers / reference_temperature))
    )

    def partition_ratio(T, Tref):
        ratio = (
            jnp.interp(T, temperatures, partitions)
            / jnp.interp(Tref, temperatures, partitions)
        )
        return jnp.full((count,), ratio)

    return SimpleNamespace(
        dbtype="exoatom", Tref=reference_temperature,
        nu_lines=centers, A=jnp.asarray(einstein_a),
        logsij0=jnp.asarray(log_strength), elower=jnp.asarray(energy),
        line_masses=jnp.full(count, 39.0983 if element == 19 else 22.989769),
        _ielem=np.full(count, element), _iion=np.ones(count, dtype=int),
        qr_interp_lines=partition_ratio,
        gamma_natural=jnp.asarray(0.02 + 0.02 * np.arange(count)),
        gupper=upper_weights, T_gQT=temperatures, gQT=partitions,
    )


def _scipy_spectrum(adb, temperature, width, grid=GRID, line_profile="voigt"):
    """Evaluate LTE strengths and the selected profile independently with SciPy."""
    partition = np.interp(temperature, adb.T_gQT, adb.gQT)
    strength = (
        np.asarray(adb.A) * adb.gupper
        * np.exp(-hcperk * np.asarray(adb.elower) / temperature)
        * -np.expm1(-hcperk * adb.nu_lines / temperature)
        / (8.0 * np.pi * ccgs * adb.nu_lines**2 * partition)
    )
    sigma = np.asarray(doppler_sigma(adb.nu_lines, temperature, adb.line_masses))
    offsets = grid[None, :] - adb.nu_lines[:, None]
    widths = np.asarray(width)[:, None]
    profiles = voigt_profile(offsets, sigma[:, None], widths)
    if line_profile == "alkali_subvoigt":
        detuning_ref, cutoff = (30.0, 5000.0) if adb._ielem[0] == 11 else (20.0, 1600.0)
        detuning = detuning_ref * (temperature / 500.0)**0.6
        distance = np.maximum(np.abs(offsets), detuning)
        wings = (
            voigt_profile(detuning, sigma[:, None], widths)
            * (detuning / distance)**1.5
            * np.exp(-hcperk * distance**2 / (temperature * cutoff))
        )
        profiles = np.where(np.abs(offsets) < detuning, profiles, wings) / 0.998
        profiles[np.abs(offsets) > 9000.0] = 0.0
    return np.sum(strength[:, None] * profiles, axis=0)


@pytest.mark.parametrize("reference_temperature", [296.0, 1000.0])
def test_natural_widths_match_scipy_and_are_pressure_independent(reference_temperature):
    adb = _database(reference_temperature=reference_temperature)
    opacity = OpaDirect(adb, GRID)
    temperatures = jnp.array([750.0, 1800.0])
    pressures = jnp.array([0.1, 4.0])
    matrix = jax.jit(opacity.xsmatrix)(temperatures, pressures)
    vectors = jnp.stack([
        jax.jit(opacity.xsvector)(T, P) for T, P in zip(temperatures, pressures)
    ])
    np.testing.assert_allclose(matrix, vectors, rtol=1e-12, atol=0)
    for index, temperature in enumerate(temperatures):
        np.testing.assert_allclose(
            matrix[index], _scipy_spectrum(adb, float(temperature), adb.gamma_natural),
            rtol=2e-6, atol=0,
        )
    np.testing.assert_array_equal(
        opacity.xsvector(1800.0, 0.0), opacity.xsvector(1800.0, 10.0),
    )
    assert jax.grad(lambda P: jnp.sum(opacity.xsvector(1800.0, P)))(1.0) == 0.0


@pytest.mark.parametrize("natural_width", [
    "missing", None, [np.nan, 0.04], [np.inf, 0.04], [-0.02, 0.04],
    [0.02], [[0.02, 0.04]],
])
@pytest.mark.parametrize("line_profile", ["voigt", "alkali_subvoigt"])
def test_incomplete_natural_widths_require_explicit_broadening(natural_width, line_profile):
    adb = _database()
    if isinstance(natural_width, str):
        del adb.gamma_natural
    else:
        adb.gamma_natural = natural_width
    with pytest.raises(ValueError, match="ExoAtom requires.*atomic_broadening"):
        OpaDirect(adb, GRID, line_profile=line_profile)


@pytest.mark.parametrize("line_profile", ["voigt", "alkali_subvoigt"])
def test_total_width_override_matches_scipy_and_has_temperature_pressure_gradients(line_profile):
    adb = _database()
    adb.gamma_natural = jnp.array([np.nan, np.inf])

    def broadening(T, P):
        return jnp.array([0.03, 0.05]) * P * (296.0 / T)**0.7 + 0.004

    opacity = OpaDirect(adb, GRID, line_profile=line_profile, atomic_broadening=broadening)
    temperatures, pressures = jnp.array([750.0, 1800.0]), jnp.array([0.1, 4.0])
    matrix = jax.jit(opacity.xsmatrix)(temperatures, pressures)
    vectors = jax.jit(jax.vmap(opacity.xsvector))(temperatures, pressures)
    np.testing.assert_allclose(matrix, vectors, rtol=1e-12, atol=0)
    for index, (temperature, pressure) in enumerate(zip(temperatures, pressures)):
        np.testing.assert_allclose(
            matrix[index],
            _scipy_spectrum(
                adb, float(temperature), broadening(temperature, pressure),
                line_profile=line_profile,
            ),
            rtol=2e-6, atol=0,
        )

    center = np.argmin(np.abs(GRID - adb.nu_lines[0]))
    log_peak = jax.jit(lambda T, P: jnp.log(opacity.xsvector(T, P)[center]))
    gradient = jax.jit(jax.grad(log_peak, argnums=(0, 1)))(1800.0, 4.0)
    finite_difference = (
        (log_peak(1800.1, 4.0) - log_peak(1799.9, 4.0)) / 0.2,
        (log_peak(1800.0, 4.0001) - log_peak(1800.0, 3.9999)) / 0.0002,
    )
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) > 1e-8)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-10)


def test_override_is_total_width_without_adding_natural_broadening():
    adb = _database()
    opacity = OpaDirect(
        adb, GRID, atomic_broadening=lambda T, P: jnp.full_like(adb.A, 0.007),
    )
    np.testing.assert_allclose(
        opacity.xsvector(1800.0, 1.0), _scipy_spectrum(adb, 1800.0, np.full(2, 0.007)),
        rtol=2e-6, atol=0,
    )


@pytest.mark.parametrize("width", [0.1, [0.1], [[0.1, 0.2]]])
def test_override_rejects_non_line_shapes(width):
    opacity = OpaDirect(_database(), GRID, atomic_broadening=lambda T, P: width)
    with pytest.raises(ValueError, match="atomic_broadening must return shape"):
        jax.jit(opacity.xsvector)(1800.0, 1.0)


def test_high_energy_lines_remain_usable_when_reference_strength_underflows():
    adb = _database(elower=(160000.0,))
    assert np.exp(np.asarray(adb.logsij0))[0] == 0.0
    opacity = OpaDirect(adb, GRID)
    actual = jax.jit(opacity.xsvector)(8000.0, 1.0)
    expected = _scipy_spectrum(adb, 8000.0, adb.gamma_natural)
    assert np.all(np.isfinite(actual)) and np.max(actual) > 0
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("element", [11, 19])
@pytest.mark.parametrize("override_width", [False, True])
def test_alkali_spectrum_wrappers_and_gradients(element, override_width):
    adb = _database(element=element, center=17000.0 if element == 11 else 13000.0)
    grid = adb.nu_lines[0] + np.array([-300.0, -0.03, 0.0, 2.03, 300.0, 9002.1])
    broadening = None
    if override_width:
        broadening = lambda T, P: jnp.array([0.03, 0.05]) * P * (296.0 / T)**0.7 + 0.004
    opacity = OpaDirect(
        adb, grid, line_profile="alkali_subvoigt", atomic_broadening=broadening,
    )
    wrapper = OpaAlkali(adb, grid, atomic_broadening=broadening)
    temperatures, pressures = jnp.array([750.0, 1800.0]), jnp.array([0.1, 4.0])
    matrix = jax.jit(opacity.xsmatrix)(temperatures, pressures)
    vectors = jax.jit(jax.vmap(opacity.xsvector))(temperatures, pressures)
    np.testing.assert_allclose(matrix, vectors, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        jax.jit(wrapper.xsmatrix)(temperatures, pressures), matrix, rtol=1e-12, atol=0,
    )
    for index, (temperature, pressure) in enumerate(zip(temperatures, pressures)):
        width = broadening(temperature, pressure) if override_width else adb.gamma_natural
        expected = _scipy_spectrum(adb, float(temperature), width, grid, "alkali_subvoigt")
        np.testing.assert_allclose(matrix[index], expected, rtol=2e-6, atol=0)
    np.testing.assert_array_equal(matrix[:, -1], 0.0)

    # Sample both cores and wings, away from the moving joins and zero tails.
    signal = jax.jit(lambda T, P: jnp.mean(jnp.log(opacity.xsvector(T, P)[:-1])))
    gradient = jax.jit(jax.grad(signal, argnums=(0, 1)))(1800.0, 4.0)
    finite_difference = (
        (signal(1800.1, 4.0) - signal(1799.9, 4.0)) / 0.2,
        (signal(1800.0, 4.0001) - signal(1800.0, 3.9999)) / 0.0002,
    )
    assert np.all(np.isfinite(gradient)) and abs(gradient[0]) > 1e-8
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-10)
    if override_width:
        assert abs(gradient[1]) > 1e-8
    else:
        assert gradient[1] == 0.0
        np.testing.assert_array_equal(
            opacity.xsvector(1800.0, 0.0), opacity.xsvector(1800.0, 10.0),
        )


@pytest.mark.parametrize("elements,ions", [
    ([3, 3], [1, 1]), ([11, 11], [2, 2]), ([19, 19], [2, 2]),
    ([11, 19], [1, 1]), ([11, 11], [1, 2]), ([], []),
])
def test_alkali_rejects_other_species_ions_and_empty_selections(elements, ions):
    adb = _database(elower=np.full(len(elements), 200.0))
    adb._ielem, adb._iion = np.asarray(elements), np.asarray(ions)
    with pytest.raises(ValueError, match="single neutral species, Na I or K I"):
        OpaAlkali(adb, GRID)
