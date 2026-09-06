"""Offline checks for Na/K opacity with real atomic database adapters."""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import voigt_profile

from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.line_strength import line_strength_numpy
from exojax.opacity import OpaAlkali
from exojax.opacity.alkali import subvoigt
from exojax.opacity.lpf.lpf import vald
from exojax.utils.constants import hcperk


def reference_profile(offsets, sigma, gamma, temperature, detuning_ref, cutoff):
    """Cthulhu's piecewise prescription evaluated with SciPy's exact Voigt."""
    distance = np.abs(offsets)
    detuning = detuning_ref * (temperature / 500.0) ** 0.6
    profile = voigt_profile(offsets, sigma, gamma)
    wing = distance >= detuning
    profile[wing] = (
        voigt_profile(detuning, sigma, gamma)
        * (detuning / distance[wing]) ** 1.5
        * np.exp(-hcperk * distance[wing] ** 2 / (temperature * cutoff))
    )
    profile[distance > 9000.0] = 0.0
    return profile / 0.998


@pytest.mark.parametrize("detuning_ref,cutoff", [(30.0, 5000.0), (20.0, 1600.0)])
@pytest.mark.parametrize("temperature", [500.0, 1370.0])
def test_subvoigt_core_join_wings_and_truncation(detuning_ref, cutoff, temperature):
    detuning = detuning_ref * (temperature / 500.0) ** 0.6
    positive = np.array([
        0.0, 0.03, 0.5, detuning * (1.0 - 1.0e-7), detuning,
        detuning * (1.0 + 1.0e-7), 300.0, 1000.0, 8999.9, 9000.0, 9000.1,
    ])
    offsets = np.concatenate([-positive[:0:-1], positive])
    sigma, gamma = 0.03, 0.11
    expected = reference_profile(
        offsets, sigma, gamma, temperature, detuning_ref, cutoff
    )
    actual = subvoigt(offsets, sigma, gamma, temperature, detuning_ref, cutoff)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=0.0)
    np.testing.assert_array_equal(actual, actual[::-1])
    assert actual[0] == actual[-1] == 0.0
    assert actual[1] > 0.0
    # Preserve the source's small join discontinuity, including the exact join.
    join_value = subvoigt(
        jnp.array([detuning]), sigma, gamma, temperature, detuning_ref, cutoff
    )[0]
    expected_join_ratio = np.exp(-hcperk * detuning**2 / (temperature * cutoff))
    np.testing.assert_allclose(
        join_value / (voigt_profile(detuning, sigma, gamma) / 0.998),
        expected_join_ratio, rtol=2.0e-6,
    )


@pytest.fixture(params=["vald", "kurucz"])
def make_adb(request, monkeypatch, tmp_path):
    """Replace file readers while retaining line strengths and partition data."""
    api = importlib.import_module(f"exojax.database.{request.param}.api")
    frequencies = np.array([12985.0, 13042.0, 16956.0, 16973.0, 17010.0, 17020.0])
    lower_energies = np.array([0.0, 100.0, 0.0, 150.0, 10.0, 30.0])
    transitions = (
        [3.8e7, 3.7e7, 6.2e7, 6.1e7, 1.0e7, 2.0e7],
        frequencies,
        lower_energies,
        frequencies + lower_energies,
        [4.0, 2.0, 4.0, 2.0, 3.0, 4.0],
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.5],
        [1.5, 0.5, 1.5, 0.5, 1.0, 1.5],
        [19, 19, 11, 11, 26, 11],
        [1, 1, 1, 1, 1, 2],
        [8.4, 8.5, 8.6, 8.7, 8.8, 8.9],
        [-6.0, -6.1, -6.2, -6.3, -6.4, -6.5],
        [-7.0, -7.1, -7.2, -7.3, -7.4, -7.5],
    )

    def read_transitions(_):
        return tuple(np.array(values) for values in transitions)

    if request.param == "vald":
        monkeypatch.setattr(api, "_load_vald_dataframe", lambda *args: None)
        monkeypatch.setattr(api, "pickup_param", read_transitions)
        adb_class = api.AdbVald
    else:
        monkeypatch.setattr(api, "read_kurucz", read_transitions)
        adb_class = api.AdbKurucz

    def make(element=None, ion=1, **kwargs):
        adb = adb_class(tmp_path / "synthetic.lines", **kwargs)
        if element is not None:
            adb.masking(np.asarray((adb.ielem == element) & (adb.iion == ion)))
        return adb

    return make


@pytest.mark.parametrize("element", [11, 19])
@pytest.mark.parametrize("override_width", [False, True])
def test_alkali_matches_scipy_with_default_and_custom_widths(
    make_adb, element, override_width
):
    adb = make_adb(element, vmr_fraction=[0.1, 0.2, 0.7])
    nu_grid = np.unique(np.concatenate([
        center + np.array([-1000.0, -70.0, -0.03, 0.0, 0.03, 70.0, 1000.0])
        for center in adb.nu_lines
    ]))
    temperature, pressure = 1370.0, 0.3
    broadening = None
    if override_width:
        broadening = lambda T, P: jnp.array([0.05, 0.09]) * P * (500.0 / T)**0.7
    opa = OpaAlkali(adb, nu_grid, atomic_broadening=broadening)
    if override_width:
        gamma = np.asarray(broadening(temperature, pressure))
        sigma = np.asarray(doppler_sigma(adb.nu_lines, temperature, adb.line_masses))
        strengths = line_strength_numpy(
            temperature, adb.Sij0, adb.nu_lines, np.asarray(adb.elower),
            np.asarray(adb.qr_interp_lines(temperature, adb.Tref)), adb.Tref,
        )
    else:
        strengths, gamma, sigma = (
            np.asarray(values[0]) for values in vald(
                adb, jnp.array([temperature]), jnp.array([pressure * adb.vmrH]),
                jnp.array([pressure * adb.vmrHe]), jnp.array([pressure * adb.vmrHH]),
            )
        )
    detuning_ref, cutoff = (30.0, 5000.0) if element == 11 else (20.0, 1600.0)
    expected = sum(
        strength * reference_profile(
            nu_grid - center, doppler, lorentz, temperature, detuning_ref, cutoff
        )
        for center, strength, lorentz, doppler in zip(
            adb.nu_lines, strengths, gamma, sigma
        )
    )
    np.testing.assert_allclose(
        opa.xsvector(temperature, pressure), expected, rtol=2.0e-6, atol=0.0
    )


@pytest.mark.parametrize("element", [11, 19])
def test_alkali_vector_matrix_and_gradients(make_adb, element):
    adb = make_adb(element, vmr_fraction=[0.1, 0.2, 0.7])
    nu_grid = np.array([
        adb.nu_lines[0] - 300.0,
        adb.nu_lines[0] - 0.03,
        adb.nu_lines[0],
        adb.nu_lines[1] + 0.03,
        adb.nu_lines[1] + 1000.0,
    ])
    opa = OpaAlkali(adb, nu_grid)
    temperatures = jnp.array([830.0, 1370.0, 1830.0])
    pressures = jnp.array([0.01, 0.1, 1.0])
    matrix = jax.jit(opa.xsmatrix)(temperatures, pressures)
    vectors = jnp.stack([
        opa.xsvector(temperature, pressure)
        for temperature, pressure in zip(temperatures, pressures)
    ])
    assert matrix.shape == (3, len(nu_grid))
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix > 0.0)
    np.testing.assert_allclose(matrix, vectors, rtol=1.0e-11, atol=0.0)

    def signal(temperature, pressure):
        return jnp.mean(jnp.log(opa.xsvector(temperature, pressure)))

    def matrix_signal(temperature, pressure):
        result = opa.xsmatrix(
            jnp.array([temperature, 830.0, 1830.0]),
            jnp.array([pressure, 0.01, 1.0]),
        )
        return jnp.mean(jnp.log(result[0]))

    temperature, pressure = 1370.0, 0.1
    value, derivatives = jax.jit(jax.value_and_grad(signal, (0, 1)))(
        temperature, pressure
    )
    matrix_value, matrix_derivatives = jax.jit(
        jax.value_and_grad(matrix_signal, (0, 1))
    )(temperature, pressure)
    np.testing.assert_allclose(matrix_value, value, rtol=1.0e-12)
    np.testing.assert_allclose(matrix_derivatives, derivatives, rtol=1.0e-7, atol=1.0e-12)
    expected_derivatives = (
        (signal(temperature + 0.1, pressure) - signal(temperature - 0.1, pressure))
        / 0.2,
        (signal(temperature, pressure + 1.0e-5)
         - signal(temperature, pressure - 1.0e-5)) / 2.0e-5,
    )
    assert np.all(np.isfinite(derivatives))
    assert np.all(np.abs(derivatives) > 1.0e-8)
    np.testing.assert_allclose(derivatives, expected_derivatives, rtol=1.0e-4)


@pytest.mark.parametrize("element,ion", [(None, 1), (26, 1), (11, 2)])
def test_alkali_rejects_mixed_or_unsupported_species(make_adb, element, ion):
    with pytest.raises(ValueError):
        OpaAlkali(make_adb(element, ion), np.linspace(12000.0, 18000.0, 21))


def test_alkali_rejects_empty_selection(make_adb):
    with pytest.warns(UserWarning, match="no lines"):
        adb = make_adb(99)
    with pytest.raises(ValueError):
        OpaAlkali(adb, np.linspace(12000.0, 18000.0, 21))


@pytest.mark.parametrize("dbtype", ["exomol", "nist"])
def test_alkali_rejects_other_databases(dbtype):
    with pytest.raises(ValueError):
        OpaAlkali(SimpleNamespace(dbtype=dbtype), np.linspace(12000.0, 18000.0, 21))
