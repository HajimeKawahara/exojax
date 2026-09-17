import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import vald, xsmatrix


@pytest.fixture(params=["vald", "kurucz"])
def make_adb(request, monkeypatch, tmp_path):
    """Construct real adapters for Fe I, Fe II, and Na I transitions."""
    api = importlib.import_module(f"exojax.database.{request.param}.api")
    transitions = (
        [1.0e6, 2.0e6, 3.0e6, 4.0e6], [1000.0, 1001.0, 1002.0, 1003.0],
        [0.0, 10.0, 20.0, 30.0], [1000.0, 1011.0, 1022.0, 1033.0],
        [3.0, 4.0, 4.0, 7.0], [0.0, 0.5, 0.5, 2.0], [1.0, 1.5, 1.5, 3.0],
        [26, 26, 11, 26], [1, 2, 1, 1],
        [8.0, 8.1, 8.2, 8.3], [-6.0, -6.1, -6.2, -6.3],
        [-7.0, -7.1, -7.2, -7.3],
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
    return lambda **kwargs: adb_class(tmp_path / "synthetic.lines", **kwargs)


@pytest.mark.parametrize("irwin", [False, True])
def test_opadirect_atomic_vectors_match_matrix_and_lowlevel_lpf(make_adb, irwin):
    adb = make_adb(Irwin=irwin, vmr_fraction=[0.1, 0.2, 0.7])
    nu_grid = np.linspace(999.8, 1003.2, 86)
    opa = OpaDirect(adb, nu_grid=nu_grid)
    temperatures = jnp.array([1300.0, 1800.0])
    pressures = jnp.array([0.01, 0.1])

    actual = opa.xsmatrix(temperatures, pressures)
    vectors = jnp.stack([
        opa.xsvector(temperature, pressure)
        for temperature, pressure in zip(temperatures, pressures)
    ])
    strengths, gammas, sigmas = vald(
        adb, temperatures, pressures * adb.vmrH,
        pressures * adb.vmrHe, pressures * adb.vmrHH,
    )
    expected = xsmatrix(opa.opainfo, sigmas, gammas, strengths)

    assert len(adb.nu_lines) != len(temperatures)
    assert actual.shape == (2, len(nu_grid))
    assert np.all(np.isfinite(actual))
    assert np.all(actual > 0.0)
    np.testing.assert_allclose(actual, vectors, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)


def test_opadirect_atomic_temperature_and_pressure_gradients(make_adb):
    adb = make_adb(Irwin=True, vmr_fraction=[0.1, 0.2, 0.7])
    nu_grid = np.linspace(999.8, 1003.2, 86)
    opa = OpaDirect(adb, nu_grid=nu_grid)
    line_center = np.argmin(np.abs(nu_grid - adb.nu_lines[0]))

    def log_opacity(temperature, pressure):
        return jnp.log(opa.xsvector(temperature, pressure)[line_center])

    def log_matrix_opacity(temperature, pressure):
        matrix = opa.xsmatrix(
            jnp.array([temperature, 1300.0]), jnp.array([pressure, 0.01])
        )
        return jnp.log(matrix[0, line_center])

    compiled = jax.jit(log_opacity)
    temperature, pressure = 1800.0, 0.1
    np.testing.assert_allclose(
        compiled(temperature, pressure), log_opacity(temperature, pressure),
        rtol=1.0e-12,
    )
    actual = jax.jit(jax.grad(log_opacity, argnums=(0, 1)))(temperature, pressure)
    matrix_value, matrix_grad = jax.jit(
        jax.value_and_grad(log_matrix_opacity, argnums=(0, 1))
    )(temperature, pressure)
    np.testing.assert_allclose(matrix_value, compiled(temperature, pressure), rtol=1.0e-12)
    np.testing.assert_allclose(matrix_grad, actual, rtol=1.0e-10, atol=1.0e-12)
    temperature_step, pressure_step = 0.1, 1.0e-5
    expected = (
        (compiled(temperature + temperature_step, pressure)
         - compiled(temperature - temperature_step, pressure)) / (2 * temperature_step),
        (compiled(temperature, pressure + pressure_step)
         - compiled(temperature, pressure - pressure_step)) / (2 * pressure_step),
    )

    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) > 1.0e-8)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-9)


def test_opadirect_atomic_broadening_override(make_adb):
    from exojax.database.core.broadening import doppler_sigma
    from exojax.database.core.line_strength import line_strength
    from exojax.opacity.lpf.lpf import xsvector

    adb = make_adb()
    broadening = lambda T, P: jnp.full_like(adb.A, 0.02)
    opa = OpaDirect(
        adb, np.linspace(999.8, 1002.2, 61), atomic_broadening=broadening
    )
    temperature, pressure = 1800.0, 0.1
    strengths = line_strength(
        temperature, adb.logsij0, adb.nu_lines, adb.elower,
        adb.qr_interp_lines(temperature, adb.Tref), adb.Tref,
    )
    expected = xsvector(
        opa.opainfo, doppler_sigma(adb.nu_lines, temperature, adb.line_masses),
        broadening(temperature, pressure), strengths,
    )
    np.testing.assert_allclose(
        opa.xsvector(temperature, pressure), expected, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("irwin", [False, True])
def test_separated_atomic_line_parameters_preserve_values_and_padding(make_adb, irwin):
    from exojax.database import AdbSepVald
    from exojax.database.core_atom.pf import interp_QT_284
    from exojax.opacity.lpf.lpf import vald_each
    from exojax.opacity.modit.modit import vald_all

    adb = make_adb(Irwin=irwin)
    asdb = AdbSepVald(adb)
    temperatures = jnp.array([1300.0, 1800.0, 2300.0])
    pressures = jnp.array([0.01, 0.1, 1.0])
    ph, phe, phh = pressures * 0.1, pressures * 0.2, pressures * 0.7
    resolution = 1.0e5
    expected = vald(adb, temperatures, ph, phe, phh)
    strengths, ngamma, nsigma = vald_all(asdb, temperatures, ph, phe, phh, resolution)
    partition = jax.vmap(interp_QT_284, (0, None, None, None))(
        temperatures, adb.T_gQT, adb.gQT_284species, irwin,
    )

    assert asdb.L_max == 2
    assert np.any(asdb.nu_lines == 0.0)
    for index, (element, ion) in enumerate(np.asarray(asdb.uspecies)):
        selected = np.asarray((adb.ielem == element) & (adb.iion == ion))
        valid = asdb.nu_lines[index] != 0.0
        widths = asdb.nu_lines[index, valid] / resolution
        actual = (
            strengths[index][:, valid],
            ngamma[index][:, valid] * widths,
            nsigma[index] * widths,
        )
        for values, reference in zip(actual, expected):
            np.testing.assert_allclose(values, reference[:, selected], rtol=1.0e-12, atol=0.0)

        padded = vald_each(
            temperatures, ph, phe, phh, partition, asdb.QTmask[index], asdb.QTref_284,
            asdb.logsij0[index], asdb.nu_lines[index], element, ion,
            asdb.dev_nu_lines[index], asdb.elower[index], asdb.eupper[index],
            asdb.atomicmass[index], asdb.ionE[index], asdb.gamRad[index],
            asdb.gamSta[index], asdb.vdWdamp[index], asdb.Tref,
        )
        for values, reference in zip(padded, expected):
            np.testing.assert_allclose(values[:, valid], reference[:, selected], rtol=1.0e-12, atol=0.0)
        assert np.all(~np.isfinite(ngamma[index][:, ~valid]))
        assert np.all(np.asarray(padded[0])[:, ~valid] == 0.0)
        assert np.all(np.asarray(padded[2])[:, ~valid] == 1.0)
