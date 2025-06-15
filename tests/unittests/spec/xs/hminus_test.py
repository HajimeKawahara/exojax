import numpy as np
import jax.numpy as jnp
import pytest
from exojax.database.hminus  import free_free_absorption
from exojax.database.hminus  import bound_free_absorption
from exojax.database.hminus  import log_hminus_continuum
from exojax.database.hminus  import log_hminus_continuum_single
from exojax.rt.layeropacity import layer_optical_depth_Hminus
from exojax.rt.layeropacity import single_layer_optical_depth_Hminus
from exojax.opacity import OpaHminus
from exojax.utils.grids import wavenumber_grid


def test_hminus_ff():
    Tin = 3000.0
    wav = 1.4
    ref = 2.0075e-26
    val = free_free_absorption(wav, Tin)
    diff = np.abs(ref - val)
    assert diff < 1.0e-30


def test_hminus_bf():
    Tin = 3000.0
    wav = 1.4
    ref = 4.065769e-25
    val = bound_free_absorption(wav, Tin)
    diff = np.abs(ref - val)
    print(diff)
    assert diff < 1.0e-30


def _setting_test_hminus():
    """setting for test_log_hminus_continuum, test_log_hminus_continuum_single

    Returns:
        temperatures, number_density_e, number_density_h, nu_grid, ref_value
    """
    N = 1000
    temperatures = jnp.array([3000.0])
    number_density_e = jnp.array([1.0])
    number_density_h = jnp.array([1.0])
    nu_grid, wav, res = wavenumber_grid(
        9000.0, 18000.0, N, xsmode="premodit", unit="AA"
    )
    ref_value = -36814.26
    return temperatures, number_density_e, number_density_h, nu_grid, ref_value


def test_log_hminus_continuum():
    temperatures, number_density_e, number_density_h, nu_grid, ref_value = (
        _setting_test_hminus()
    )
    a = log_hminus_continuum(nu_grid, temperatures, number_density_e, number_density_h)

    assert a.shape == (1, 1000)
    assert np.sum(a) == pytest.approx(ref_value)


def test_log_hminus_continuum_single():
    temperatures, number_density_e, number_density_h, nu_grid, ref_value = (
        _setting_test_hminus()
    )
    a = log_hminus_continuum_single(
        nu_grid, temperatures[0], number_density_e[0], number_density_h[0]
    )

    assert a.shape == (1000,)
    assert np.sum(a) == pytest.approx(ref_value)


def test_opahminus():
    Tarr, ne, nh, nu_grid, ref_value = _setting_test_hminus()
    opa = OpaHminus(nu_grid)
    a = opa.logahminus_matrix(Tarr, ne, nh)
    assert np.sum(a) == pytest.approx(ref_value)


def _setting_test_layer_hminus_dtau():
    N = 1000
    nu_grid, wav, res = wavenumber_grid(
        9000.0, 18000.0, N, xsmode="premodit", unit="AA"
    )
    temperatures = np.array([3000.0])  # K
    pressures = np.array([1.0])  # bar
    dParr = np.array([0.01])  # bar
    vmre = np.array([0.001])
    vmrh = np.array([0.001])
    mmw = 2.3
    gravity = 1.0e5
    ref_value = 16.029575
    return nu_grid, temperatures, pressures, dParr, vmre, vmrh, mmw, gravity, ref_value


def test_layer_optical_depth_Hminus():
    nu_grid, temperatures, pressures, dParr, vmre, vmrh, mmw, gravity, ref_value = (
        _setting_test_layer_hminus_dtau()
    )
    a = layer_optical_depth_Hminus(
        nu_grid, temperatures, pressures, dParr, vmre, vmrh, mmw, gravity
    )

    assert a.shape == (1, 1000)
    assert np.sum(a) == pytest.approx(ref_value)


def test_single_layer_optical_depth_Hminus():
    nu_grid, temperatures, pressures, dParr, vmre, vmrh, mmw, gravity, ref_value = (
        _setting_test_layer_hminus_dtau()
    )
    a = single_layer_optical_depth_Hminus(
        nu_grid, temperatures[0], pressures[0], dParr[0], vmre[0], vmrh[0], mmw, gravity
    )

    assert a.shape == (1000,)
    assert np.sum(a) == pytest.approx(ref_value)


if __name__ == "__main__":
    test_hminus_ff()
    test_hminus_bf()
    test_log_hminus_continuum()
    test_log_hminus_continuum_single()
    test_layer_optical_depth_Hminus()
    test_single_layer_optical_depth_Hminus()
    test_opahminus()
