import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt import OpartEmisScat, OpartReflectEmis, OpartReflectPure
from exojax.rt.planck import piB, piBarr
from exojax.rt.rtransfer import (
    rtrun_emis_scat_fluxadding_toonhm,
    rtrun_reflect_fluxadding_toonhm,
)


class MockOpaLayer:
    def __init__(self):
        self.nu_grid = jnp.array([1000.0])

    def __call__(self, params):
        _, dtau, single_scattering_albedo, asymmetric_parameter = params
        return dtau, single_scattering_albedo, asymmetric_parameter


def layer_update_function(opart):
    def update(carry, params):
        return opart.update_layer(carry, params), None

    return update


def layer_params_top_to_bottom():
    temperature = jnp.array([500.0, 1000.0])
    dtau = jnp.array([0.2, 2.0])
    single_scattering_albedo = jnp.array([0.3, 0.7])
    asymmetric_parameter = jnp.array([0.0, 0.2])
    return [temperature, dtau, single_scattering_albedo, asymmetric_parameter]


@pytest.mark.parametrize("opart_class", [OpartEmisScat, OpartReflectEmis])
def test_opart_layer_uses_full_thermal_source(opart_class):
    opalayer = MockOpaLayer()
    opart = opart_class(opalayer, nlayer=2)
    temperature = 1000.0
    dtau = jnp.array([1.0e-3])
    params = [temperature, dtau, jnp.zeros(1), jnp.zeros(1)]
    initial_carry = [jnp.zeros(1), jnp.zeros(1)]

    _, actual_source = opart.update_layer(initial_carry, params)

    expected_source = (1.0 - jnp.exp(-2.0 * dtau)) * piB(
        temperature, opalayer.nu_grid
    )
    np.testing.assert_allclose(actual_source, expected_source, rtol=1.0e-5)


def test_opart_emis_scat_matches_batch_for_top_to_bottom_layers():
    opalayer = MockOpaLayer()
    opart = OpartEmisScat(opalayer, nlayer=2)
    layer_params = layer_params_top_to_bottom()

    actual = opart(layer_params, layer_update_function(opart))
    temperature, dtau, single_scattering_albedo, asymmetric_parameter = layer_params
    expected = rtrun_emis_scat_fluxadding_toonhm(
        dtau[:, None],
        single_scattering_albedo[:, None],
        asymmetric_parameter[:, None],
        piBarr(temperature, opalayer.nu_grid),
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_opart_reflect_pure_matches_batch_for_top_to_bottom_layers():
    opalayer = MockOpaLayer()
    opart = OpartReflectPure(opalayer, nlayer=2)
    layer_params = layer_params_top_to_bottom()
    reflectivity_bottom = jnp.array([0.4])
    incoming_flux = jnp.ones(1)

    actual = opart(
        layer_params,
        layer_update_function(opart),
        reflectivity_bottom,
        incoming_flux,
    )
    _, dtau, single_scattering_albedo, asymmetric_parameter = layer_params
    expected = rtrun_reflect_fluxadding_toonhm(
        dtau[:, None],
        single_scattering_albedo[:, None],
        asymmetric_parameter[:, None],
        jnp.zeros((2, 1)),
        jnp.zeros(1),
        reflectivity_bottom,
        incoming_flux,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_opart_reflect_emis_matches_batch_for_top_to_bottom_layers():
    opalayer = MockOpaLayer()
    opart = OpartReflectEmis(opalayer, nlayer=2)
    layer_params = layer_params_top_to_bottom()
    source_bottom = jnp.array([0.1])
    reflectivity_bottom = jnp.array([0.4])
    incoming_flux = jnp.ones(1)

    actual = opart(
        layer_params,
        layer_update_function(opart),
        source_bottom,
        reflectivity_bottom,
        incoming_flux,
    )
    temperature, dtau, single_scattering_albedo, asymmetric_parameter = layer_params
    expected = rtrun_reflect_fluxadding_toonhm(
        dtau[:, None],
        single_scattering_albedo[:, None],
        asymmetric_parameter[:, None],
        piBarr(temperature, opalayer.nu_grid),
        source_bottom,
        reflectivity_bottom,
        incoming_flux,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_reflect_fluxadding_toonhm_conservative_scattering_limit():
    asymmetric_parameter = jnp.array([[0.0, 0.5, 1.0]])
    reflected_flux = rtrun_reflect_fluxadding_toonhm(
        jnp.ones((1, 3)),
        jnp.ones((1, 3)),
        asymmetric_parameter,
        jnp.zeros((1, 3)),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.ones(3),
    )

    expected = (1.0 - asymmetric_parameter[0]) / (2.0 - asymmetric_parameter[0])
    np.testing.assert_allclose(reflected_flux, expected, rtol=1.0e-6)
