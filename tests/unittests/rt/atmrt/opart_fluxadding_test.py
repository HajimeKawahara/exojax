import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt import OpartEmisScat, OpartReflectEmis, OpartReflectPure
from exojax.rt.planck import piB, piBarr
from exojax.rt.rtransfer import (
    rtrun_emis_scat_lart_toonhm,
    rtrun_emis_scat_lart_toonhm_surface,
    rtrun_emis_scat_fluxadding_toonhm,
    rtrun_reflect_fluxadding_toonhm,
    setrt_toonhm,
    setrt_toonhm_with_absorption,
    settridiag_toohm,
)
from exojax.rt.twostream import (
    solve_fluxadding_twostream,
    solve_fluxadding_twostream_fluxes,
    solve_lart_twostream,
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


def thin_toon_inputs(single_scattering_albedo, dtau_value):
    shape = (100, 1)
    dtau = jnp.full(shape, dtau_value, dtype=jnp.float32)
    albedo = jnp.full(shape, single_scattering_albedo, dtype=jnp.float32)
    asymmetry = jnp.zeros(shape, dtype=jnp.float32)
    source = jnp.ones(shape, dtype=jnp.float32)
    boundary = jnp.zeros(1, dtype=jnp.float32)
    return dtau, albedo, asymmetry, source, boundary


@pytest.mark.parametrize("opart_class", [OpartEmisScat, OpartReflectEmis])
def test_opart_layer_uses_full_thermal_source(opart_class):
    opalayer = MockOpaLayer()
    opart = opart_class(opalayer, nlayer=2)
    temperature = 1000.0
    dtau = jnp.array([1.0e-8], dtype=jnp.float32)
    zeros = jnp.zeros(1, dtype=jnp.float32)
    params = [temperature, dtau, zeros, zeros]
    initial_carry = [zeros, zeros]

    _, actual_source = opart.update_layer(initial_carry, params)

    expected_source = -jnp.expm1(-2.0 * dtau) * piB(
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


@pytest.mark.parametrize(
    "single_scattering_albedo, dtau_value, expected",
    [
        (0.5, 1.0e-8, 9.9999948118e-7),
        (0.99, 1.0e-6, 1.9999980088e-6),
    ],
)
def test_fluxadding_toonhm_thin_float32_layers(
    single_scattering_albedo, dtau_value, expected
):
    dtau, albedo, asymmetry, source, boundary = thin_toon_inputs(
        single_scattering_albedo, dtau_value
    )

    actual = rtrun_emis_scat_fluxadding_toonhm(
        dtau, albedo, asymmetry, source
    )
    reflected_emission = rtrun_reflect_fluxadding_toonhm(
        dtau,
        albedo,
        asymmetry,
        source,
        boundary,
        boundary,
        boundary,
    )

    np.testing.assert_allclose(actual, expected, rtol=5.0e-6, atol=0.0)
    np.testing.assert_allclose(
        reflected_emission, expected, rtol=5.0e-6, atol=0.0
    )


def test_fluxadding_toonhm_thin_float32_layers_both_directions():
    dtau, albedo, asymmetry, source, boundary = thin_toon_inputs(0.5, 1.0e-8)
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, albedo, asymmetry, source
    )
    trans_coeff, scat_coeff, absorption_coeff, reduced_source = toon_coeffs[:4]

    flux_plus, flux_minus = solve_fluxadding_twostream_fluxes(
        trans_coeff,
        scat_coeff,
        reduced_source,
        boundary,
        boundary,
        absorption_coeff=absorption_coeff,
    )

    expected = 9.9999948118e-7
    np.testing.assert_allclose(flux_plus[0], expected, rtol=5.0e-6, atol=0.0)
    np.testing.assert_allclose(flux_minus[-1], expected, rtol=5.0e-6, atol=0.0)


def test_lart_toonhm_thin_float32_layers():
    dtau, albedo, asymmetry, source, source_surface = thin_toon_inputs(
        0.5, 1.0e-8
    )

    spectrum = rtrun_emis_scat_lart_toonhm(
        dtau, albedo, asymmetry, source
    )[0]
    spectrum_surface = rtrun_emis_scat_lart_toonhm_surface(
        dtau, albedo, asymmetry, source, source_surface
    )[0]

    expected = 9.9999948118e-7
    np.testing.assert_allclose(spectrum, expected, rtol=2.0e-6, atol=0.0)
    np.testing.assert_allclose(
        spectrum_surface, expected, rtol=2.0e-6, atol=0.0
    )


def test_fluxadding_toonhm_thick_conservative_scattering():
    def thermal_flux(layer_depth):
        shape = (2, 1)
        dtau = jnp.full(shape, layer_depth, dtype=jnp.float32)
        albedo = jnp.ones(shape, dtype=jnp.float32)
        asymmetry = jnp.zeros(shape, dtype=jnp.float32)
        source = jnp.ones(shape, dtype=jnp.float32)
        return rtrun_emis_scat_fluxadding_toonhm(
            dtau, albedo, asymmetry, source
        )[0]

    def reflected_flux(layer_depth):
        shape = (1, 1)
        dtau = jnp.full(shape, layer_depth, dtype=jnp.float32)
        albedo = jnp.ones(shape, dtype=jnp.float32)
        asymmetry = jnp.zeros(shape, dtype=jnp.float32)
        source = jnp.ones(shape, dtype=jnp.float32)
        zeros = jnp.zeros(1, dtype=jnp.float32)
        ones = jnp.ones(1, dtype=jnp.float32)
        return rtrun_reflect_fluxadding_toonhm(
            dtau,
            albedo,
            asymmetry,
            source,
            zeros,
            ones,
            ones,
        )[0]

    layer_depth = jnp.float32(2**24)
    np.testing.assert_array_equal(thermal_flux(layer_depth), 0.0)
    np.testing.assert_array_equal(reflected_flux(layer_depth), 1.0)
    np.testing.assert_array_equal(
        thermal_flux(jnp.float32(2**25)), 0.0
    )
    np.testing.assert_array_equal(
        reflected_flux(jnp.float32(2**25)), 1.0
    )

    dtau = jnp.full((2, 1), layer_depth, dtype=jnp.float32)
    albedo = jnp.ones_like(dtau)
    asymmetry = jnp.zeros_like(dtau)
    source = jnp.ones_like(dtau)
    boundary = jnp.zeros(1, dtype=jnp.float32)
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, albedo, asymmetry, source
    )
    _, transmitted_bottom_source = solve_fluxadding_twostream(
        toon_coeffs[0],
        toon_coeffs[1],
        toon_coeffs[3],
        boundary,
        jnp.ones(1, dtype=jnp.float32),
        absorption_coeff=toon_coeffs[2],
    )
    expected_transmission = jnp.array(
        [1.0 / (1.0 + 2.0 * layer_depth)], dtype=jnp.float32
    )
    np.testing.assert_allclose(
        transmitted_bottom_source, expected_transmission, rtol=2.0e-6
    )

    flux_plus, flux_minus = solve_fluxadding_twostream_fluxes(
        toon_coeffs[0],
        toon_coeffs[1],
        toon_coeffs[3],
        boundary,
        boundary,
        absorption_coeff=toon_coeffs[2],
    )
    np.testing.assert_array_equal(flux_plus, jnp.zeros_like(flux_plus))
    np.testing.assert_array_equal(flux_minus, jnp.zeros_like(flux_minus))


def test_lart_toonhm_thick_conservative_scattering_legacy_coefficients():
    depth = 2**24
    shape = (2, 1)
    dtau = jnp.full(shape, depth, dtype=jnp.float32)
    albedo = jnp.ones(shape, dtype=jnp.float32)
    asymmetry = jnp.zeros(shape, dtype=jnp.float32)
    source = jnp.ones(shape, dtype=jnp.float32)

    trans_coeff, scat_coeff, reduced_source, zeta_plus, zeta_minus, lambdan = (
        setrt_toonhm(dtau, albedo, asymmetry, source)
    )
    diagonal, lower_diagonal, upper_diagonal, vector = settridiag_toohm(
        dtau,
        zeta_plus,
        zeta_minus,
        lambdan,
        trans_coeff,
        scat_coeff,
        reduced_source,
    )
    cumulative_transmission, source_terms, spectrum = solve_lart_twostream(
        diagonal,
        lower_diagonal,
        upper_diagonal,
        vector,
        jnp.zeros(1, dtype=jnp.float32),
    )

    assert jnp.all(jnp.isfinite(cumulative_transmission))
    assert jnp.all(jnp.isfinite(source_terms))
    expected_transmission = jnp.array(
        [1.0, 1.0 / (1.0 + depth), 1.0 / (1.0 + 2.0 * depth)],
        dtype=jnp.float32,
    )[:, None]
    np.testing.assert_allclose(
        cumulative_transmission,
        expected_transmission,
        rtol=2.0e-6,
    )
    np.testing.assert_array_equal(source_terms, jnp.zeros_like(source_terms))
    np.testing.assert_array_equal(spectrum, jnp.zeros_like(spectrum))
    np.testing.assert_array_equal(
        rtrun_emis_scat_lart_toonhm(
            dtau, albedo, asymmetry, source
        )[0],
        jnp.zeros(1, dtype=jnp.float32),
    )


def test_fluxadding_uses_explicit_absorption_in_denominator():
    trans_coeff = jnp.full((1, 1), 2.0**-27, dtype=jnp.float32)
    scat_coeff = jnp.ones((1, 1), dtype=jnp.float32)
    absorption_coeff = jnp.float32(2.0**-26)
    source = jnp.ones((1, 1), dtype=jnp.float32)

    _, effective_source = solve_fluxadding_twostream(
        trans_coeff,
        scat_coeff,
        source,
        jnp.ones(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        absorption_coeff=absorption_coeff,
    )

    expected = absorption_coeff + (
        trans_coeff[0, 0]
        * absorption_coeff
        / (trans_coeff[0, 0] + absorption_coeff)
    )
    np.testing.assert_array_equal(effective_source, expected)
