import jax
import jax.numpy as jnp

from exojax.rt.toon import zetalambda_coeffs
from exojax.rt.toon import reduced_source_function_isothermal_layer
from exojax.rt.toon import reduced_source_function
from exojax.rt.toon import params_eddington
from exojax.rt.toon import params_quadrature
from exojax.rt.toon import params_hemispheric_mean
from exojax.rt.twostream import (
    set_scat_trans_absorption_coeffs,
    set_scat_trans_coeffs,
)


def test_zetalambda_coeffs():
    gamma_1 = 2.0
    gamma_2 = 1.0
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(gamma_1, gamma_2)
    zeta_plus_ref = 0.7886751
    zeta_minus_ref = 0.21132487
    lambdan_ref = 1.7320508
    assert jnp.isclose(zeta_plus, zeta_plus_ref), f"Expected 0.75, got {zeta_plus}"
    assert jnp.isclose(zeta_minus, zeta_minus_ref), f"Expected 0.25, got {zeta_minus}"
    assert jnp.isclose(lambdan, lambdan_ref), f"Expected {jnp.sqrt(3.0)}, got {lambdan}"


def test_zetalambda_coeffs_zero_gamma():
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(0.0, 0.0)

    assert jnp.isclose(zeta_plus, 0.5)
    assert jnp.isclose(zeta_minus, 0.5)
    assert jnp.isclose(lambdan, 0.0)


def test_absorption_coeff_zero_depth_value_and_gradient():
    def absorption(dtau):
        return set_scat_trans_absorption_coeffs(1.5, 0.5, dtau)[2]

    dtau = jnp.float32(0.0)
    assert jnp.isclose(absorption(dtau), 0.0)
    assert jnp.isclose(jax.grad(absorption)(dtau), 1.0)


def test_legacy_scat_trans_coeff_api_matches_full_coefficients():
    gamma_1 = jnp.full(2, 1.5, dtype=jnp.float32)
    gamma_2 = jnp.full(2, 0.5, dtype=jnp.float32)
    dtau = jnp.array([1.0e-8, 0.1], dtype=jnp.float32)

    legacy_coeffs = set_scat_trans_coeffs(gamma_1, gamma_2, dtau)
    full_coeffs = set_scat_trans_absorption_coeffs(
        gamma_1, gamma_2, dtau
    )
    jitted_legacy_coeffs = jax.jit(set_scat_trans_coeffs)(
        gamma_1, gamma_2, dtau
    )

    assert len(legacy_coeffs) == 2
    assert len(full_coeffs) == 3
    for legacy, full, jitted in zip(
        legacy_coeffs, full_coeffs[:2], jitted_legacy_coeffs
    ):
        assert jnp.array_equal(legacy, full)
        assert jnp.array_equal(jitted, full)


def test_thick_conservative_scattering_coefficients_float32():
    thin_dtau = jnp.float32(1.0e-8)
    dtau = jnp.float32(2**24)

    for coefficient_function in (
        set_scat_trans_coeffs,
        set_scat_trans_absorption_coeffs,
    ):
        thin_scat_coeff = coefficient_function(
            1.0, 1.0, thin_dtau
        )[1]
        assert thin_scat_coeff > 0.0
        assert jnp.isclose(thin_scat_coeff, thin_dtau)

    legacy_coeffs = set_scat_trans_coeffs(1.0, 1.0, dtau)
    full_coeffs = set_scat_trans_absorption_coeffs(1.0, 1.0, dtau)

    for trans_coeff, scat_coeff in (legacy_coeffs, full_coeffs[:2]):
        assert trans_coeff > 0.0
        assert scat_coeff < 1.0
        assert scat_coeff == 1.0 - trans_coeff
    assert full_coeffs[2] == 0.0

    grad_trans, grad_scat, grad_absorption = jax.jacfwd(
        lambda depth: set_scat_trans_absorption_coeffs(1.0, 1.0, depth)
    )(dtau)
    assert jnp.isfinite(grad_trans)
    assert grad_scat == -grad_trans
    assert grad_absorption == 0.0
    legacy_depth_grad = jax.grad(
        lambda depth: set_scat_trans_coeffs(1, 1, depth)[1]
    )(dtau)
    assert legacy_depth_grad == grad_scat

    def scattering_coeff(single_scattering_albedo, coefficient_function):
        return coefficient_function(
            2.0 - single_scattering_albedo,
            single_scattering_albedo,
            dtau,
        )[1]

    legacy_grad = jax.grad(
        lambda albedo: scattering_coeff(albedo, set_scat_trans_coeffs)
    )(jnp.float32(1.0))
    full_grad = jax.grad(
        lambda albedo: scattering_coeff(
            albedo, set_scat_trans_absorption_coeffs
        )
    )(jnp.float32(1.0))
    assert legacy_grad > 0.0
    assert jnp.isclose(legacy_grad, full_grad)

    def thick_nonconservative_scat(gamma_1, coefficient_function):
        return coefficient_function(
            gamma_1,
            jnp.float32(0.21573891),
            jnp.float32(89321872.0),
        )[1]

    gamma_1 = jnp.float32(1.9849839)
    thick_nonconservative_grad = jax.grad(
        lambda value: thick_nonconservative_scat(
            value, set_scat_trans_coeffs
        )
    )(gamma_1)
    expected_grad = jax.grad(
        lambda value: thick_nonconservative_scat(
            value, set_scat_trans_absorption_coeffs
        )
    )(gamma_1)
    assert jnp.isfinite(thick_nonconservative_grad)
    assert jnp.isclose(thick_nonconservative_grad, expected_grad)


def test_reduced_source_function_isothermal_layer():
    single_scattering_albedo = 0.5
    gamma_1 = 2.0
    gamma_2 = 1.0
    source_function = 3.0
    result = reduced_source_function_isothermal_layer(
        single_scattering_albedo, gamma_1, gamma_2, source_function
    )
    expected_result = (
        2.0 * (1.0 - single_scattering_albedo) / (gamma_1 - gamma_2) * source_function
    )

    assert jnp.isclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"


def test_reduced_source_function():
    single_scattering_albedo = 0.5
    gamma_1 = 2.0
    gamma_2 = 1.0
    source_function = 3.0
    source_function_derivative = 0.5

    result_plus = reduced_source_function(
        single_scattering_albedo,
        gamma_1,
        gamma_2,
        source_function,
        source_function_derivative,
        sign=1.0,
    )
    expected_result_plus = (
        2.0
        * (1.0 - single_scattering_albedo)
        / (gamma_1 - gamma_2)
        * (source_function + source_function_derivative / (gamma_1 + gamma_2))
    )

    result_minus = reduced_source_function(
        single_scattering_albedo,
        gamma_1,
        gamma_2,
        source_function,
        source_function_derivative,
        sign=-1.0,
    )
    expected_result_minus = (
        2.0
        * (1.0 - single_scattering_albedo)
        / (gamma_1 - gamma_2)
        * (source_function - source_function_derivative / (gamma_1 + gamma_2))
    )

    assert jnp.isclose(
        result_plus, expected_result_plus
    ), f"Expected {expected_result_plus}, got {result_plus}"
    assert jnp.isclose(
        result_minus, expected_result_minus
    ), f"Expected {expected_result_minus}, got {result_minus}"


def test_params_eddington():
    single_scattering_albedo = 0.5
    asymmetric_parameter = 0.3
    mu0 = 0.8

    gamma_1, gamma_2, gamma_3, mu1 = params_eddington(
        single_scattering_albedo, asymmetric_parameter, mu0
    )

    expected_gamma_1 = (
        7.0 - single_scattering_albedo * (4.0 + 3.0 * asymmetric_parameter)
    ) / 4.0
    expected_gamma_2 = (
        -(1.0 - single_scattering_albedo * (4.0 - 3.0 * asymmetric_parameter)) / 4.0
    )
    expected_gamma_3 = (2.0 - 3.0 * asymmetric_parameter * mu0) / 4.0
    expected_mu1 = 0.5

    assert jnp.isclose(
        gamma_1, expected_gamma_1
    ), f"Expected {expected_gamma_1}, got {gamma_1}"
    assert jnp.isclose(
        gamma_2, expected_gamma_2
    ), f"Expected {expected_gamma_2}, got {gamma_2}"
    assert jnp.isclose(
        gamma_3, expected_gamma_3
    ), f"Expected {expected_gamma_3}, got {gamma_3}"
    assert jnp.isclose(mu1, expected_mu1), f"Expected {expected_mu1}, got {mu1}"


def test_params_quadrature():
    single_scattering_albedo = 0.5
    asymmetric_parameter = 0.3
    mu0 = 0.8

    gamma_1, gamma_2, gamma_3, mu1 = params_quadrature(
        single_scattering_albedo, asymmetric_parameter, mu0
    )

    s3 = jnp.sqrt(3.0)
    expected_gamma_1 = (
        s3 * (2.0 - single_scattering_albedo * (1.0 + asymmetric_parameter)) / 2.0
    )
    expected_gamma_2 = (
        single_scattering_albedo * s3 * (1.0 - asymmetric_parameter) / 2.0
    )
    expected_gamma_3 = (1.0 - s3 * asymmetric_parameter * mu0) / 2.0
    expected_mu1 = 1.0 / s3

    assert jnp.isclose(
        gamma_1, expected_gamma_1
    ), f"Expected {expected_gamma_1}, got {gamma_1}"
    assert jnp.isclose(
        gamma_2, expected_gamma_2
    ), f"Expected {expected_gamma_2}, got {gamma_2}"
    assert jnp.isclose(
        gamma_3, expected_gamma_3
    ), f"Expected {expected_gamma_3}, got {gamma_3}"
    assert jnp.isclose(mu1, expected_mu1), f"Expected {expected_mu1}, got {mu1}"


def test_params_hemispheric_mean():
    single_scattering_albedo = 0.5
    asymmetric_parameter = 0.3

    gamma_1, gamma_2, mu1 = params_hemispheric_mean(
        single_scattering_albedo, asymmetric_parameter
    )

    expected_gamma_1 = 2.0 - single_scattering_albedo * (1.0 + asymmetric_parameter)
    expected_gamma_2 = single_scattering_albedo * (1.0 - asymmetric_parameter)
    expected_mu1 = 0.5

    assert jnp.isclose(
        gamma_1, expected_gamma_1
    ), f"Expected {expected_gamma_1}, got {gamma_1}"
    assert jnp.isclose(
        gamma_2, expected_gamma_2
    ), f"Expected {expected_gamma_2}, got {gamma_2}"
    assert jnp.isclose(mu1, expected_mu1), f"Expected {expected_mu1}, got {mu1}"


if __name__ == "__main__":
    test_reduced_source_function_isothermal_layer()
    test_zetalambda_coeffs()
    test_reduced_source_function()
    test_params_eddington()
    test_params_quadrature()
    test_params_hemispheric_mean()
