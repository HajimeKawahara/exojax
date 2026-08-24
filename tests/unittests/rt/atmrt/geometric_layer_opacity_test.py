"""Tests for optical depth integrated over geometric layer thickness."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt.layeropacity import layer_optical_depth_from_cross_section
from exojax.rt.layeropacity import layer_optical_depth_from_extinction
from exojax.rt.layeropacity import layer_optical_depth_from_log_cia


def test_cross_section_common_spectrum_matches_analytic_equation():
    cross_section = jnp.array([1.0e-20, 3.0e-20, 5.0e-20])
    number_density = jnp.array([2.0e10, 4.0e10])
    layer_height = jnp.array([1.0e5, 3.0e5])

    optical_depth = layer_optical_depth_from_cross_section(
        cross_section, number_density, layer_height
    )
    expected = (
        cross_section[None, :]
        * number_density[:, None]
        * layer_height[:, None]
    )

    assert optical_depth.shape == (2, 3)
    np.testing.assert_allclose(optical_depth, expected)


def test_cross_section_layer_resolved_lbl_array():
    cross_section = jnp.array(
        [[1.0e-20, 2.0e-20, 3.0e-20], [4.0e-20, 5.0e-20, 6.0e-20]]
    )
    number_density = 2.0e10
    layer_height = jnp.array([1.0e5, 2.0e5])

    optical_depth = layer_optical_depth_from_cross_section(
        cross_section, number_density, layer_height
    )
    expected = cross_section * number_density * layer_height[:, None]

    assert optical_depth.shape == cross_section.shape
    np.testing.assert_allclose(optical_depth, expected)


def test_cross_section_layer_resolved_ckd_array():
    cross_section = jnp.arange(1.0, 13.0).reshape((2, 2, 3)) * 1.0e-20
    number_density = jnp.array([2.0e10, 4.0e10])
    layer_height = jnp.array([1.0e5, 2.0e5])

    optical_depth = layer_optical_depth_from_cross_section(
        cross_section, number_density, layer_height
    )
    expected = (
        cross_section
        * number_density[:, None, None]
        * layer_height[:, None, None]
    )

    assert optical_depth.shape == cross_section.shape
    np.testing.assert_allclose(optical_depth, expected)


def test_log_cia_matches_analytic_equation_for_layered_ckd_array():
    log_cia = jnp.log10(jnp.arange(1.0, 13.0).reshape((2, 2, 3)) * 1.0e-45)
    number_density_1 = jnp.array([2.0e19, 3.0e19])
    number_density_2 = jnp.array([4.0e18, 5.0e18])
    layer_height = jnp.array([1.0e5, 2.0e5])

    optical_depth = layer_optical_depth_from_log_cia(
        log_cia, number_density_1, number_density_2, layer_height
    )
    expected = (
        10.0**log_cia
        * number_density_1[:, None, None]
        * number_density_2[:, None, None]
        * layer_height[:, None, None]
    )

    assert optical_depth.shape == log_cia.shape
    np.testing.assert_allclose(optical_depth, expected, rtol=2.0e-6)


def test_extinction_common_spectrum_matches_analytic_equation():
    extinction_coefficient = jnp.array([1.0e-7, 3.0e-7])
    layer_height = jnp.array([1.0e5, 2.0e5, 4.0e5])

    optical_depth = layer_optical_depth_from_extinction(
        extinction_coefficient, layer_height
    )
    expected = extinction_coefficient[None, :] * layer_height[:, None]

    assert optical_depth.shape == (3, 2)
    np.testing.assert_allclose(optical_depth, expected)


def test_geometric_optical_depth_does_not_silently_make_opacity_positive():
    optical_depth = layer_optical_depth_from_cross_section(
        jnp.array([-2.0e-20]), 3.0e10, jnp.array([4.0e5])
    )

    np.testing.assert_allclose(optical_depth, jnp.array([[-2.4e-4]]))


def test_log_cia_preserves_float32_dynamic_range_and_identical_partner_rule():
    log_cia = jnp.array([-46.0, -60.0], dtype=jnp.float32)
    number_density = jnp.asarray(1.0e20, dtype=jnp.float32)
    layer_height = jnp.array([1.0e5], dtype=jnp.float32)

    linearized_coefficient = jnp.power(jnp.float32(10.0), log_cia)
    optical_depth = layer_optical_depth_from_log_cia(
        log_cia, number_density, number_density, layer_height
    )

    np.testing.assert_array_equal(linearized_coefficient, jnp.zeros(2))
    np.testing.assert_allclose(
        optical_depth,
        jnp.array([[1.0e-1, 1.0e-15]], dtype=jnp.float32),
        rtol=2.0e-5,
    )


def test_log_cia_zero_density_and_height_return_zero_without_nan_gradient():
    log_cia = jnp.full((2, 2), -46.0)
    number_density_1 = jnp.array([0.0, 1.0e20])
    number_density_2 = jnp.full(2, 1.0e20)
    layer_height = jnp.array([1.0e5, 0.0])

    optical_depth = layer_optical_depth_from_log_cia(
        log_cia, number_density_1, number_density_2, layer_height
    )
    gradient = jax.grad(
        lambda density: jnp.sum(
            layer_optical_depth_from_log_cia(
                log_cia, density, number_density_2, layer_height
            )
        )
    )(number_density_1)

    np.testing.assert_array_equal(optical_depth, jnp.zeros((2, 2)))
    assert np.all(np.isfinite(gradient))


def test_geometric_optical_depth_supports_jit_and_grad():
    cross_section = jnp.array([1.0e-20, 2.0e-20])
    log_cia = jnp.array([-46.0, -45.0])
    extinction_coefficient = jnp.array([1.0e-7, 2.0e-7])
    number_density = jnp.array([1.0e20, 2.0e20])
    layer_height = jnp.array([1.0e5, 2.0e5])

    def summed_optical_depth(scale):
        cross_section_depth = layer_optical_depth_from_cross_section(
            scale * cross_section, number_density, layer_height
        )
        cia_depth = layer_optical_depth_from_log_cia(
            log_cia + jnp.log10(scale),
            number_density,
            number_density,
            layer_height,
        )
        extinction_depth = layer_optical_depth_from_extinction(
            scale * extinction_coefficient, layer_height
        )
        return (
            jnp.sum(cross_section_depth)
            + jnp.sum(cia_depth)
            + jnp.sum(extinction_depth)
        )

    value, gradient = jax.jit(jax.value_and_grad(summed_optical_depth))(1.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_geometric_optical_depth_rejects_invalid_shapes():
    layer_height = jnp.ones(2)

    with pytest.raises(ValueError, match="layer_height must be a one-dimensional"):
        layer_optical_depth_from_extinction(jnp.ones(3), jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="at least one layer"):
        layer_optical_depth_from_extinction(jnp.ones(3), jnp.empty(0))
    with pytest.raises(ValueError, match="at least one dimension"):
        layer_optical_depth_from_cross_section(1.0, 1.0, layer_height)
    with pytest.raises(ValueError, match="leading axis"):
        layer_optical_depth_from_cross_section(
            jnp.ones((3, 4)), jnp.ones(2), layer_height
        )
    with pytest.raises(ValueError, match="absorber_number_density"):
        layer_optical_depth_from_cross_section(
            jnp.ones((2, 4)), jnp.ones((2, 1)), layer_height
        )
    with pytest.raises(ValueError, match="number_density_2"):
        layer_optical_depth_from_log_cia(
            jnp.ones((2, 4)), jnp.ones(2), jnp.ones(3), layer_height
        )
