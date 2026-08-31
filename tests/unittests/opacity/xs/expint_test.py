import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expn
from exojax.rt import rtransfer as rt
from exojax.special.expn import E1


def test_comparison_expint():
    x = np.logspace(-4, 1.9, 1000).astype(np.float32)
    dif = np.abs(2.0 * expn(3, x) - rt.trans2E3(x))
    assert np.max(dif) < 2.0e-7


def test_fbased2st_thin_float32_layers():
    dtau = jnp.full((100, 1), 1.0e-8, dtype=jnp.float32)
    source = jnp.ones_like(dtau)
    half_source_surface = jnp.full(1, 0.5, dtype=jnp.float32)

    transmission = 2.0 * expn(3, 1.0e-8)
    expected = -np.expm1(100.0 * np.log(transmission))
    expected_with_surface = expected + 0.5 * transmission**100

    actual = rt.rtrun_emis_pureabs_fbased2st(dtau, source)
    np.testing.assert_allclose(
        actual, expected, rtol=2.0e-6, atol=0.0
    )
    np.testing.assert_array_equal(
        rt.rtrun_emis_pureabs_fbased2st_surface(dtau, source, 0.0), actual
    )
    np.testing.assert_allclose(
        rt.rtrun_emis_pureabs_fbased2st_surface(
            dtau, source, half_source_surface
        ),
        expected_with_surface,
        rtol=1.0e-7,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        rt.rtrun_emis_pureabs_fbased2st_surface(
            dtau, source, jnp.ones(1, dtype=jnp.float32)
        ),
        jnp.ones(1, dtype=jnp.float32),
    )


def test_fbased2st_thin_float32_layers_attenuate_deep_emission():
    thin_dtau = 1.0e-8
    bottom_dtau = 10.0
    dtau = jnp.full((101, 1), thin_dtau, dtype=jnp.float32)
    dtau = dtau.at[-1, 0].set(bottom_dtau)
    source = jnp.zeros_like(dtau).at[-1, 0].set(1.0)

    transmission_thin = 2.0 * expn(3, thin_dtau)
    absorption_bottom = 1.0 - 2.0 * expn(3, bottom_dtau)
    expected = transmission_thin**100 * absorption_bottom

    actual = rt.rtrun_emis_pureabs_fbased2st(dtau, source)
    np.testing.assert_allclose(
        actual, expected, rtol=1.0e-7, atol=0.0
    )
    np.testing.assert_array_equal(
        rt.rtrun_emis_pureabs_fbased2st_surface(dtau, source, 0.0), actual
    )


def test_fbased2st_thick_float32_layer_with_high_source_contrast():
    source = jnp.ones((1, 1), dtype=jnp.float32)
    source_surface = jnp.full(1, 1.0e12, dtype=jnp.float32)

    for depth in (10.0, 15.0, 20.0):
        transmission = 2.0 * expn(3, depth)
        expected = transmission * 1.0e12 + (1.0 - transmission)
        actual = rt.rtrun_emis_pureabs_fbased2st_surface(
            jnp.full((1, 1), depth, dtype=jnp.float32),
            source,
            source_surface,
        )
        np.testing.assert_allclose(actual, expected, rtol=3.0e-5)


def test_fbased2st_complementary_form_boundary_value_and_gradient():
    source = jnp.full((1, 1), 3.0, dtype=jnp.float32)
    source_surface = jnp.full(1, 7.0, dtype=jnp.float32)

    def flux(depth):
        return rt.rtrun_emis_pureabs_fbased2st_surface(
            depth.reshape(1, 1), source, source_surface
        )[0]

    depth = jnp.float32(0.4190354)
    transmission = 2.0 * expn(3, float(depth))
    expected = transmission * 7.0 + (1.0 - transmission) * 3.0
    expected_gradient = -2.0 * expn(2, float(depth)) * (7.0 - 3.0)

    np.testing.assert_allclose(flux(depth), expected, rtol=1.0e-6)
    np.testing.assert_allclose(
        jax.grad(flux)(depth), expected_gradient, rtol=1.0e-6
    )


def test_trans2E3_zero_limit_and_reverse_gradient():
    x = jnp.float32(0.0)
    np.testing.assert_allclose(rt.trans2E3(x), 1.0)
    np.testing.assert_allclose(jax.grad(rt.trans2E3)(x), -2.0)


def test_trans2E3_reverse_gradient_near_zero():
    gradient = jax.grad(rt.trans2E3)(jnp.float32(1.0e-20))
    np.testing.assert_allclose(gradient, -2.0)


def test_large_float32_optical_depth():
    x = jnp.float32(1.0e10)
    np.testing.assert_allclose(E1(x), 0.0)
    np.testing.assert_allclose(jax.grad(E1)(x), 0.0)
    np.testing.assert_allclose(rt.trans2E3(x), 0.0)

    dtau = jnp.array([[0.1], [1.0e10]], dtype=jnp.float32)
    source = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    flux = rt.rtrun_emis_pureabs_fbased2st(dtau, source)
    np.testing.assert_allclose(flux, [1.8325829], rtol=1.0e-6)

    def objective(d, s):
        return jnp.sum(rt.rtrun_emis_pureabs_fbased2st(d, s))

    gradients = jax.grad(objective, argnums=(0, 1))(dtau, source)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)

    x_overflow = jnp.float32(1.0e20)
    np.testing.assert_allclose(rt.trans2E3(x_overflow), 0.0)
    np.testing.assert_allclose(jax.grad(rt.trans2E3)(x_overflow), 0.0)


def test_fbased2st_zero_optical_depth_reverse_gradient():
    source = jnp.ones((1, 1), dtype=jnp.float32)

    def flux(x):
        return rt.rtrun_emis_pureabs_fbased2st(x.reshape(1, 1), source)[0]

    gradient = jax.grad(flux)(jnp.float32(0.0))
    np.testing.assert_allclose(gradient, 2.0)

    source_surface = jnp.full(1, 3.0, dtype=jnp.float32)

    def flux_with_surface(x):
        return rt.rtrun_emis_pureabs_fbased2st_surface(
            x.reshape(1, 1), source, source_surface
        )[0]

    gradient_with_surface = jax.grad(flux_with_surface)(jnp.float32(0.0))
    np.testing.assert_allclose(gradient_with_surface, -4.0)


if __name__ == "__main__":
    test_comparison_expint()
