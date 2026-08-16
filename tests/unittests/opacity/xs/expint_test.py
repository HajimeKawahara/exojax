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
