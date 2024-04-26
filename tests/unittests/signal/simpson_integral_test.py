import pytest
import numpy as np
import jax.numpy as jnp
from exojax.signal.integrate import simpson
from jax import config

config.update('jax_enable_x64', True)


def test_compare_simpson_with_manual_computation():
    """ test simpson integral
    
    Notes:
        Settings
        Nlayer = 3, Nnus = 1
        f = 0.01, (0.3), 1.0, (1.3), 2.0, (2.7), 3.0
        h = 0.7, 0.8, 0.9
    """
    f_lower = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).T  #(Nlayer, Nnus)
    f_top = jnp.array([0.01, 0.01])
    f = jnp.array([[0.3, 1.3, 2.7], [0.3, 1.3, 2.7]]).T
    h = jnp.array([0.7, 0.8, 0.9])
    simpson0 = h[0] * (f_top + 4.0 * f[0, :] + f_lower[0, :]) / 6.0
    simpson1 = h[1] * (f_lower[0, :] + 4.0 * f[1, :] + f_lower[1, :]) / 6.0
    simpson2 = h[2] * (f_lower[1, :] + 4.0 * f[2, :] + f_lower[2, :]) / 6.0
    ref_integral = simpson0 + simpson1 + simpson2
    integral = simpson(f, f_lower, f_top, h)

    #comparison with jnp.trapz
    #fx =jnp.array([[0.01,1.0,2.0,3.0]]).T
    #x = jnp.array([0.0,h[0],jnp.sum(h[0:2]),jnp.sum(h[0:3])])
    #print("TRAPZ",jnp.trapz(fx,x=x,axis=0))
    #print("SIMPSON",integral)

    assert np.all(integral == pytest.approx(ref_integral))


if __name__ == "__main__":
    test_compare_simpson_with_manual_computation()
