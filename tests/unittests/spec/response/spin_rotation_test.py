import pytest
from exojax.spec.spin_rotation import rotkernel
import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import c
from exojax.spec.setrt import gen_wavenumber_grid


def _naive_rigidrot(nus, F0, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan.

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    dvmat = jnp.array(c * (jnp.log(nus[None, :]) - jnp.log(nus[:, None])))
    x = dvmat / vsini
    kernel = rotkernel(x, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)
    F = kernel.T @ F0
    return F




def convolve_rigid_rotation(nus, F0, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F.

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    dvmat = jnp.array(c * (jnp.log(nus[None, :]) - jnp.log(nus[:, None])))
    x = c * jnp.log(nus) / vsini
    kernel = rotkernel(x, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)
    F = kernel.T @ F0
    return F




def test_convolve_rigid_rotation(fig=False):
    from jax.config import config
    #config.update("jax_enable_x64", True)
    from exojax.utils.constants import c
    nus, wav, resolution = gen_wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")
    print(c * np.log(nus[1] / nus[0]))
    print(c * jnp.log(nus[1] / nus[0]))
    print(c * (np.log(nus[1]) - np.log(nus[0])))
    print(c * (jnp.log(nus[1]) - jnp.log(nus[0])))
    print(c * jnp.log(1.0 / resolution + 1.0))
    #print(c/resolution)
    import sys
    sys.exit()

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5
    vsini = 40.0
    Frot = _naive_rigidrot(nus, F0, vsini, u1=0.1, u2=0.1)
    print(c / resolution)

    if fig:
        import matplotlib.pyplot as plt
        plt.plot(nus, Frot)
        plt.plot(nus, F0)

        plt.show()

    #convolve_rigid_rotation(nus, F0, vsini, u1=0.1, u2=0.1)


def test_rotkernel(fig=False):
    N = 201
    x_1 = jnp.linspace(-2.0, 2.0, N)
    u1 = 0.1
    u2 = 0.1
    kernel_1 = rotkernel(x_1, u1, u2)
    N = 101
    x_2 = jnp.linspace(-1.0, 1.0, N)
    kernel_2 = rotkernel(x_2, u1, u2)
    assert jnp.sum(kernel_1) == pytest.approx(143.85559)
    assert jnp.sum(kernel_2) == pytest.approx(143.85559)

    if fig:
        import matplotlib.pyplot as plt
        plt.plot(x_1, kernel_1)
        plt.plot(x_2, kernel_2)
        plt.show()


if __name__ == "__main__":
    #test_rotkernel(fig=True)
    test_convolve_rigid_rotation(fig=True)