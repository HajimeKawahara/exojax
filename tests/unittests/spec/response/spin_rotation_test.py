import pytest
from exojax.spec.spin_rotation import rotkernel
import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import c
from exojax.spec.setrt import gen_wavenumber_grid


def _naive_rigidrot(nus, F0, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F (numpy)

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        vsini: V sini for rotation
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    dvmat = np.array(c * (np.log(nus[None, :]) - np.log(nus[:, None])))
    x = dvmat / vsini
    kernel = rotkernel(x, u1, u2)
    kernel = kernel / np.sum(kernel, axis=0)
    F = kernel.T @ F0
    return F


from exojax.signal.ola import olaconv, ola_lengths, generate_zeropad
from exojax.utils.delta_velocity import dvgrid_rigid_rotation
from jax.numpy import index_exp


def convolve_rigid_rotation(resolution, F0, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F.

    Args:
        resolution: spectral resolution of wavenumber bin (ESLOG)
        F0: original spectrum (F0)
        vsini: V sini for rotation (km/s)
        RV: radial velocity
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    x = dvgrid_rigid_rotation(resolution, vsini)
    kernel = rotkernel(x, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)
    #F = jnp.convolve(F0,kernel,mode="same")

    #No OLA
    input_length = len(F0)
    filter_length = len(kernel)
    fft_length = input_length# + filter_length - 1
    F0_zeropad = jnp.zeros(fft_length)
    F0_zeropad = F0_zeropad.at[index_exp[0:input_length]].add(F0)
    filter_zeropad = jnp.zeros(fft_length)
    filter_zeropad = filter_zeropad.at[index_exp[0:filter_length]].add(kernel)
    convolved_signal = jnp.fft.irfft(
        jnp.fft.rfft(F0_zeropad) * jnp.fft.rfft(filter_zeropad))
    n = int((filter_length - 1) / 2)
    #convolved_signal = convolved_signal[n:-n]

    #F_zeropad = jnp.zeros((ndiv*fft_length,))
    #F_zeropad = F_zeropad.at[index_exp[0:input_length]].add(F0)
    #F_zeropad.reshape((ndiv,fft_length))
    #ola = olaconv(F0_hat, kernel_hat, ndiv, div_length, filter_length)

    return convolved_signal


def test_convolve_rigid_rotation(fig=False):
    from jax.config import config
    #config.update("jax_enable_x64", True)
    from exojax.utils.constants import c
    nus, wav, resolution = gen_wavenumber_grid(4000.0,
                                               4010.0,
                                               1000,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5
    vsini = 40.0
    Frot_ = _naive_rigidrot(nus, F0, vsini, u1=0.1, u2=0.1)
    Frot = convolve_rigid_rotation(resolution, F0, vsini, u1=0.1, u2=0.1)
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(Frot_)
        plt.plot(Frot)
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
