import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pytest
from jax import grad
from jax import jvp
from exojax.postproc.spin_rotation import rotkernel
from exojax.postproc.spin_rotation import rotkernel_Gimenez
import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.postproc.spin_rotation import convolve_rigid_rotation_ola
import matplotlib.pyplot as plt
        

def _convolve_rigid_rotation_np(resolution, F0, vsini, u1=0.0, u2=0.0):
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
    x = velocity_grid(resolution, vsini)
    kernel = rotkernel(x/vsini, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)
    #F = jnp.convolve(F0,kernel,mode="same")

    #No OLA
    input_length = len(F0)
    filter_length = len(kernel)
    #fft_length = input_length + filter_length - 1
    convolved_signal = np.convolve(F0, kernel, mode="same")
    return convolved_signal

def test_SopRotation(N=1000):
    from jax import config
    from exojax.postproc.specop import SopRotation
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5

    vsini = 40.0
    sos = SopRotation(nus, vsini)
    
    Frot = sos.rigid_rotation(F0, vsini, u1=0.1, u2=0.1)
    Frot_ = _convolve_rigid_rotation_np(resolution, F0, vsini, u1=0.1, u2=0.1)
    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5


def test_convolve_rigid_rotation(N=1000, fig=False):
    from jax import config
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5
    vsini = 40.0
    vr_array = velocity_grid(resolution, vsini)
    
    Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1=0.1, u2=0.1)
    Frot_ = _convolve_rigid_rotation_np(resolution, F0, vsini, u1=0.1, u2=0.1)
    
    if fig:
        _plotfig(Frot, Frot_)

    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5

def test_convolve_rigid_rotation_ola(N=10000, fig=False):
    from jax import config
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[2500 - 50:2500 + 50] = 0.5
    vsini = 4.0
    vr_array = velocity_grid(resolution, vsini)
    input_matrix = F0.reshape((5,int(float(N)/5)))
    
    Frot = convolve_rigid_rotation_ola(input_matrix, vr_array, vsini, u1=0.1, u2=0.1)
    Frot_ = _convolve_rigid_rotation_np(resolution, F0, vsini, u1=0.1, u2=0.1)
    if fig:
        _plotfig(Frot, Frot_)
    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5

def test_SopRotation_ola(N=10000, fig=False):
    from jax import config
    from exojax.postproc.specop import SopRotation
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[2500 - 50:2500 + 50] = 0.5
    vsini = 4.0
    
    sos = SopRotation(nus, vsini, convolution_method = "exojax.signal.ola" )    
    Frot = sos.rigid_rotation(F0, vsini, u1=0.1, u2=0.1)
    Frot_ = _convolve_rigid_rotation_np(resolution, F0, vsini, u1=0.1, u2=0.1)

    if fig:
        _plotfig(Frot, Frot_)
    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5

def _plotfig(Frot, Frot_):
    figx = plt.figure()
    ax = figx.add_subplot(211)
    plt.plot(Frot_, label="numpy.convolve")
    plt.plot(Frot, label="exojax")
    plt.legend()
    ax = figx.add_subplot(212)
    plt.plot(1.0-Frot/Frot_, label="diff")
    plt.legend()
    plt.show()



def test_rotkernel(fig=False):
    N = 201
    x_1 = jnp.linspace(-2.0, 2.0, N)
    u1 = 0.1
    u2 = 0.1
    kernel_1 = rotkernel(x_1, u1, u2)
    one_minus_x2 = jnp.maximum(1.0 - x_1**2, 0.0)
    expected = jnp.where(
        x_1**2 <= 1.0,
        2.0 * (1.0 - u1 - u2) * jnp.sqrt(one_minus_x2)
        + jnp.pi / 2.0 * (u1 + 2.0 * u2) * one_minus_x2
        - 4.0 / 3.0 * u2 * one_minus_x2**1.5,
        0.0,
    )
    assert kernel_1 == pytest.approx(expected)
    N = 101
    x_2 = jnp.linspace(-1.0, 1.0, N)
    kernel_2 = rotkernel(x_2, u1, u2)
    assert jnp.sum(kernel_1) == pytest.approx(149.08960)
    assert jnp.sum(kernel_2) == pytest.approx(149.08960)

    if fig:
        import matplotlib.pyplot as plt
        plt.plot(x_1, kernel_1)
        plt.plot(x_2, kernel_2)
        plt.show()


def test_rotkernel_Gimenez():
    x = jnp.linspace(-2.0, 2.0, 201)
    u1_g = 0.1
    u2_g = 0.1
    kernel = rotkernel_Gimenez(x, u1_g, u2_g)
    one_minus_x2 = jnp.maximum(1.0 - x**2, 0.0)
    expected = jnp.where(
        x**2 <= 1.0,
        2.0 * (1.0 - u1_g - u2_g) * jnp.sqrt(one_minus_x2)
        + jnp.pi / 2.0 * u1_g * one_minus_x2
        + 4.0 / 3.0 * u2_g * one_minus_x2**1.5,
        0.0,
    )
    assert kernel == pytest.approx(expected)


def test_rotkernel_Gimenez_jvp():
    x = 0.2
    u1_g = 0.1
    u2_g = 0.1
    one_minus_x2 = 1.0 - x**2
    expected = (
        -2.0 * (1.0 - u1_g - u2_g) * x / jnp.sqrt(one_minus_x2)
        - jnp.pi * u1_g * x
        - 4.0 * u2_g * x * jnp.sqrt(one_minus_x2)
    )
    primal, tangent = jvp(
        lambda z: rotkernel_Gimenez(z, u1_g, u2_g), (x,), (1.0,)
    )
    assert primal == pytest.approx(rotkernel_Gimenez(x, u1_g, u2_g))
    assert tangent == pytest.approx(expected)
    assert grad(lambda z: rotkernel_Gimenez(z, u1_g, u2_g))(x) == pytest.approx(
        expected
    )
    assert grad(lambda z: rotkernel(z, 0.1, 0.1))(1.0) == pytest.approx(0.0)
    assert grad(grad(lambda z: rotkernel(z, 0.1, 0.1)))(1.1) == pytest.approx(0.0)


if __name__ == "__main__":
    test_rotkernel(fig=True)
    test_convolve_rigid_rotation(1000,fig=True)
    test_convolve_rigid_rotation_ola(10000, fig=True)
    test_SopRotation(1000)
    test_SopRotation_ola(10000, fig=True)
