import pytest
from exojax.postproc.spin_rotation import integrated_rotkernel_trans
import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.grids import velocity_grid
from exojax.utils.grids import delta_velocity_from_resolution
from exojax.postproc.spin_rotation import convolve_rigid_rotation_trans
from exojax.postproc.spin_rotation import convolve_rigid_rotation_ola_trans
from exojax.postproc.spin_rotation import generate_equal_theta_array, apply_weighted_rv_shifts
import matplotlib.pyplot as plt
        

def _convolve_rigid_rotation_trans_np(resolution, F0, vsini):
    """Apply the Rotation response to a spectrum F.

    Args:
        resolution: spectral resolution of wavenumber bin (ESLOG)
        F0: original spectrum (F0)
        vsini: V sini for rotation (km/s)

    Return:
        response-applied spectrum (F)
    """
    x = velocity_grid(resolution, vsini)
    dv = delta_velocity_from_resolution(resolution)
    int_kernel = integrated_rotkernel_trans((x-0.5*dv)/vsini, (x+0.5*dv)/vsini)
    int_kernel = int_kernel / jnp.sum(int_kernel, axis=0)
    #F = jnp.convolve(F0,kernel,mode="same")

    #No OLA
    input_length = len(F0)
    filter_length = len(int_kernel)
    #fft_length = input_length + filter_length - 1
    convolved_signal = np.convolve(F0, int_kernel, mode="same")
    return convolved_signal

def test_SopRotation_trans(N=1000):
    from jax import config
    from exojax.postproc.specop import SopRotation
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5

    vsini = 4.0
    sos = SopRotation(nus, vsini)
    
    Frot = sos.rigid_rotation_trans(F0, vsini)
    Frot_ = _convolve_rigid_rotation_trans_np(resolution, F0, vsini)
    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5


def test_convolve_rigid_rotation_trans(N=1000, fig=False):
    from jax import config
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5
    vsini = 4.0
    vr_array = velocity_grid(resolution, vsini)
    dv = delta_velocity_from_resolution(resolution)

    Frot = convolve_rigid_rotation_trans(F0, vr_array, dv, vsini)
    Frot_ = _convolve_rigid_rotation_trans_np(resolution, F0, vsini)
    
    if fig:
        _plotfig(Frot, Frot_)

    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5

def test_convolve_rigid_rotation_trans_ola(N=10000, fig=False):
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
    dv = delta_velocity_from_resolution(resolution)

    Frot = convolve_rigid_rotation_ola_trans(input_matrix, vr_array, dv, vsini)
    Frot_ = _convolve_rigid_rotation_trans_np(resolution, F0, vsini)
    if fig:
        _plotfig(Frot, Frot_)
    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-5

def test_SopRotation_trans_ola(N=10000, fig=False):
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
    Frot = sos.rigid_rotation_trans(F0, vsini)
    Frot_ = _convolve_rigid_rotation_trans_np(resolution, F0, vsini)

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



def test_integrated_rotkernel_trans(fig=False):
    N = 201
    x_1 = jnp.linspace(-2.0, 2.0, N)
    dx = (jnp.max(x_1) - jnp.min(x_1)) / (N-1)
    kernel_1 = integrated_rotkernel_trans(x_1-dx, x_1+dx)
    N = 101
    x_2 = jnp.linspace(-1.0, 1.0, N)
    dx = (jnp.max(x_2) - jnp.min(x_2)) / (N-1)
    kernel_2 = integrated_rotkernel_trans(x_2-dx, x_2+dx)
    assert jnp.sum(kernel_1) == pytest.approx(1.999999)
    assert jnp.sum(kernel_2) == pytest.approx(1.9998436)

    if fig:
        import matplotlib.pyplot as plt
        plt.plot(x_1, kernel_1)
        plt.plot(x_2, kernel_2)
        plt.show()



def test_generate_equal_theta_array():
    Nt = 100
    theta_array_1 = generate_equal_theta_array(Nt)
    theta_array_2 = generate_equal_theta_array(Nt, hemisphere=False)
    assert jnp.sum(theta_array_1) == pytest.approx(157.07964)
    assert jnp.sum(theta_array_2) == pytest.approx(314.15927)

def test_apply_weighted_rv_shifts(N=1000, fig=False):
    from jax import config
    from exojax.utils.constants import c
    config.update("jax_enable_x64", True)
    nus, wav, resolution = wavenumber_grid(4000.0,
                                               4010.0,
                                               N,
                                               xsmode="premodit")

    F0 = np.ones_like(nus)
    F0[250 - 5:250 + 5] = 0.5
    vsini = 4.0
    dv = delta_velocity_from_resolution(resolution)
    dtheta = jnp.arcsin(dv/vsini)
    Nt = jnp.ceil(jnp.pi / dtheta).astype(jnp.int32)
    theta_array = generate_equal_theta_array(Nt)
    rv_array = jnp.cos(theta_array) * vsini
    
    Frot = apply_weighted_rv_shifts(F0, nus, rv_array, jnp.ones_like(rv_array))
    Frot_ = _convolve_rigid_rotation_trans_np(resolution, F0, vsini)

    # remove invalid edge points
    mask = (nus >= jnp.min(nus / (1.0 + jnp.min(rv_array)/c))) * (nus <= jnp.max(nus / (1.0 + jnp.max(rv_array)/c)))
    Frot = Frot[mask]
    Frot_ = Frot_[mask]

    if fig:
        _plotfig(Frot, Frot_)

    res = np.sqrt(np.sum(np.abs(1.0 - Frot / Frot_)**2))
    assert res < 1.e-1

if __name__ == "__main__":
    test_integrated_rotkernel_trans(fig=True)
    test_convolve_rigid_rotation_trans(1000,fig=True)
    test_convolve_rigid_rotation_trans_ola(10000, fig=True)
    test_SopRotation_trans(1000)
    test_SopRotation_trans_ola(10000, fig=True)
    test_generate_equal_theta_array()
    test_apply_weighted_rv_shifts(N=10000, fig=True)
