import jax.numpy as jnp
from jax import custom_jvp, jit
from jax.lax import scan

from exojax.signal.convolve import convolve_same
from exojax.signal.ola import generate_zeropad, ola_lengths, olaconv
from exojax.postproc.response import sampling


@jit
def convolve_rigid_rotation_ola(folded_F0, vr_array, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F (No OLA and No cuDNN).

    Args:
        folded_F0: original spectrum (F0) folded to (ndiv, div_length) form
        vr_array: fix-sized vr array for kernel, see utils.dvgrid_rigid_rotation
        vsini: V sini for rotation (km/s)
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    kernel = rotkernel(vr_array/vsini, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)

    ndiv, div_length, filter_length = ola_lengths(folded_F0, kernel)
    F0_hat, kernel_hat = generate_zeropad(folded_F0, kernel)
    ola = olaconv(F0_hat, kernel_hat, ndiv, div_length, filter_length)
    
    edge = int((len(kernel) - 1) / 2)
    convolved_signal = ola[edge:-edge]
    
    return convolved_signal


@jit
def convolve_rigid_rotation(F0, vr_array, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F (No OLA and No cuDNN).

    Args:
        F0: original spectrum (F0)
        vr_array: fix-sized vr array for kernel, see utils.dvgrid_rigid_rotation
        vsini: V sini for rotation (km/s)
        RV: radial velocity
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    kernel = rotkernel(vr_array/vsini, u1, u2)
    kernel = kernel / jnp.sum(kernel, axis=0)

    #==== still require cuDNN in Oct.15 2022================
    #convolved_signal = jnp.convolve(F0,kernel,mode="same")
    #=======================================================

    convolved_signal = convolve_same(F0, kernel)

    return convolved_signal


@custom_jvp
def rotkernel(x, u1, u2):
    """rotation kernel w/ the quadratic limb darkening law, the numerator of (56) in Kawahara+2022

    Args:
        x: x variable
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        rotational kernel
    """
    x2 = x * x
    kernel = jnp.where(
        x2 <= 1.0,
        jnp.pi / 2.0 * u1 * (1.0 - x2) - 2.0 / 3.0 * jnp.sqrt(1.0 - x2) *
        (-3.0 + 3.0 * u1 + u2 + 2.0 * u2 * x2), 0.0)
    return kernel


@rotkernel.defjvp
def rotkernel_jvp(primals, tangents):
    x, u1, u2 = primals
    ux, uu1, uu2 = tangents
    x2 = x * x
    dHdx = jnp.where(
        x2 <= 1.0, -jnp.pi * x * u1 + 2.0 / 3.0 * x / jnp.sqrt(1.0 - x2) *
        (-3.0 + 3.0 * u1 + u2 + 2.0 * u2 * x2) +
        8.0 * x * u2 * jnp.sqrt(1.0 - x2), 0.0)
    dHdu1 = jnp.where(x2 <= 1.0,
                      -2.0 * jnp.sqrt(1.0 - x2) + jnp.pi / 2.0 * (1.0 - x2),
                      0.0)
    dHdu2 = jnp.where(x2 <= 1.0,
                      -2.0 * (1.0 + 2.0 * x2) * (jnp.sqrt(1.0 - x2)) / 3.0,
                      0.0)

    primal_out = rotkernel(x, u1, u2)
    tangent_out = dHdx * ux + dHdu1 * uu1 + dHdu2 * uu2
    return primal_out, tangent_out


@jit
def convolve_rigid_rotation_ola_trans(folded_F0, vr_array, dv, vsini):
    """Apply the Rotation response to a transmission spectrum F (OLA).

    Args:
        folded_F0: original spectrum (F0) folded to (ndiv, div_length) form
        vr_array: fix-sized vr array for kernel, see utils.dvgrid_rigid_rotation
        dv: velocity grid width
        vsini: V sini for rotation (km/s)

    Return:
        response-applied spectrum (F)
    """
    int_kernel = integrated_rotkernel_trans((vr_array-0.5*dv)/vsini, (vr_array+0.5*dv)/vsini)
    int_kernel = int_kernel / jnp.sum(int_kernel, axis=0)

    ndiv, div_length, filter_length = ola_lengths(folded_F0, int_kernel)
    F0_hat, int_kernel_hat = generate_zeropad(folded_F0, int_kernel)
    ola = olaconv(F0_hat, int_kernel_hat, ndiv, div_length, filter_length)

    edge = int((len(int_kernel) - 1) / 2)
    convolved_signal = ola[edge:-edge]

    return convolved_signal


@jit
def convolve_rigid_rotation_trans(F0, vr_array, dv, vsini):
    """Apply the Rotation response to a transmission spectrum F (No OLA and No cuDNN).

    Args:
        F0: original spectrum (F0)
        vr_array: fix-sized vr array for kernel, see utils.dvgrid_rigid_rotation
        dv: velocity grid width
        vsini: V sini for rotation (km/s)

    Return:
        response-applied spectrum (F)
    """
    int_kernel = integrated_rotkernel_trans((vr_array-0.5*dv)/vsini, (vr_array+0.5*dv)/vsini)
    int_kernel = int_kernel / jnp.sum(int_kernel, axis=0)

    #==== still require cuDNN in Oct.15 2022================
    #convolved_signal = jnp.convolve(F0,kernel,mode="same")
    #=======================================================

    convolved_signal = convolve_same(F0, int_kernel)

    return convolved_signal


@custom_jvp
def integrated_rotkernel_trans(x1, x2):
    """integrated rotation kernel

    Args:
        x1: Velocity at bin_edge / vsini
        x2: Velocity at bin_edge / vsini

    Return:
        integrated rotational kernel
    """
    # clip outside the [-vsini, vsini] range
    x1_c = jnp.clip(x1, -1., 1.)
    x2_c = jnp.clip(x2, -1., 1.)
    int_kernel = jnp.abs(jnp.arcsin(x2_c) - jnp.arcsin(x1_c)) / jnp.pi
    return int_kernel


@integrated_rotkernel_trans.defjvp
def integrated_rotkernel_trans_jvp(primals, tangents):
    x1, x2 = primals
    ux1, ux2 = tangents
    x1_2 = x1 * x1
    x2_2 = x2 * x2
    dHdx1 = jnp.where(x1_2 < 1.0, 1./jnp.sqrt(1. - x1_2)/jnp.pi, 0.0)
    dHdx2 = jnp.where(x2_2 < 1.0, 1./jnp.sqrt(1. - x2_2)/jnp.pi, 0.0)

    primal_out = integrated_rotkernel_trans(x1, x2)
    tangent_out = dHdx1 * ux1 + dHdx2 * ux2
    return primal_out, tangent_out


def generate_equal_theta_array(Nt, hemisphere=True):
    """generate equal theta array

    Args:
        Nt (int): number of theta array
        hemisphere (bool): If true, provide grids from 0 to pi

    Returns:
        array: theta array
     """
    if hemisphere:
        theta_array = (jnp.arange(Nt) + 0.5) / Nt * jnp.pi
    else:
        theta_array = (jnp.arange(Nt) + 0.5) / Nt * 2. * jnp.pi
    return theta_array

@jit
def apply_weighted_rv_shifts(F0, nu_grid, rv_array, weight_array):
    """Apply the user-specified radial velocity response to a spectrum F

    Args:
        F0: original spectrum (F0)
        nu_grid: wavenumber grid in cm-1
        rv_array: radial velocity array
        weight_array: weight corresponding to the rv_array

    Return:
        response-applied spectrum (F)
    """
    def f(acc, ipt):
        rv, weight = ipt
        sp_sft = sampling(nu_grid, nu_grid, F0, rv)
        acc = acc + sp_sft * weight
        return acc, None

    acc0 = jnp.zeros_like(F0)
    ipt = [rv_array, weight_array]
    acc, _ = scan(f, acc0, ipt)

    return acc/jnp.sum(weight_array)
