"""response.

* input nus/wav should be spaced evenly on a log scale (ESLOG).
* response is a response operation for the wavenumber grid spaced evenly on a log scale.
* rigidrot2 and ipgauss2 are faster than default when N >~ 10000, where N is the dimension of the wavenumber grid.
* response uses jax.numpy.convolve, therefore, convolve in cuDNN.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np
from jax import custom_jvp
from exojax.utils.constants import c


@custom_jvp
def rotkernel(x, u1, u2):
    """rotation kernel w/ the quadratic Limb dwarkening law, numerator of (54) in Kawahara+2022

    Args:
        x: x variable
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        rotational kernel
    """    
    x2 = x*x
    kernel = jnp.where(x2 <= 1.0, jnp.pi/2.0*u1*(1.0 - x2) -
                       2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2+2.0*u2*x2), 0.0)
    return kernel


@rotkernel.defjvp
def rotkernel_jvp(primals, tangents):
    x, u1, u2 = primals
    ux, uu1, uu2 = tangents
    x2 = x*x
    dHdx = jnp.where(x2 <= 1.0, - jnp.pi*x*u1 +
                    2.0/3.0*x/jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2+2.0*u2*x2)+8.0*x*u2*jnp.sqrt(1.0-x2), 0.0)
    dHdu1= jnp.where(x2 <= 1.0, -2.0*jnp.sqrt(1.0-x2)+jnp.pi/2.0*(1.0-x2), 0.0)
    dHdu2= jnp.where(x2 <= 1.0, -2.0*(1.0+2.0*x2)*(jnp.sqrt(1.0-x2))/3.0, 0.0)
    
    primal_out = rotkernel(x, u1, u2)
    tangent_out = dHdx * ux + dHdu1 * uu1 + dHdu2 * uu2
    return primal_out, tangent_out


def rigidrot(nus, F0, vsini, u1=0.0, u2=0.0):
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
    dvmat = jnp.array(c*(jnp.log(nus[None, :])-jnp.log(nus[:, None])))
    x = dvmat/vsini
    kernel = rotkernel(x, u1, u2)
    kernel = kernel/jnp.sum(kernel, axis=0)
    F = kernel.T@F0
    return F


@jit
def ipgauss(nus, F0, beta):
    """Apply the Gaussian IP response to a spectrum F using jax.lax.scan.

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)
    """
    dvmat = jnp.array(c*jnp.log(nus[None, :]/nus[:, None]))
    kernel = jnp.exp(-(dvmat)**2/(2.0*beta**2))
    kernel = kernel/jnp.sum(kernel, axis=0)  # axis=N
    F = kernel.T@F0
    return F


def ipgauss_sampling(nusd, nus, F0, beta, RV):
    """Apply the Gaussian IP response + sampling to a spectrum F.

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)

    Return:
        response-applied spectrum (F)
    """
    if(np.min(nusd) < np.min(nus) or np.max(nusd) > np.max(nus)):
        print('WARNING: The wavenumber range of the observational grid [', np.min(nusd), '-', np.max(nusd), ' cm^(-1)] is not fully covered by that of the model grid [', np.min(nus), '-', np.max(nus), ' cm^(-1)]. This can result in the incorrect response-applied spectrum. Check the wavenumber grids for the model and observation.', sep='')

    @jit
    def ipgauss_sampling_jax(nusd, nus, F0, beta, RV):
        dvmat = jnp.array(c*jnp.log(nusd[None, :]/nus[:, None]))
        kernel = jnp.exp(-(dvmat+RV)**2/(2.0*beta**2))
        kernel = kernel/jnp.sum(kernel, axis=0)  # axis=N
        F = kernel.T@F0
        return F

    F = ipgauss_sampling_jax(nusd, nus, F0, beta, RV)
    return F


@jit
def rigidrot2(nus, F0, varr_kernel, vsini, u1=0.0, u2=0.0):
    """Apply the Rotation response to a spectrum F using jax.lax.scan.

    Args:
        nus: wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        vsini: V sini for rotation
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        response-applied spectrum (F)
    """
    x = varr_kernel/vsini
    x2 = x*x
    kernel = rotkernel(x, u1, u2)
    kernel = kernel/jnp.sum(kernel, axis=0)
    F = jnp.convolve(F0, kernel, mode='same')

    return F


@jit
def ipgauss2(nus, F0, varr_kernel, beta):
    """Apply the Gaussian IP response to a spectrum F.

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)
    """
    x = varr_kernel/beta
    kernel = jnp.exp(-x*x/2.0)
    kernel = kernel/jnp.sum(kernel, axis=0)
    F = jnp.convolve(F0, kernel, mode='same')

    return F


@jit
def sampling(nusd, nus, F, RV):
    """Sampling w/ RV.

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber
        F: input spectrum
        RV: radial velocity (km/s)

    Returns:
       sampled spectrum
    """
    return jnp.interp(nusd*(1.0+RV/c), nus, F)
