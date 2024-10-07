"""Response

* input nus/wav should be spaced evenly on a log scale (ESLOG).
* response is a response operation for the wavenumber grid spaced evenly on a log scale.

"""

from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from exojax.utils.constants import c
from exojax.signal.convolve import convolve_same
from exojax.signal.ola import olaconv, ola_lengths, generate_zeropad


@jit
def ipgauss_ola_sampling(nusd, nus, folded_spectrum, beta, RV, varr_kernel):
    """Apply the Gaussian IP response using OLA + sampling to a spectrum F.


    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        folded_spectrum: original spectrum (F0) folded to (ndiv, div_length) form
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)
        varr_kernel: velocity array for the rotational kernel

    Return:
        response-applied spectrum (F)
    """
    Fgauss = ipgauss_ola(folded_spectrum, varr_kernel, beta)
    return sampling(nusd, nus, Fgauss, RV)


@jit
def ipgauss_ola(folded_spectrum, varr_kernel, beta):
    """Apply the Gaussian IP response to a spectrum F using OLA.

    Args:
        folded_spectrum: original spectrum (F0) folded to (ndiv, div_length) form
        varr_kernel: velocity array for the rotational kernel
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)
    """
    x = varr_kernel / beta
    kernel = jnp.exp(-x * x / 2.0)
    kernel = kernel / jnp.sum(kernel, axis=0)

    ndiv, div_length, filter_length = ola_lengths(folded_spectrum, kernel)
    F0_hat, kernel_hat = generate_zeropad(folded_spectrum, kernel)
    ola = olaconv(F0_hat, kernel_hat, ndiv, div_length, filter_length)

    edge = int((len(kernel) - 1) / 2)
    F = ola[edge:-edge]

    return F


@jit
def ipgauss_sampling(nusd, nus, spectrum, beta, RV, varr_kernel):
    """Apply the Gaussian IP response + sampling to a spectrum F.


    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        spectrum: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)
        RV: radial velocity (km/s)
        varr_kernel: velocity array for the rotational kernel

    Return:
        response-applied spectrum (F)
    """
    Fgauss = ipgauss(spectrum, varr_kernel, beta)
    return sampling(nusd, nus, Fgauss, RV)


@jit
def ipgauss(spectrum, varr_kernel, beta):
    """Apply the Gaussian IP response to a spectrum F.

    Args:
        spectrum: original spectrum (F0)
        varr_kernel: velocity array for the rotational kernel
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)
    """
    x = varr_kernel / beta
    kernel = jnp.exp(-x * x / 2.0)
    kernel = kernel / jnp.sum(kernel, axis=0)
    # F = jnp.convolve(F0, kernel, mode='same')
    F = convolve_same(spectrum, kernel)

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
    return jnp.interp(nusd * (1.0 + RV / c), nus, F)


@jit
def ipgauss_variable_sampling(nusd, nus, spectrum, beta_variable, RV):
    """Apply the variable Gaussian IP response + sampling to a spectrum F.

    Notes:
        STD is a function of nusd

    Args:
        nusd: sampling wavenumber
        nus: input wavenumber, evenly log-spaced
        spectrum: original spectrum (F0)
        beta_variable (1D array): STD of a Gaussian broadening, shape=(len(nusd),)
        RV: radial velocity (km/s)
    Return:
        response-applied spectrum (F)
    """

    def convolve_ipgauss_scan(carry, arr):
        nusd_each = arr[0]
        beta_each = arr[1]
        dvgrid = c * (jnp.log1p(1.0 - nus / nusd_each))
        kernel = jnp.exp(-((dvgrid + RV) ** 2) / (2.0 * beta_each**2))
        kernel = kernel / jnp.sum(kernel)
        return carry, kernel @ spectrum

    mat = jnp.vstack([nusd, beta_variable]).T
    _, F_convolved = scan(convolve_ipgauss_scan, 0, mat)
    return F_convolved
