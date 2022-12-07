"""response.

* input nus/wav should be spaced evenly on a log scale (ESLOG).
* response is a response operation for the wavenumber grid spaced evenly on a log scale.
* ipgauss2 are faster than default when N >~ 10000, where N is the dimension of the wavenumber grid.
"""
import jax.numpy as jnp
from jax import jit
from exojax.utils.constants import c
from jax.lax import scan
from jax import jit

from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from exojax.utils.instfunc import resolution_eslog
import warnings


def rigidrot(nus, F0, vsini, u1, u2, vsinimax=100.0):
    """(deprecated) rigid rotation. This will be removed.
    """
    warnings.warn(
        "rigidrot is deprecated and do not work for VJP. Use convolve_rigid_rotation instead.",
        UserWarning)
    resolution = resolution_eslog(nus)
    vr_array = velocity_grid(resolution, vsinimax)
    return convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)


@jit
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
    def convolve_ipgauss_scan(carry, nusd_each):
        dvgrid = c * (jnp.log1p(1.0 - nus / nusd_each))
        kernel = jnp.exp(-(dvgrid + RV)**2 / (2.0 * beta**2))
        kernel = kernel / jnp.sum(kernel)
        return carry, kernel @ F0

    _, F_convolved = scan(convolve_ipgauss_scan, 0, nusd)
    return F_convolved


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
    x = varr_kernel / beta
    kernel = jnp.exp(-x * x / 2.0)
    kernel = kernel / jnp.sum(kernel, axis=0)
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
    return jnp.interp(nusd * (1.0 + RV / c), nus, F)
