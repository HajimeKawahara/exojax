import pytest
import jax.numpy as jnp
from jax import jit
@jit
def _ipgauss_naive(nus, F0, beta):
    """Apply the Gaussian IP response to a spectrum F using jax.lax.scan.

    Args:
        nus: input wavenumber, evenly log-spaced
        F0: original spectrum (F0)
        beta: STD of a Gaussian broadening (IP+microturbulence)

    Return:
        response-applied spectrum (F)
    """
    dvmat = jnp.array(c * jnp.log(nus[None, :] / nus[:, None]))
    kernel = jnp.exp(-(dvmat)**2 / (2.0 * beta**2))
    kernel = kernel / jnp.sum(kernel, axis=0)  # axis=N
    F = kernel.T @ F0
    return F

