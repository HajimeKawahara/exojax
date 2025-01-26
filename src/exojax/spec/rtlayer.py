"""layer by layer radiative transfer module
"""
import jax.numpy as jnp
from jax.lax import scan

# fluxsum vetrorized
def fluxsum_vector(tauup, taulow, mus, weights):
    """flux sum for vectorized calculation

    Args:
        tauup (array): optical depth at the top of the layer [N_wavenumber]
        taulow (array): optical depth at the bottom of the layer [N_wavenumber]
        mus (array): array of cosine of zenith angles [N_stream]
        weights (array): array of weights for each zenith angle [N_stream]

    Returns:
        array: flux sum for each wavenumber [N_wavenumber]
    """

    return jnp.sum(
        weights[:, jnp.newaxis]
        * mus[:, jnp.newaxis]
        * (
            jnp.exp(-tauup / mus[:, jnp.newaxis])
            - jnp.exp(-taulow / mus[:, jnp.newaxis])
        ),
        axis=0,
    )


# fluxsum scan
# @jit
def fluxsum_scan(tauup, taulow, mus, weights):
    """flux sum using jax.lax.scan (but unrolling)

    Args:
        tauup (array): optical depth at the top of the layer [N_wavenumber]
        taulow (array): optical depth at the bottom of the layer [N_wavenumber]
        mus (array): array of cosine of zenith angles [N_stream]
        weights (array): array of weights for each zenith angle [N_stream]

    Returns:
        array: flux sum for each wavenumber [N_wavenumber]
    """

    def f(carry_fmu, muw):
        mu, w = muw
        carry_fmu = carry_fmu + mu * w * (jnp.exp(-tauup / mu) - jnp.exp(-taulow / mu))
        return carry_fmu, None

    # scan part
    muws = [mus, weights]
    flux, _ = scan(f, jnp.zeros_like(tauup), muws)
    return flux

