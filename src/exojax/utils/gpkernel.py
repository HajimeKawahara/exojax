"""Kernels used in Gaussian process."""

import jax.numpy as jnp


def gpkernel_RBF(t, tau, a, err):
    """RBF kernel with diagnoal error.

    Args:
       t: variable vector (N)
       tau: scale parameter (scalar)
       a: amplitude (scalar)
       err: diagnonal error vector (N)

    Returns:
       kernel
    """

    Dt = t - jnp.array([t]).T
    K = a*jnp.exp(-(Dt)**2/2/(tau**2))+jnp.diag(err**2)
    return K
