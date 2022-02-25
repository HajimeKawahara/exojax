"""Limb darkening functions."""

import jax.numpy as jnp


def ld_kipping(q1, q2):
    """Uninformative prior conversion of the limb darkening by Kipping
    (arxiv:1308.0009)

    Args:
       q1: U(0,1)
       q2: U(0,1)

    Returns:
       u1: quadratic LD coefficient u1
       u2: quadratic LD coefficient u2
    """
    sqrtq1 = jnp.sqrt(q1)
    return 2.0*sqrtq1*q2, sqrtq1*(1.0-2.0*q2)
