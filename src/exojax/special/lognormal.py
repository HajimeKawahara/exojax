"""lognormal distribution relationals
"""

import jax.numpy as jnp


def pdf(r, rg, sigmag):
    """probability density function (PDF) of lognormal distribution

    Args:
        r (float, array): variable
        rg (float)): rg parameter
        sigmag (float): sigmag parameter must be > 1.

    Returns:
        float, array: lognormal PDF
    """
    lnsigmag = jnp.log(sigmag)
    fac = jnp.sqrt(2.0 * jnp.pi) * lnsigmag * r
    return 1.0 / fac * jnp.exp(-((jnp.log(r) - jnp.log(rg)) ** 2) / (2.0 * lnsigmag**2))


def moment(rg, sigmag, k):
    """k-th order moment of the lognormal distribution <r^k>

    Args:
        rg (float)): rg parameter
        sigmag (float): sigmag parameter must be > 1.
        k (int): the order of the moment

    Returns:
        float: k-th order moment
    """
    power = 0.5 * k**2 * jnp.log(sigmag) ** 2
    return rg**k * jnp.exp(power)


def cubeweighted_pdf(r, rg, sigmag):
    """cube weighted lognormal distribution

    Args:
        r (float, array): variable
        rg (float)): rg parameter
        sigmag (float): sigmag parameter must be > 1.

    Note:
        The cube-weighted lognormal distribution is proporitnal to x**3 p(x), where p(x) is the lognormal distribution.
        The normalization is given by 1/<x**3>, where <x**N> is the N-th moment of the lognormal distribution.

    Returns:
        float, array: cube-weighted lognormal PDF
    """
    return r**3 * pdf(r, rg, sigmag) / moment(rg, sigmag, 3)


def cubeweighted_mean(rg, sigmag):
    """mean of the cube weighted lognormal distribution

    Args:
        rg (float)): rg parameter
        sigmag (float): sigmag parameter must be > 1.

    Returns:
        float: mean of the cube weighted lognormal distribution
    """
    return moment(rg, sigmag, 4) / moment(rg, sigmag, 3)


def cubeweighted_std(rg, sigmag):
    """variance of the cube weighted lognormal distribution

    Args:
        rg (float)): rg parameter
        sigmag (float): sigmag parameter must be > 1.

    Returns:
        float: variance of the cube weighted lognormal distribution
    """
    mu = cubeweighted_mean(rg, sigmag)
    var = moment(rg, sigmag, 5) / moment(rg, sigmag, 3)
    return jnp.sqrt(var - mu**2)
