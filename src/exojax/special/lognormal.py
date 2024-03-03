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
    return r**3*pdf(r,rg,sigmag)/moment(rg, sigmag, 3)

def cubeweighted_mean(rg, sigmag):
    return moment(rg, sigmag, 4)/moment(rg, sigmag, 3)

def cubeweighted_std(rg, sigmag):
    mu = cubeweighted_mean(rg, sigmag)
    var = moment(rg, sigmag, 5)/moment(rg, sigmag, 3)
    return jnp.sqrt(var - mu**2)

if __name__ == "__main__":
    rg = 1.0e-4
    sigmag = 2.0
    e = jnp.sqrt((moment(rg, sigmag, 5) - moment(rg, sigmag, 4) ** 2))/moment(rg, sigmag, 3)
    rmean = moment(rg, sigmag, 4)/moment(rg, sigmag, 3)
    print(rg, rmean, e)
    from jax.scipy.integrate import trapezoid
    
    import matplotlib.pyplot as plt
    arr = jnp.logspace(-8,0,100)
    
    i = trapezoid(pdf(arr,rg,sigmag), arr)
    print(i)
    
    i = trapezoid(cubeweighted_pdf(arr,rg,sigmag), arr)
    print(i)

    
    plt.plot(arr, pdf(arr,rg,sigmag))
    plt.plot(arr, cubeweighted_pdf(arr, rg, sigmag))
    plt.axvline(rg, color="C0")
    plt.axvline(rmean, color="C1")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.show()