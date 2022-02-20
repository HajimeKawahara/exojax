"""Eccentric anomaly from Mean anomaly.

  * JAX autograd/jit compatible version of Markley (Markley 1995, CeMDA, 63, 101) E solver getE(). The original code is taken from PyAstronomy (MIT license).https://github.com/sczesla/PyAstronomy.

"""

import jax.numpy as jnp
from jax import grad
from jax import jit
import numpy as np


@jit
def _alpha(e, M):
    """Solve Eq.

    20
    """
    pi = jnp.pi
    pi2 = pi**2
    return (3. * pi2 + 1.6 * pi * (pi - jnp.abs(M)) / (1. + e)) / (pi2 - 6.)


@jit
def _d(alpha, e):
    """Solve Eq.

    5
    """
    return 3. * (1. - e) + alpha * e


@jit
def _r(alpha, d, M, e):
    """Solve Eq.

    10
    """
    return 3. * alpha * d * (d - 1. + e) * M + M**3


@jit
def _q(alpha, d, e, M):
    """Solve Eq.

    9
    """
    return 2. * alpha * d * (1. - e) - M**2


@jit
def _w(r, q):
    """Solve Eq.

    14
    """
    return (jnp.abs(r) + jnp.sqrt(q**3 + r**2))**(2. / 3.)


@jit
def _E1(d, r, w, q, M):
    """Solve Eq.

    15
    """
    return (2. * r * w / (w**2 + w * q + q**2) + M) / d


@jit
def _f01234(e, E, M):
    """Solve Eq.

    21, 25, 26, 27, and 28 (f, f', f'', f''', and f'''')
    """
    f0 = E - e * jnp.sin(E) - M
    f1 = 1. - e * jnp.cos(E)
    f2 = e * jnp.sin(E)
    return f0, f1, f2, 1. - f1, -f2


@jit
def _d3(E, f):
    """Solve Eq.

    22
    """
    return -f[0] / (f[1] - 0.5 * f[0] * f[2] / f[1])


@jit
def _d4(E, f, d3):
    """Solve Eq.

    23
    """
    return -f[0] / (f[1] + 0.5 * d3 * f[2] + (d3**2) * f[3] / 6.)


@jit
def _d5(E, f, d4):
    """Solve Eq.

    24
    """
    return -f[0] / (f[1] + 0.5 * d4 * f[2] + d4**2 * f[3] / 6. + d4**3 * f[4] / 24.)


@jit
def getE(M, e):
    """JAX autograd compatible version of the Solver of Kepler's Equation for
    the "eccentric anomaly", E.

    Args:
       M : Mean anomaly
       e : Eccentricity

    Returns:
       Eccentric anomaly
    """
    pi = jnp.pi
    Mt = M - (jnp.floor(M / (2. * pi)) * 2. * pi)
    Mt = jnp.where(M > pi, 2.*pi - Mt, Mt)
    Mt = jnp.where(Mt == 0.0, 0.0, Mt)

    alpha = _alpha(e, Mt)
    d = _d(alpha, e)
    r = _r(alpha, d, Mt, e)
    q = _q(alpha, d, e, Mt)
    w = _w(r, q)
    E1 = _E1(d, r, w, q, Mt)
    f = _f01234(e, E1, Mt)
    d3 = _d3(E1, f)
    d4 = _d4(E1, f, d3)
    d5 = _d5(E1, f, d4)
    # Eq. 29
    E5 = E1 + d5
    # if flip:
    E5 = jnp.where(M > pi, 2. * pi - E5, E5)
    #    E5 = 2. * pi - E5
    E = E5
    return E5


if __name__ == '__main__':
    M = 0.1
    e = 0.2
    Ec = getE(M, e)
    print(Ec)
    grad_f = grad(getE, argnums=[0, 1])
    for e in np.linspace(0, 0.99, 100):
        print(grad_f(M, e))
