import jax.numpy as jnp
from jax import jit


@jit
def E1(x):
    """Abramowitz Stegun (1970) approximation of the exponential integral of
    the first order, E1.

    Args:
       x: input

    Returns:
       The exponential integral of the first order, E1(x)
    """
    A0 = -0.57721566
    A1 = 0.99999193
    A2 = -0.24991055
    A3 = 0.05519968
    A4 = -0.00976004
    A5 = 0.00107857
    B1 = 8.5733287401
    B2 = 18.059016973
    B3 = 8.6347608925
    B4 = 0.2677737343
    C1 = 9.5733223454
    C2 = 25.6329561486
    C3 = 21.0996530827
    C4 = 3.9584969228

    x2 = x**2
    x3 = x**3
    x4 = x**4
    x5 = x**5
    ep1A = -jnp.log(x)+A0+A1*x+A2*x2+A3*x3+A4*x4+A5*x5
    ep1B = jnp.exp(-x)/x *\
        (x4+B1*x3+B2*x2+B3*x+B4) /\
        (x4+C1*x3+C2*x2+C3*x+C4)
    ep = jnp.where(x <= 1.0, ep1A, ep1B)
    return ep
