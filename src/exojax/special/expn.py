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

    # Branch-local inputs keep the inactive expression from contaminating autodiff.
    use_small_x = x <= 1.0
    x_small = jnp.where(use_small_x, x, 1.0)
    x_large = jnp.where(use_small_x, 1.0, x)

    x2 = x_small**2
    x3 = x_small**3
    x4 = x_small**4
    x5 = x_small**5
    ep1A = -jnp.log(x_small) + A0 + A1 * x_small + A2 * x2 + A3 * x3 + A4 * x4 + A5 * x5

    z = 1.0 / x_large
    z2 = z**2
    z3 = z**3
    z4 = z**4
    ep1B = (
        jnp.exp(-x_large)
        * z
        * (1.0 + B1 * z + B2 * z2 + B3 * z3 + B4 * z4)
        / (1.0 + C1 * z + C2 * z2 + C3 * z3 + C4 * z4)
    )
    ep = jnp.where(use_small_x, ep1A, ep1B)
    return ep
