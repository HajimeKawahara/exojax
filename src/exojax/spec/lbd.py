import numpy as np
from exojax.utils.constants import hcperk
import jax.numpy as jnp
from jax import grad

def _f(t, tref, E):
    return jnp.exp(- hcperk * (t - tref) * E)

def weight_point1(t, tref, El, E1, E2):
    """weight at point 1 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        El (float): line envergy (cm-1)
        E1 (float): energy at rid point 1
        E2 (float): energy at rid point 2

    Returns:
        weight at point 1
    """
    xl = _f(t, tref, El)
    x1 = _f(t, tref, E1)
    x2 = _f(t, tref, E2)
    return (x2 - xl) / (x2 - x1)


def weight_point2(t, tref, El, E1, E2):
    """weight at point 2 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        El (float): line envergy (cm-1)
        E1 (float): energy at rid point 1
        E2 (float): energy at rid point 2

    Returns:
        weight at point 2
    """
    return 1.0 - weight_point1(t, tref, El, E1, E2)





