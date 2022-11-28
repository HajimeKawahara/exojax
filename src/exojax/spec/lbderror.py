from exojax.utils.constants import hcperk
import jax.numpy as jnp
from jax import grad

def _beta(t, tref):
    return hcperk * (t - tref)

def weight_point2_dE(t, tref, dE, p=0.5):
    """dE version of the weight at point 2 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        dE (float): envergy interval between points 1 nad 2 (cm-1)
        p (float): between 0 to 1

    Returns:
        weight at point 2
    """

    fac1 = 1.0 - jnp.exp(_beta(t, tref) * p * dE)
    fac2 = jnp.exp(-_beta(t, tref) *
                   (1.0 - p) * dE) - jnp.exp(_beta(t, tref) * p * dE)
    return fac1 / fac2

def weight_point1_dE(t, tref, dE, p=0.5):
    """dE version of the weight at point 1 for PreMODIT

    Args:
        t (float): inverse temperature
        tref (float): reference inverse temperature
        dE (float): envergy interval between points 1 nad 2 (cm-1)
        p (float): between 0 to 1

    Returns:
        weight at point 1
    """
    return 1.0 - weight_point2_dE(t, tref, dE, p)


def single_tilde_line_strength(t, w1, w2, tref, dE, p=0.5):
    """

    Args:
        t (float): inverse temperature
        w1 (_type_): weight at point 1
        w2 (_type_): weight at point 2
        tref (_type_): reference temperature
        dE (_type_): energy interval in cm-1
        p (float, optional): between 0 to 1 Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    v1 = w1 * jnp.exp(_beta(t, tref) * p * dE)
    v2 = w2 * jnp.exp(-_beta(t, tref) * (1.0 - p) * dE)
    return v1 + v2 - 1.0


def single_tilde_line_strength_zeroth(t, twp, tref, dE, p=0.5):
    w1 = weight_point1_dE(twp, tref, dE, p)
    w2 = weight_point2_dE(twp, tref, dE, p)
    return single_tilde_line_strength(t, w1, w2, tref, dE, p)


def single_tilde_line_strength_first(t, twp, tref, dE, p=0.5):
    dfw1 = grad(weight_point1_dE, argnums=0)
    dfw2 = grad(weight_point2_dE, argnums=0)
    w1 = weight_point1_dE(twp, tref, dE,
                          p) + dfw1(twp, tref, dE, p) * (t - twp)
    w2 = weight_point2_dE(twp, tref, dE,
                          p) + dfw2(twp, tref, dE, p) * (t - twp)
    return single_tilde_line_strength(t, w1, w2, tref, dE, p)
