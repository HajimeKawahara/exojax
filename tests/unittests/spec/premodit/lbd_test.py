"""unit tests for premodit LBD 

"""
import pytest
import numpy as np
from jax import grad
from exojax.utils.constants import hcperk
from exojax.spec.lbd import weight_point1, weight_point2
from jax.config import config
config.update("jax_enable_x64", True)

def lbd_first_approx(T, Twp, lbd_zeroth_Twp, lbd_first_Twp):
    """construct lbd from zeroth and first terms in the first Taylor approximation

    Args:
        T (_type_): _description_
        Twp (_type_): _description_
        lbd_zeroth_Twp (_type_): _description_
        lbd_first_Twp (_type_): _description_

    Returns:
        _type_: LBD (line basis density at T)
    """
    return lbd_zeroth_Twp + lbd_first_Twp*(1.0/T - 1.0/Twp)


def test_construct_lbd():
    return

if __name__ == "__main__":
    test_construct_lbd()
