import numpy as np
import jax.numpy as jnp
import pytest
from exojax.spec.hminus import free_free_absorption
from exojax.spec.hminus import bound_free_absorption


def test_hminus_ff():
    Tin = 3000.0
    wav = 1.4
    ref = 2.0075e-26
    val = free_free_absorption(wav, Tin)
    diff = np.abs(ref-val)
    assert diff < 1.e-30


def test_hminus_bf():
    Tin = 3000.0
    wav = 1.4
    ref = 4.065769e-25
    val = bound_free_absorption(wav, Tin)
    diff = np.abs(ref-val)
    print(diff)
    assert diff < 1.e-30


if __name__ == '__main__':
    test_hminus_ff()
    test_hminus_bf()
