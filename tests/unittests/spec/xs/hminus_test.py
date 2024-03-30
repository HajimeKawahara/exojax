import numpy as np
import jax.numpy as jnp
import pytest
from exojax.spec.hminus import free_free_absorption
from exojax.spec.hminus import bound_free_absorption
from exojax.spec.opacont import OpaHminus
from exojax.utils.grids import wavenumber_grid


def test_hminus_ff():
    Tin = 3000.0
    wav = 1.4
    ref = 2.0075e-26
    val = free_free_absorption(wav, Tin)
    diff = np.abs(ref - val)
    assert diff < 1.0e-30


def test_hminus_bf():
    Tin = 3000.0
    wav = 1.4
    ref = 4.065769e-25
    val = bound_free_absorption(wav, Tin)
    diff = np.abs(ref - val)
    print(diff)
    assert diff < 1.0e-30


def test_opahminus():
    N = 1000
    Tarr = jnp.array([3000.0])
    ne = jnp.array([1.0])
    nh = jnp.array([1.0])
    nu_grid, wav, res = wavenumber_grid(9000.0, 18000.0, N, xsmode="premodit")
    opa = OpaHminus(nu_grid)

    a = opa.logahminus_matrix(Tarr, ne, nh)
    assert np.sum(a) == pytest.approx(-36358.67)


if __name__ == "__main__":
    test_hminus_ff()
    test_hminus_bf()
    test_opahminus()
