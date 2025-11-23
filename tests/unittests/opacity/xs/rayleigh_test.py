import numpy as np
import jax.numpy as jnp
import pytest
from exojax.opacity.rayleigh import xsvector_rayleigh_gas
from exojax.atm.polarizability import polarizability
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaRayleigh

def test_rayleigh():
    """
    Notes:
        See #430 https://github.com/HajimeKawahara/exojax/pull/430
    """
    N=1000
    nu_grid, wav, res = wavenumber_grid(300, 40000.0, N, xsmode="premodit", unit="nm")
    p = polarizability["N2"]
    xs_exojax = xsvector_rayleigh_gas(nu_grid, p, king_factor=1.0)
    val = (np.sum(np.log10(xs_exojax)))
    assert val == pytest.approx(-29561.42591713118)

def test_oparayleigh():
    N=1000
    nu_grid, wav, res = wavenumber_grid(300, 40000.0, N, xsmode="premodit", unit="nm")
    opa = OpaRayleigh(nu_grid,"N2")
    xs = opa.xsvector()
    val = (np.sum(np.log10(xs)))
    assert val == pytest.approx(-29561.42591713118)

if __name__ == "__main__":
    test_rayleigh()
    test_oparayleigh()