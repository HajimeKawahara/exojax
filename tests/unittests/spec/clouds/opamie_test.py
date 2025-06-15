"""opacity for mie test
"""
from exojax.test.emulate_pdb import mock_PdbPlouds
from exojax.opacity import OpaMie
from exojax.utils.grids import wavenumber_grid
import numpy as np
import jax.numpy as jnp


def test_loading_opamie():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    N = 1000
    nus, wav, res = wavenumber_grid(12050.0, 15950.0, N, xsmode="premodit")
    opa = OpaMie(pdb, nus)


def test_mieparams_vector():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    N = 1000
    nus, wav, res = wavenumber_grid(12050.0, 15950.0, N, xsmode="premodit")
    opa = OpaMie(pdb, nus)
    rg = 1.0e-5
    sigmag = 2.0
    dtau, w, g = opa.mieparams_vector(rg, sigmag)


def test_mieparams_matrix():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    N = 1000
    nus, wav, res = wavenumber_grid(12050.0, 15950.0, N, xsmode="premodit")
    opa = OpaMie(pdb, nus)
    rg_layer = jnp.array([1.0e-5, 2.0e-5])
    sigmag_layer = jnp.array([2.0, 1.0])
    dtau, w, g = opa.mieparams_matrix(rg_layer, sigmag_layer)

    # shape check
    assert np.all(np.shape(dtau) == np.array([len(rg_layer), N]))
    assert np.all(np.shape(w) == np.array([len(rg_layer), N]))
    assert np.all(np.shape(g) == np.array([len(rg_layer), N]))

    # not validated yet
    print(np.real(np.sum(dtau)))
    print(np.real(np.nansum(w)))
    print(np.sum(g))


if __name__ == "__main__":
    # test_loading_opamie()
    # test_mieparams_vector()
    test_mieparams_matrix()
