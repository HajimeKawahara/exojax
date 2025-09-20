from exojax.database.core.abscoeff import interp_logacia_matrix
from exojax.database.core.abscoeff import interp_logacia_vector
from exojax.database.cia.io import read_cia
from exojax.test.data import TESTDATA_H2_H2_CIA
from exojax.utils.grids import wavenumber_grid
from importlib.resources import files
import numpy as np
import pytest

def test_interp_logacia_matrix():
    nus = 4310.0
    nue = 4390.0
    filename = files("exojax").joinpath("data/testdata/" + TESTDATA_H2_H2_CIA)
    nucia, tcia, ac = read_cia(str(filename), nus, nue)
    Tarr = np.array([1000.0, 2000.0])
    logac = np.log10(ac)
    nu_grid, wav, r = wavenumber_grid(nus, nue, 10000, xsmode="premodit")
    logac_cia = interp_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac)
    assert np.all(np.shape(logac_cia) == (2, 10000))
    assert np.sum(logac_cia) == pytest.approx(-891133.44)


def test_interp_logacia_vector():
    nus = 4310.0
    nue = 4390.0
    filename = files("exojax").joinpath("data/testdata/" + TESTDATA_H2_H2_CIA)
    nucia, tcia, ac = read_cia(str(filename), nus, nue)
    T = 2000.0
    logac = np.log10(ac)
    nu_grid, wav, r = wavenumber_grid(nus, nue, 10000, xsmode="premodit")
    logac_cia = interp_logacia_vector(T, nu_grid, nucia, tcia, logac)
    assert np.all(np.shape(logac_cia) == (10000,))
    assert np.sum(logac_cia) == pytest.approx(-445566.72)


if __name__ == "__main__":
    test_interp_logacia_matrix()
    test_interp_logacia_vector()
