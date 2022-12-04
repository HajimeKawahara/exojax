"""unit tests for premodit basic functions

    * See premodit_xsection_test.py for the cross section test
    * See premodit_spectrum_test.py for the spectrum test

"""

import pytest
import numpy as np
from exojax.spec.premodit import compute_dElower
from exojax.spec.premodit import make_elower_grid
from exojax.spec.premodit import make_broadpar_grid
from exojax.spec.premodit import broadpar_getix
from exojax.spec.premodit import parallel_merge_grids
from exojax.spec.premodit import unbiased_ngamma_grid
from exojax.test.emulate_broadpar import mock_broadpar_exomol
from exojax.spec.premodit import unbiased_lsd_zeroth
from exojax.spec.premodit import unbiased_lsd_first


def test_compute_dElower():
    assert compute_dElower(
        1000.0, interval_contrast=0.1) == pytest.approx(160.03762408883165)


def test_make_elower_grid():
    maxe = 12001.0
    mine = 99.01
    eg = make_elower_grid(1000, [mine, maxe], 1.0)
    assert eg[-1] >= maxe and eg[0] <= mine


def test_parallel_merge_grids():
    grid1 = np.array([1, 2, 3])
    grid2 = np.array([4, 5, 7])
    mg = parallel_merge_grids(grid1, grid2)
    ref = np.array([[1, 4], [2, 5], [3, 7]])
    assert np.all(mg == pytest.approx(ref))


def test_make_broadpar_grid():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp = 3000.0
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(ngamma_ref,
                                                      n_Texp,
                                                      Ttyp,
                                                      dit_grid_resolution=0.2)
    assert np.all(
        ngamma_ref_grid == pytest.approx([0.1, 0.11447142, 0.13103707, 0.15]))
    assert np.all(n_Texp_grid == pytest.approx([0.4, 0.45, 0.5]))


def test_broadpar_getix():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp = 3000.0
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(ngamma_ref,
                                                      n_Texp,
                                                      Ttyp,
                                                      dit_grid_resolution=0.2)
    multi_index_lines, multi_cont_lines, uidx_lines, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)
    iline_interest = len(n_Texp) - 1
    uniq_index = uidx_lines[iline_interest]
    assert np.all(uidx_lines == [1, 0, 3, 1, 3, 2])
    assert np.all(
        multi_index_lines[iline_interest] == multi_index_uniqgrid[uniq_index])
    np.max(uidx_lines) + 1 == np.shape(neighbor_uidx)[0]
    assert np.all(
        multi_cont_lines[iline_interest] == pytest.approx([0.93739636, 0.]))
    assert uniq_index == 2
    assert np.all(neighbor_uidx[uniq_index, :] == [7, 4, 3])
    ref = [[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [0, 2], [1, 2], [2, 0],
           [3, 1], [2, 2], [3, 2]]
    assert np.all(multi_index_uniqgrid == ref)
    assert Ng_broadpar == len(multi_index_uniqgrid)


def test_unbias_ngamma_grid():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp = 3000.0
    Ttest = 2000.0
    Ptest = 10.0
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(ngamma_ref,
                                                      n_Texp,
                                                      Ttyp,
                                                      dit_grid_resolution=0.2)
    multi_index_lines, multi_cont_lines, uidx_lines, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)
    ngamma_grid = unbiased_ngamma_grid(Ttest, Ptest, ngamma_ref_grid,
                                       n_Texp_grid, multi_index_uniqgrid)
    ref = [
        0.46569834, 0.42327028, 0.53309152, 0.55464097, 0.48452351, 0.38470768,
        0.44038036, 0.61023745, 0.63490541, 0.50410967, 0.57706152
    ]
    assert np.all(ngamma_grid == pytest.approx(ref))


def _example_lbd():
    from exojax.utils.grids import wavenumber_grid
    N_nu_grid = 20000
    nu_grid, wav, resolution = wavenumber_grid(4000.0,
                                               4100.0,
                                               N_nu_grid,
                                               unit="cm-1",
                                               xsmode="premodit")
    n_E_h = 10
    n_broadening_k = 19
    lbd_zeroth = -300.0*np.ones((N_nu_grid, n_broadening_k, n_E_h))
    lbd_first = -300.0*np.ones((N_nu_grid, n_broadening_k, n_E_h))
    elower_grid = np.logspace(2.0, 5.0, n_E_h)
    qt = 1.0
    return lbd_zeroth, lbd_first, nu_grid, elower_grid, qt

from exojax.spec.premodit import logf_bias, g_bias
def test_unbiased_lsd_zeroth():
    from jax.config import config
    config.update("jax_enable_x64", True)

    lbd_zeroth, lbd_first, nu_grid, elower_grid, qt = _example_lbd()
    T = 1000.0
    print(logf_bias(elower_grid, T))
    print(g_bias(nu_grid, T))
    lsd = unbiased_lsd_zeroth(lbd_zeroth, T, nu_grid, elower_grid, qt)
    print(np.nansum(lsd))
    return


def test_unbiased_lsd_first():
    return


if __name__ == "__main__":
    test_unbiased_lsd_zeroth()