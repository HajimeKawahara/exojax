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
from exojax.spec.premodit import logf_bias, g_bias
from exojax.utils.constants import Tref_original
from exojax.spec.premodit import unbiased_lsd_zeroth
from exojax.spec.premodit import unbiased_lsd_first
from exojax.spec.premodit import unbiased_lsd_second


def test_compute_dElower():
    assert compute_dElower(
        1000.0, interval_contrast=0.1) == pytest.approx(160.03762408883165)


def test_make_elower_grid():
    elower = np.array([1.0, 10.0, 100.0])
    dE = 25.0
    elower_grid = make_elower_grid(elower, dE)
    assert np.all(elower_grid == [1., 26., 51., 76., 101.])


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
    lbd_zeroth = -50.0 * np.ones((N_nu_grid, n_broadening_k, n_E_h))
    lbd_first = 1.e-18 * np.ones((N_nu_grid, n_broadening_k, n_E_h))
    lbd_second = 1.e-18 * np.ones((N_nu_grid, n_broadening_k, n_E_h))

    elower_grid = np.logspace(2.0, 5.0, n_E_h)
    qt = 1.0
    return lbd_zeroth, lbd_first, lbd_second, nu_grid, elower_grid, qt


def test_unbiased_lsd():
    from jax.config import config
    config.update("jax_enable_x64", True)

    lbd_zeroth, lbd_first, lbd_second, nu_grid, elower_grid, qt = _example_lbd(
    )
    T = 1000.0
    assert np.sum(logf_bias(elower_grid, T,
                            Tref_original)) == 638.3176862916531
    assert np.sum(g_bias(nu_grid, T, Tref_original)) == 19940.996503337694

    Tref = 500.0
    Twt = 1100.0
    qt = 1.0
    ref = [
        2.2340181325030985e+46, 3.2869932381896974e+46, 3.287041100694501e+46
    ]

    #for zeroth, the input is lbd_zeroth
    lsd = unbiased_lsd_zeroth(lbd_zeroth, T, Tref, nu_grid, elower_grid, qt)
    #print(np.sum(lsd))
    assert np.sum(lsd) == pytest.approx(ref[0])

    #for first and second, the input is lbd_coeff
    lbd_coeff = [lbd_zeroth, lbd_first, lbd_second]
    lsd = unbiased_lsd_first(lbd_coeff, T, Tref, Twt, nu_grid, elower_grid, qt)
    #print(np.sum(lsd))
    assert np.sum(lsd) == pytest.approx(ref[1])

    lsd = unbiased_lsd_second(lbd_coeff, T, Tref, Twt, nu_grid, elower_grid,
                              qt)
    #print(np.sum(lsd))
    assert np.sum(lsd) == pytest.approx(ref[2])


if __name__ == "__main__":
    test_unbiased_lsd()
    #test_make_elower_grid()