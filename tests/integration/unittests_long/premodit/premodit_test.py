"""unit tests for premodit basic functions

    * See premodit_xsection_test.py for the cross section test
    * See premodit_spectrum_test.py for the spectrum test

"""

import pytest
import numpy as np
from exojax.opacity.premodit.premodit import compute_dElower
from exojax.opacity.premodit.premodit import make_elower_grid
from exojax.opacity.premodit.premodit import make_broadpar_grid
from exojax.opacity.premodit.premodit import broadpar_getix
from exojax.opacity.premodit.premodit import parallel_merge_grids
from exojax.opacity.premodit.premodit import unbiased_ngamma_grid
from exojax.test.emulate_broadpar import mock_broadpar_exomol
from exojax.test.emulate_broadpar import mock_broadpar
from exojax.opacity.premodit.premodit import logf_bias, g_bias
from exojax.utils.constants import Tref_original
from exojax.opacity.premodit.premodit import unbiased_lsd_zeroth
from exojax.opacity.premodit.premodit import unbiased_lsd_first
from exojax.opacity.premodit.premodit import unbiased_lsd_second
from exojax.opacity.premodit.premodit import reference_temperature_broadening_at_midpoint


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
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
        ngamma_ref,
        n_Texp,
        Tmax=Ttyp,
        Tmin=Tref_original,
        Tref_broadening=Tref_original,
        twod_factor=1.0,
        dit_grid_resolution=0.2)
    assert np.all(
        ngamma_ref_grid == pytest.approx([0.1, 0.11447142, 0.13103707, 0.15]))
    assert np.all(n_Texp_grid == pytest.approx([0.4, 0.45, 0.5]))


def test_broadpar_getix():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp = 3000.0
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
        ngamma_ref,
        n_Texp,
        Tmax=Ttyp,
        Tmin=Tref_original,
        Tref_broadening=Tref_original,
        twod_factor=1.0,
        dit_grid_resolution=0.2)

    multi_index_lines, multi_cont_lines, uidx_lines, neighbor_uidx, multi_index_uniqgrid, ngrid_broadpar = broadpar_getix(
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
    assert ngrid_broadpar == len(multi_index_uniqgrid)


    

def test_unbias_ngamma_grid():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp = 3000.0
    Ttest = 2000.0
    Ptest = 10.0
    Tref_broadening = Tref_original
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
        ngamma_ref,
        n_Texp,
        Tmax=Ttyp,
        Tmin=Tref_original,
        Tref_broadening=Tref_broadening,
        twod_factor=1.0,
        dit_grid_resolution=0.2)
    multi_index_lines, multi_cont_lines, uidx_lines, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)

    ngamma_grid = unbiased_ngamma_grid(Ttest, Ptest, ngamma_ref_grid,
                                       n_Texp_grid, multi_index_uniqgrid,
                                       Tref_broadening)

    ref = [
        0.46569834, 0.42327028, 0.53309152, 0.55464097, 0.48452351, 0.38470768,
        0.44038036, 0.61023745, 0.63490541, 0.50410967, 0.57706152
    ]
    assert np.all(ngamma_grid == pytest.approx(ref))


def test_unbias_ngamma_grid_works_for_single_broadening_parameter():
    Nline=10
    ngamma_ref_grid = np.array([1.0])
    n_Texp_grid = np.array([0.5])    
    Ttest = 2000.0
    Ptest = 10.0
    Tref_broadening = Tref_original
    multi_index_uniqgrid = np.array([[0,0]])

    ngamma_grid = unbiased_ngamma_grid(Ttest, Ptest, ngamma_ref_grid,
                                       n_Texp_grid, multi_index_uniqgrid,
                                       Tref_broadening)

    assert ngamma_grid[0] == pytest.approx(3.84707681)

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
    from jax import config
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

@pytest.mark.parametrize("db", ["exomol","hitemp"])
def test_broadpar_grid_as_a_function_of_Tref_broadening(db):
    """ comparison of non-optimized and optimized broadening parameter grid in PreMODIT #366 
    """
    ngamma_ref, n_Texp = mock_broadpar(db)
    Tmax = 3000.0
    Tmin = 400.0

    # use original gamma and n
    ngamma_ref_grid_1, n_Texp_grid_1 = make_broadpar_grid(
        ngamma_ref,
        n_Texp,
        Tmax=Tmax,
        Tmin=Tmin,
        Tref_broadening=Tref_original,
        dit_grid_resolution=0.1)

    # rescale gamma assuming Tref_broadning at midpoint (optimized)
    ngamma_ref_grid_2, n_Texp_grid_2 = make_broadpar_grid(
        ngamma_ref,
        n_Texp,
        Tmax=Tmax,
        Tmin=Tref_original,
        Tref_broadening=reference_temperature_broadening_at_midpoint(
            Tmax, Tmin),
        dit_grid_resolution=0.1)
    print("using midpoint reduces the number of n_Texp_grid from ",
          len(n_Texp_grid_1), "to", len(n_Texp_grid_2))

    assert len(n_Texp_grid_2) < len(n_Texp_grid_1)


if __name__ == "__main__":
    #test_broadpar_grid_as_a_function_of_Tref_broadening("exomol")
    #test_broadpar_grid_as_a_function_of_Tref_broadening("hitemp")
    test_unbias_ngamma_grid_works_for_single_broadening_parameter()
    #test_unbiased_lsd()
    #test_make_elower_grid()
    #test_make_broadpar_grid()