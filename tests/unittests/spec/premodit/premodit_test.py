import pytest
import numpy as np
from exojax.spec.premodit import compute_dElower
from exojax.spec.premodit import make_elower_grid
from exojax.spec.premodit import make_broadpar_grid
from exojax.spec.premodit import broadpar_getix, parallel_merge_grids
from exojax.test.emulate_broadpar import mock_broadpar_exomol
from exojax.test.emulate_mdb import mock_mdbExoMol
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.utils.instfunc import resolution_eslog


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


#if __name__ == "__main__":
#test_unbiased_lsd()
#test_make_lbd()
#test_merge_grids()
# test_make_broadpar_grid()
#test_broadpar_getix()
