import pytest
import numpy as np
from exojax.spec.premodit import make_elower_grid
from exojax.spec.premodit import make_broadpar_grid
from exojax.spec.premodit import generate_lbd
from exojax.spec.premodit import unbiased_lsd
from exojax.test.emulate_mdb import mock_mdbExoMol
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.utils.instfunc import resolution_eslog

def test_generate_lsd():
    """integrate to make (unbiased) LSD
    """
    interval_contrast = 0.1
    dit_grid_resolution = 0.1
    Ttyp = 2000.0
    mdb = mock_mdbExoMol()
    Nx = 5000
    nu_grid, wav, res = gen_wavenumber_grid(22800.0,
                                            23100.0,
                                            Nx,
                                            unit='AA',
                                            xsmode="premodit")
    R = resolution_eslog(nu_grid)
    ngamma_ref = mdb.alpha_ref / mdb.nu_lines * R
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
        ngamma_ref, mdb.n_Texp, Ttyp, dit_grid_resolution=dit_grid_resolution)
    elower_grid = make_elower_grid(Ttyp,
                                   mdb.elower,
                                   interval_contrast=interval_contrast)
    lbd = generate_lbd(mdb.Sij0, mdb.nu_lines, nu_grid, ngamma_ref,
                       ngamma_ref_grid, mdb.n_Texp, n_Texp_grid, mdb.elower,
                       elower_grid, Ttyp)
    assert np.sum(np.exp(lbd)) == pytest.approx(7.6101915e-20)

    Ttest=1500.0
    unlsd = unbiased_lsd(lbd, Ttest, nu_grid, elower_grid, mdb.qr_interp)
    assert np.sum(unlsd) == pytest.approx(7.472002e-20)

    
if __name__ == "__main__":
    test_generate_lsd()
