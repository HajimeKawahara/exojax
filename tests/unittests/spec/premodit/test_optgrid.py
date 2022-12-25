from exojax.test.emulate_mdb import mock_mdbExomol
import pytest
    

def optelower(mdb, nu_grid, Tmax, Pmin):
    from exojax.spec.premodit import xsvector_zeroth
    from exojax.spec.opacalc import OpaPremodit
    from exojax.utils.constants import Tref_original
    from exojax.spec import normalized_doppler_sigma


    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, Twt=Tmax, diffmode=0)
    lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    nsigmaD = normalized_doppler_sigma(Tmax, mdb.molmass, R)
    qt = mdb.qr_interp(Tmax)
    
    #for single temperature, 0-th order is sufficient 
    xsv = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff, Tref_original, R, pmarray,
                          opa.nu_grid, elower_grid, multi_index_uniqgrid,
                          ngamma_ref_grid, n_Texp_grid, qt)
    print(xsv)
    eopt=1.0
    return eopt


def test_optelower():
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    Nx = 5000
    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")
    Tmax = 1000.0  #K
    Pmin = 0.1  #bar
    mdb = mock_mdbExomol()
    eopt = optelower(mdb, nu_grid, Tmax, Pmin)


if __name__ == "__main__":
    test_optelower()