from exojax.test.emulate_mdb import mock_mdbExomol
import pytest
import numpy as np

def optelower(mdb, nu_grid, Tmax, Pmin):
    from exojax.spec.premodit import xsvector_zeroth
    from exojax.spec.opacalc import OpaPremodit
    from exojax.utils.constants import Tref_original
    from exojax.spec import normalized_doppler_sigma
    import matplotlib.pyplot as plt

    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, Twt=Tmax, dE=100.0, diffmode=0)
    lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    nsigmaD = normalized_doppler_sigma(Tmax, mdb.molmass, R)
    qt = mdb.qr_interp(Tmax)
    
    #for single temperature, 0-th order is sufficient
    q=-2
    xsv_master = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff, Tref_original, R, pmarray,
                          opa.nu_grid, elower_grid, multi_index_uniqgrid,
                          ngamma_ref_grid, n_Texp_grid, qt)
    print(np.shape(elower_grid),np.shape(lbd_coeff))
    xsv = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff[:,:,:,:q], Tref_original, R, pmarray,
                          opa.nu_grid, elower_grid[:q], multi_index_uniqgrid,
                          ngamma_ref_grid, n_Texp_grid, qt)

    fig=plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(nu_grid, xsv_master)
    plt.plot(nu_grid, xsv)
    
    plt.yscale("log") 
    ax = fig.add_subplot(212)
    plt.plot(nu_grid, xsv/xsv_master - 1.0)
    plt.ylim(-0.03,0.03)
    plt.show()
    
    eopt=1.0
    return eopt


def test_optelower():
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    Nx = 15000
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