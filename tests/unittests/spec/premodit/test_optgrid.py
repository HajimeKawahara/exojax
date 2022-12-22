import pytest
    

def optelower(mdb, nu_grid, Tmax, Pmin):
    from exojax.spec.premodit import xsvector_zeroth
    from exojax.spec.opacalc import OpaPremodit
    from exojax.utils.constants import Tref_original
    from exojax.spec import normalized_doppler_sigma


    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, Twt=Tmax, diffmode=0)
    lbd_coeff, multi_index_uniqgrid, elower_grid, \
        ngamma_ref_grid, n_Texp_grid, R, pmarray = opa.opainfo
    nsigmaD = normalized_doppler_sigma(Tmax, Mmol, R)

    #for single temperature, 0-th order is sufficient 
    xsv = xsvector_zeroth(Tmax, Pmin, nsigmaD, lbd_coeff, Tref_original, R, pmarray,
                          opa.nu_grid, elower_grid, multi_index_uniqgrid,
                          ngamma_ref_grid, n_Texp_grid, qt)
    return eopt


def test_optelower():
    Tmax = 1000.0  #K
    Pmin = 0.1  #bar
    eopt = optelower(Tmax, Pmin)
