""" short integration tests for PreMODIT spectrum"""
import pytest
from jax.config import config
from exojax.test.emulate_mdb import mock_mdb
from exojax.spec.opacalc import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.atmrt import ArtEmisPure


@pytest.mark.parametrize("db, diffmode", [("exomol", 1), ("exomol", 2),
                                          ("hitemp", 1), ("hitemp", 2)])
def test_ArtEmisPure_ibased(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(pressure_top=1.e-5,
                      pressure_btm=1.e1,
                      nlayer=200,
                      nu_grid=nu_grid,
                      rtsolver="ibased",
                      nstream=4)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.01)
    gravity = 2478.57
    #gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    #mdb = api.MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)

    F0 = art.run(dtau, Tarr)

    return nu_grid, F0, F0
    

if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt
    diffmode = 0
    #nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)
    nus, F0, Fref = test_ArtEmisPure_ibased("exomol", diffmode)  #
    
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(nus, F0, label="intensity based")
    plt.legend()
    ax = fig.add_subplot(312)
    plt.legend()
    plt.ylabel("flux (cgs)")

    ax = fig.add_subplot(313)
    ax.plot(nus[10300:10700],
            1.0 - F0[10300:10700] / Fref[10300:10700],
            alpha=0.7,
            label="dif = (MO - PreMO)/MO Exomol")
    #ax.plot(nus_hitemp,
    #        1.0 - F0_hitemp / Fref_hitemp,
    #        alpha=0.7,
    #        label="dif = (MO - PreMO)/MO HITEMP")
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.05, color="gray", lw=0.5)
    plt.axhline(-0.05, color="gray", lw=0.5)
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.show()
