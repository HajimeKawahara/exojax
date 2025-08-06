""" short integration tests for PreMODIT transmission"""
import pytest
from jax import config
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtTransPure
from exojax.utils.constants import RJ

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("db, diffmode", [("exomol", 1), ("exomol", 2),
                                          ("hitemp", 1), ("hitemp", 2)])
def test_rt(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtTransPure(pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    mmw = 2.33 * np.ones_like(art.pressure)
    gravity_btm = 2478.57
    radius_btm = RJ
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    mdb = mock_mdb(db)
    #mdb =MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    #mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)
    Rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    print(Rp2)
    return nu_grid, np.sqrt(Rp2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    diffmode = 0
    nus_hitemp, Rp_hitemp = test_rt("hitemp", diffmode)  #
    nus, Rp = test_rt("exomol", diffmode)  #
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(nus, Rp, label="PreMODIT (ExoMol)", ls="dashed")
    plt.legend()
    plt.ylabel("radius (RJ)")

    #plt.yscale("log")
    ax = fig.add_subplot(212)
    ax.plot(nus_hitemp, Rp_hitemp, label="PreMODIT (HITEMP)", ls="dashed")
    plt.legend()
    plt.ylabel("radius (RJ)")

    plt.xlabel("wavenumber cm-1")
    #plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.savefig("transmission_co.png")
    plt.show()
