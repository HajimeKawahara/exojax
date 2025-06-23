""" short integration tests for PreMODIT spectrum"""
import pytest
from jax import config
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure
from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("db, diffmode", [("exomol", 1), ("exomol", 2),
                                          ("hitemp", 1), ("hitemp", 2)])
def test_ArtEmisPure_ibased(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(pressure_top=1.e-5,
                      pressure_btm=1.e1,
                      nlayer=200,
                      nu_grid=nu_grid,
                      rtsolver="ibased",
                      nstream=8)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.01)
    gravity = 2478.57
    
    mdb = mock_mdb(db)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)

    #intenstiy based 8 stream
    F0_ibased = art.run(dtau, Tarr)

    #fluxed based 2 stream
    art.rtsolver = "fbased2st"
    F0_fbased = art.run(dtau, Tarr)

    return nu_grid, F0_ibased, F0_fbased


if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt
    diffmode = 0
    #nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)
    nus, F0i, F0f = test_ArtEmisPure_ibased("exomol", diffmode)  #

    import numpy as np
    np.savetxt("ibased_spec.txt",np.array([nus,F0i]).T)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(nus, F0i, label="intensity based")
    ax.plot(nus, F0f, label="flux based")
    plt.legend()

    ax = fig.add_subplot(212)
    ax.plot(nus, 1.0 - F0i / F0f, alpha=0.7, label="difference")
    plt.xlabel("wavenumber cm-1")
    plt.axhline(0.05, color="gray", lw=0.5)
    plt.axhline(-0.05, color="gray", lw=0.5)
    plt.axhline(0.01, color="gray", lw=0.5)
    plt.axhline(-0.01, color="gray", lw=0.5)
    #plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.show()
