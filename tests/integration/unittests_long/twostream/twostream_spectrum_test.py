""" short integration tests for PreMODIT spectrum"""
import pytest
from jax.config import config
from exojax.test.emulate_mdb import mock_mdb
from exojax.spec.opacalc import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.atmrt import ArtEmisScat


@pytest.mark.parametrize("db, diffmode", [("exomol", 1), ("exomol", 2),
                                          ("hitemp", 1), ("hitemp", 2)])
def test_ArtEmisScat_LART_gives_consistent_results_with_pure_absorption(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisScat(pressure_top=1.e-5,
                      pressure_btm=1.e1,
                      nlayer=200,
                      nu_grid=nu_grid,
                      rtsolver="toon_hemispheric_mean")
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

    #almost pure absorption
    import jax.numpy as jnp
    single_scattering_albedo = jnp.ones_like(dtau) * 0.0001
    asymmetric_parameter = jnp.ones_like(dtau) * 0.0001

    F0 = art.run(dtau, single_scattering_albedo, asymmetric_parameter, Tarr, show=True)

    return nu_grid, F0, F0
    
    


if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt
    diffmode = 0
    #nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)
    nus, F0, Fref = test_ArtEmisScat_LART_gives_consistent_results_with_pure_absorption("exomol", diffmode)  #
    
    fig = plt.figure()
    ax = fig.add_subplot(311)
    #ax.plot(nus[10300:10700], F0[10300:10700], label="Toon (ExoMol)")
    ax.plot(nus, F0, label="Toon (ExoMol)")
    plt.legend()
    #plt.yscale("log")
    ax = fig.add_subplot(312)
    #ax.plot(nus_hitemp, F0_hitemp, label="PreMODIT (HITEMP)", ls="dashed")
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
