""" short integration tests for PreMODIT spectrum"""
import pytest
from jax import config
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisScat


def generate_spectrum(db, diffmode, rtsolver, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisScat(pressure_top=1.e-5,
                      pressure_btm=1.e1,
                      nlayer=200,
                      nu_grid=nu_grid,
                      rtsolver=rtsolver)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.01)
    gravity = 2478.57
    #gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    #mdb =MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    #mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)

    #almost pure absorption
    import jax.numpy as jnp
    single_scattering_albedo = jnp.ones_like(dtau) * 0.99
    asymmetric_parameter = jnp.ones_like(dtau) * 0.5

    F0 = art.run(dtau, single_scattering_albedo, asymmetric_parameter, Tarr)

    return nu_grid, F0
    
    


if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt
    diffmode = 0
    nus, F0_lart = generate_spectrum("exomol", diffmode, rtsolver="lart_toon_hemispheric_mean")  #
    nus, F0_fluxadd = generate_spectrum("exomol", diffmode, rtsolver="fluxadding_toon_hemispheric_mean")  #
    

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(nus, F0_lart, label="LART/Toon (ExoMol) g=0.5, omega=0.5")
    ax.plot(nus, F0_fluxadd, label="FluxAdding/Toon (ExoMol) g=0.5, omega=0.5", ls="dashed")
    plt.legend()
    ax = fig.add_subplot(212)
    ax.plot(nus, 1.0-F0_fluxadd/F0_lart, label="diff")
    ax.set_ylim(-0.01,0.01)
    plt.xlabel("wavenumber cm-1")
    plt.legend()
    plt.savefig("lart_fluxadding_comparison_scat.png")
    plt.show()
