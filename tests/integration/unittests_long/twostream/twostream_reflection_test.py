""" short integration tests for PreMODIT spectrum"""

import pytest
from jax import config
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtReflectEmis
from exojax.rt import ArtReflectPure
from jax import config

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "db, diffmode", [("exomol", 1), ("exomol", 2), ("hitemp", 1), ("hitemp", 2)]
)
def test_ArtReflectPure_no_scattering_reflected_by_surface(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtReflectPure(
        pressure_top=1.0e-5, pressure_btm=1.0e0, nlayer=200, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.0001)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb =MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[art.Tlow, art.Thigh]
    )

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

    # almost pure absorption
    import jax.numpy as jnp

    single_scattering_albedo = jnp.ones_like(dtau) * 0.0001
    asymmetric_parameter = jnp.ones_like(dtau) * 0.0001

    albedo = 0.5
    incoming_flux = jnp.ones_like(nu_grid)
    reflectivity_surface = albedo * jnp.ones_like(nu_grid)
    F0 = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        reflectivity_surface,
        incoming_flux,
    )

    return nu_grid, F0


@pytest.mark.parametrize(
    "db, diffmode", [("exomol", 1), ("exomol", 2), ("hitemp", 1), ("hitemp", 2)]
)
def test_ArtReflectEmis_Emission_plus_stellar_refelction(db, diffmode, fig=False):
    from exojax.rt.planck import piBarr

    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtReflectEmis(
        pressure_top=1.0e-5, pressure_btm=1.0e0, nlayer=200, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.0001)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb =MdbExomol('.database/CO/12C-1edt mru 6O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[art.Tlow, art.Thigh]
    )

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

    # almost pure absorption
    import jax.numpy as jnp

    single_scattering_albedo = jnp.ones_like(dtau) * 0.0001
    asymmetric_parameter = jnp.ones_like(dtau) * 0.0001

    albedo = 0.5
    incoming_flux = piBarr(jnp.array([1000.0]), nu_grid)[0, :]  # 1500K incoming
    reflectivity_surface = albedo * jnp.ones_like(nu_grid)

    F0 = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        Tarr,
        jnp.zeros_like(nu_grid),
        reflectivity_surface,
        incoming_flux,
    )

    return nu_grid, F0


if __name__ == "__main__":
    config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt

    diffmode = 0
    # nus_hitemp, F0_hitemp, Fref_hitemp = test_rt("hitemp", diffmode)
    nus, F0 = test_ArtReflectEmis_Emission_plus_stellar_refelction("exomol", diffmode)
    # nus, F0 = test_ArtReflectPure_no_scattering_reflected_by_surface("exomol", diffmode)  #

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(nus, F0, label="refelected+emission light")
    plt.xlabel("wavenumber cm-1")
    plt.legend()
    plt.show()
