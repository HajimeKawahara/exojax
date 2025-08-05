""" short integration tests for PreMODIT transmission"""

from jax import config
from importlib.resources import files
import pandas as pd
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ, Rs
from exojax.opacity import OpaModit
from exojax.rt import ArtTransPure
from exojax.database.api  import MdbHitran
from exojax.test.data import COMPDATA_TRANSMISSION_CO

config.update("jax_enable_x64", True)


def compress_original_kawashima_data():
    """generate feather file from original data"""
    filename = "spectrum/CO100percent_500K.dat"
    dat = pd.read_csv(filename, delimiter="   ")
    wav = dat["Wavelength[um]"]
    mask = (wav > 2.29) & (wav < 2.6)
    dat = dat[mask]
    dat[["Wavelength[um]", "Rp/Rs"]].to_feather(COMPDATA_TRANSMISSION_CO)
    print("put the feather file to exojax/data/testdata")


def read_kawashima_data():
    filename = files("exojax").joinpath("data/testdata/" + COMPDATA_TRANSMISSION_CO)
    dat = pd.read_feather(filename)
    return dat["Wavelength[um]"], dat["Rp/Rs"]


def compare_with_kawashima_code():
    mu_fid = 28.00863
    T_fid = 500.0
    Nx = 300000
    nu_grid, wav, res = wavenumber_grid(22900.0, 26000.0, Nx, unit="AA", xsmode="modit")

    art = ArtTransPure(pressure_top=1.0e-15, pressure_btm=1.0e1, nlayer=100)
    art.change_temperature_range(490.0, 510.0)
    Tarr = T_fid * np.ones_like(art.pressure)
    mmr_arr = art.constant_profile(1.0)
    gravity_btm = gravity_jupiter(1.0, 1.0)
    radius_btm = RJ

    # mdb =MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    mdb = MdbHitran("CO", nurange=nu_grid, gpu_transfer=True, inherit_dataframe=False)
    # mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)

    mmw = mu_fid * np.ones_like(art.pressure)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    # HITRAN CO has not so many lines. So we use MODIT
    # opa = OpaModit(mdb=mdb, nu_grid=nu_grid, Tarr_list=Tarr, Parr=np.zeros_like(art.pressure), Pself_ref = art.pressure) #unknown error
    opa = OpaModit(mdb=mdb, nu_grid=nu_grid, Tarr_list=Tarr, Parr=art.pressure)

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

    ### Rayleigh scattering
    from exojax.atm.polarizability import polarizability
    from exojax.atm.polarizability import king_correction_factor
    from exojax.opacity.rayleigh import xsvector_rayleigh_gas

    xsvector_rayleigh = xsvector_rayleigh_gas(
        nu_grid, polarizability["CO"], king_correction_factor["CO"]
    )
    dtau_ray = art.opacity_profile_xs(
        xsvector_rayleigh, mmr_arr, opa.mdb.molmass, gravity
    )
    print(np.max(dtau_ray))
    dtau = dtau + dtau_ray
    ###

    art.set_integration_scheme("simpson")
    Rp2_simpson = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    art.set_integration_scheme("trapezoid")
    Rp2_trapezoid = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)

    return (
        nu_grid,
        np.sqrt(Rp2_trapezoid) * radius_btm / Rs,
        np.sqrt(Rp2_simpson) * radius_btm / Rs,
    )


if __name__ == "__main__":
    # compress_original_kawashima_data() # when you want to make feather file from original data
    import matplotlib.pyplot as plt

    wav, rprs = read_kawashima_data()
    diffmode = 1
    nus_hitran, Rp_trapezoid, Rp_simpson = compare_with_kawashima_code()
    from exojax.utils.grids import nu2wav

    wav_exojax = nu2wav(nus_hitran, unit="um", wavelength_order="ascending")
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(wav, rprs * Rs / RJ, label="Kawashima")
    # plt.yscale("log")
    ax.plot(
        wav_exojax[::-1], Rp_trapezoid * Rs / RJ, label="ExoJAX trapezoid", ls="dotted"
    )
    ax.plot(wav_exojax[::-1], Rp_simpson * Rs / RJ, label="ExoJAX simpson", lw=1)
    plt.legend()
    plt.ylabel("Rp (RJ)")
    ax = fig.add_subplot(212)
    ax.plot(
        wav_exojax[::-1],
        (1.0 - Rp_trapezoid / Rp_simpson) * Rs / RJ,
        label="Delta R/R (simpson, trapezoid)",
    )
    plt.xlabel("wavenumber cm-1")
    # plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.ylabel("ratio")

    plt.savefig("comparison.png")
    plt.show()
