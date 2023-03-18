""" short integration tests for PreMODIT transmission"""
from jax.config import config
import pandas as pd
import numpy as np
import jax.numpy as jnp
from exojax.utils.grids import wavenumber_grid
from exojax.spec.opacalc import OpaModit
from exojax.spec.atmrt import ArtTransPure
from exojax.utils.constants import RJ, Rs
from exojax.spec.api import MdbHitran
from exojax.spec.api import MdbHitemp
from exojax.utils.astrofunc import gravity_jupiter
    
config.update("jax_enable_x64", True)


def read_kawashima_data():
    filename = "spectrum/CO100percent_500K.dat"
    dat = pd.read_csv(filename, delimiter="   ")
    wav = dat["Wavelength[um]"]
    mask = (wav > 2.25) & (wav < 2.6)
    #mask = (wav > 0.25) & (wav < 4.6)
    return wav[mask], dat["Rp/Rs"][mask]


def compare_with_kawashima_code():
    mu_fid = 28.00863
    T_fid = 500.

    Nx = 300000
    nu_grid, wav, res = wavenumber_grid(22900.0,
                                        26000.0,
                                        Nx,
                                        unit="AA",
                                        xsmode="premodit")

    art = ArtTransPure(nu_grid,
                       pressure_top=1.e-15,
                       pressure_btm=1.e1,
                       nlayer=100)
    art.change_temperature_range(490.0, 510.0)
    Tarr = T_fid * np.ones_like(art.pressure)
    mmr_arr = art.constant_mmr_profile(1.0)
    gravity_btm = gravity_jupiter(1.0, 1.0)
    radius_btm = RJ

    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    mdb = MdbHitran('CO', art.nu_grid, gpu_transfer=True)
    
    #mdb = MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)

    mmw = mu_fid * np.ones_like(art.pressure)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    #HITRAN CO has not so many lines. So we use MODIT
    #opa = OpaModit(mdb=mdb, nu_grid=nu_grid, Tarr_list=Tarr, Parr=np.zeros_like(art.pressure), Pself_ref = art.pressure) #unknown error
    opa = OpaModit(mdb=mdb, nu_grid=nu_grid, Tarr_list=Tarr, Parr = art.pressure)
              
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)

    Rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    return nu_grid, np.sqrt(Rp2) * radius_btm / Rs


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    wav, rprs = read_kawashima_data()
    diffmode = 1
    nus_hitran, Rp_hitran = compare_with_kawashima_code()
    from exojax.spec.unitconvert import nu2wav
    wav_exojax = nu2wav(nus_hitran, unit="um")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wav, rprs * Rs / RJ, label="Kawashima")
    #plt.yscale("log")
    ax.plot(wav_exojax[::-1], Rp_hitran * Rs / RJ, label="ExoJAX")
    plt.legend()

    plt.xlabel("wavenumber cm-1")
    #plt.ylim(-0.07, 0.07)
    plt.legend()
    plt.ylabel("Rp (RJ)")

    plt.savefig("comparison.png")
    plt.show()
