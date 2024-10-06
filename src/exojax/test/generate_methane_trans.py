""" Geneartes Methane transmission spectrum using PreMODIT
"""

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from exojax.utils.grids import wavenumber_grid
from exojax.spec.atmrt import ArtTransPure
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_TRANS
from exojax.utils.constants import RJ, Rs
from exojax.utils.astrofunc import gravity_jupiter
from exojax.spec.specop import SopInstProfile
import jax.numpy as jnp

if __name__ == "__main__":

    from jax import config

    config.update("jax_enable_x64", True)

    # obs grid
    Nx = 1500
    nusd, wavd, res = wavenumber_grid(16370.0, 16550.0, Nx, unit="AA", xsmode="modit")

    Nx = 7500
    nu_grid, wav, res = wavenumber_grid(
        np.min(wavd) - 10.0, np.max(wavd) + 10.0, Nx, unit="AA", xsmode="premodit"
    )

    T_fid = 500.0
    Tlow = 490.0
    Thigh = 510.0

    art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0e1, nlayer=100)
    art.change_temperature_range(Tlow, Thigh)

    Rinst = 100000.0
    beta_inst = resolution_to_gaussian_std(Rinst)

    ### CH4 setting (PREMODIT)
    mdb = MdbExomol(
        ".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False
    )
    
    print("N=", len(mdb.nu_lines))
    diffmode = 0
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=0.2,
        allow_32bit=True,
    )

    ## CIA setting
    # cdbH2H2 = CdbCIA(".database/H2-H2_2011.cia", nu_grid)
    mu_fid = 2.2
    # settings before HMC
    radius_btm = RJ
    gravity_btm = gravity_jupiter(1.0, 1.0)

    vrmax = 100.0  # km/s
    sop = SopInstProfile(nu_grid, vrmax)

    # raw spectrum model given T0
    def flux_model(mmr_ch4, rv):
        # T-P model
        Tarr = T_fid * np.ones_like(art.pressure)
        mmw_arr = mu_fid * np.ones_like(art.pressure)

        gravity = art.gravity_profile(Tarr, mmw_arr, radius_btm, gravity_btm)
        mmr_arr = art.constant_mmr_profile(mmr_ch4)

        # molecule
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

        Rp2 = art.run(dtau, Tarr, mmw_arr, radius_btm, gravity_btm)
        mu = sop.sampling(Rp2, rv, nusd)

        return jnp.sqrt(mu) * radius_btm / Rs

    # test and save
    mmw_ch4_fid = 0.007
    rv_fid = 10.0
    Rp_over_Rs = flux_model(mmw_ch4_fid, rv_fid)

    import matplotlib.pyplot as plt

    plt.plot(nusd, Rp_over_Rs)
    plt.savefig("sample_trans.png")
    np.savetxt(SAMPLE_SPECTRA_CH4_TRANS, np.array([nusd, Rp_over_Rs]).T, delimiter=",")
