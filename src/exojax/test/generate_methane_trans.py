""" Reverse modeling of Methane emission spectrum using PreMODIT, precomputation of F0 grids
"""

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from exojax.utils.grids import wavenumber_grid
from exojax.spec.atmrt import ArtTransPure
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.contdb import CdbCIA
from exojax.spec.opacont import OpaCIA
from exojax.spec.response import ipgauss, sampling
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.spec import molinfo
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_TRANS
from exojax.utils.constants import RJ, MJ
from exojax.utils.astrofunc import gravity_jupiter

if __name__ == "__main__":
    # given gravity, temperature exponent, MMR
    alpha = 0.1
    MMR_CH4 = 0.0059
    vsini = 20.0
    RV = 10.0
    T0 = 1200.0

    # obs grid
    Nx = 1500
    nusd, wavd, res = wavenumber_grid(16370.0, 16550.0, Nx, unit="AA", xsmode="modit")

    Nx = 7500
    nu_grid, wav, res = wavenumber_grid(
        np.min(wavd) - 10.0, np.max(wavd) + 10.0, Nx, unit="AA", xsmode="premodit"
    )

    Tlow = 400.0
    Thigh = 1500.0
    art = ArtTransPure(pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100)
    art.change_temperature_range(Tlow, Thigh)
    Mp = 33.2

    Rinst = 100000.0
    beta_inst = resolution_to_gaussian_std(Rinst)

    ### CH4 setting (PREMODIT)
    mdb = MdbExomol(
        ".database/CH4/12C-1H4/YT10to10/", nurange=nu_grid, gpu_transfer=False
    )
    print("N=", len(mdb.nu_lines))
    diffmode = 1
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=0.2,
        allow_32bit=True,
    )

    ## CIA setting
    cdbH2H2 = CdbCIA(".database/H2-H2_2011.cia", nu_grid)
    opcia = OpaCIA(cdb=cdbH2H2, nu_grid=nu_grid)
    mmw = 2.33  # mean molecular weight
    mmrH2 = 0.74
    molmassH2 = molinfo.molmass_isotope("H2")
    vmrH2 = mmrH2 * mmw / molmassH2  # VMR

    # settings before HMC
    vsini_max = 100.0
    vr_array = velocity_grid(res, vsini_max)
    radius_btm = RJ
    gravity_btm = gravity_jupiter(Rp=RJ, Mp=MJ)

    # raw spectrum model given T0
    def flux_model(T0, vsini, RV):
        # T-P model
        Tarr = art.powerlaw_temperature(T0, alpha)
        mmw_arr = mmw * np.ones_like(art.pressure)

        gravity = art.gravity_profile(Tarr, mmw_arr, radius_btm, gravity_btm)

        mmr_arr = art.constant_mmr_profile(MMR_CH4)
        # molecule
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        dtaumCH4 = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)

        # continuum
        logacia_matrix = opcia.logacia_matrix(Tarr)
        dtaucH2H2 = art.opacity_profile_cia(
            logacia_matrix, Tarr, vmrH2, vmrH2, mmw_arr, gravity
        )

        dtau = dtaumCH4 + dtaucH2H2
        Rp2 = art.run(dtau, Tarr, mmw_arr, radius_btm, gravity_btm)
        mu = sampling(nusd, nu_grid, RV=RV)

        return mu

    # test and save
    mu = flux_model(T0, vsini, RV)

    import matplotlib.pyplot as plt

    plt.plot(nusd, mu)
    plt.savefig("sample_trans.png")
    # plt.show()
    np.savetxt(SAMPLE_SPECTRA_CH4_TRANS, np.array([nusd, mu]).T, delimiter=",")
