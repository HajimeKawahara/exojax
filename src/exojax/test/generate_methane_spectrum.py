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
from exojax.spec.response import ipgauss_sampling
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.utils.grids import velocity_grid
from exojax.utils.astrofunc import gravity_jupiter

from exojax.spec import molinfo
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW

if __name__ == "__main__":
    #given gravity, temperature exponent, MMR
    g = gravity_jupiter(0.88, 33.2)
    alpha = 0.1
    MMR_CH4 = 0.0059
    vsini = 20.0
    RV = 10.0
    T0 = 1200.0
    
    #obs grid
    Nx = 1500
    nusd, wavd, res = wavenumber_grid(16370.,
                                      16550.,
                                      Nx,
                                      unit="AA",
                                      xsmode="modit")
    
    Nx = 7500
    nu_grid, wav, res = wavenumber_grid(np.min(wavd) - 10.0,
                                        np.max(wavd) + 10.0,
                                        Nx,
                                        unit='AA',
                                        xsmode='premodit')
    
    Tlow = 400.0
    Thigh = 1500.0
    art = ArtTransPure(nu_grid, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)
    art.change_temperature_range(Tlow, Thigh)
    Mp = 33.2
    
    Rinst = 100000.
    beta_inst = resolution_to_gaussian_std(Rinst)
    
    ### CH4 setting (PREMODIT)
    mdb = MdbExomol('.database/CH4/12C-1H4/YT10to10/',
                    nurange=nu_grid,
                    gpu_transfer=False)
    print('N=', len(mdb.nu_lines))
    diffmode = 1
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[Tlow, Thigh],
                      dit_grid_resolution=0.2)
    
    ## CIA setting
    cdbH2H2 = CdbCIA('.database/H2-H2_2011.cia', nu_grid)
    opcia = OpaCIA(cdb=cdbH2H2, nu_grid=nu_grid)
    mmw = 2.33  # mean molecular weight
    mmrH2 = 0.74
    molmassH2 = molinfo.molmass_isotope('H2')
    vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR
    
    gravity_btm = 2478.57
    radius_btm = RJ
    
    
    #settings before HMC
    vsini_max = 100.0
    vr_array = velocity_grid(res, vsini_max)
    
    
    # raw spectrum model given T0
    def flux_model(T0, vsini, RV):
        #T-P model
        Tarr = art.powerlaw_temperature(T0, alpha)
        gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)
        
        #molecule
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        mmr_arr = art.constant_mmr_profile(MMR_CH4)
        dtaumCH4 = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass, g)
        
        #continuum
        logacia_matrix = opcia.logacia_matrix(Tarr)
        dtaucH2H2 = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2,
                                            mmw, g)
        
        dtau = dtaumCH4 + dtaucH2H2
        F0 = art.run(dtau, Tarr)
        Frot = convolve_rigid_rotation(F0, vr_array, vsini=vsini, u1=0.0, u2=0.0)
        mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV, vr_array)
        
        return mu
    
    
#test and save
#mu = flux_model(T0, vsini, RV)
#import matplotlib.pyplot as plt
#plt.plot(nusd, mu)
#plt.show()
#np.savetxt(SAMPLE_SPECTRA_CH4_NEW, np.array([nusd, mu]).T, delimiter=",")
