"""test for opacity calculation of metal lines (VALD) with MODIT

- This test calculates atomic and ionic opacity from VALD3 line list.
  The calculation of gamma is based on the van der Waals gamma in the line list, otherwise estimated according to the Unsoeld (1955)

Note: The input line list needs to be obtained from VALD3 (http://vald.astro.uu.se/). VALD data access is free but requires registration through the Contact form (http://vald.astro.uu.se/~vald/php/vald.php?docpage=contact.html). After the registration, you can login and choose the "Extract All" mode.
      For this test, the request form should be filled as:
          Starting wavelength :    1500
          Ending wavelength :    100000
          Extraction format :    Long format
          Retrieve data via :    FTP
          Linelist configuration :    Default
          Unit selection:    Energy unit: eV - Medium: vacuum - Wavelength unit: angstrom - VdW syntax: default
      Please rename the file sent by VALD ([user_name_at_VALD].[request_number_at_VALD].gz) to "vald4214450.gz" if you would like to use the code below without editing it.
"""

path_ValdLineList = '.database/vald4214450.gz'

import numpy as np
import jax.numpy as jnp
from exojax.spec import moldb, atomll, contdb, molinfo, initspec, planck, response
from exojax.spec import api
from exojax.spec.rtransfer import pressure_layer, dtauVALD, dtauM_mmwl, dtauHminus_mmwl, dtauCIA_mmwl, rtrun
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.spec.modit import vald_all, xsmatrix_vald, exomol, xsmatrix, setdgm_vald_all, setdgm_exomol

def test_VALD_MODIT():

    #wavelength range
    wls, wll = 10395, 10405

    #Set a model atmospheric layers, wavenumber range for the model, an instrument
    NP = 100
    Parr, dParr, k = pressure_layer(NP = NP)
    Pref=1.0 #bar
    ONEARR=np.ones_like(Parr)

    Nx = 2000
    nus, wav, res = wavenumber_grid(wls - 5.0, wll + 5.0, Nx, unit="AA", xsmode="modit")

    Rinst=100000. #instrumental spectral resolution
    beta_inst=resolution_to_gaussian_std(Rinst)  #equivalent to beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R)

    #atoms and ions from VALD
    adbV = moldb.AdbVald(path_ValdLineList, nus, crit = 1e-100) #The crit is defined just in case some weak lines may cause an error that results in a gamma of 0... (220219)
    asdb = moldb.AdbSepVald(adbV)

    #molecules from exomol
    mdbH2O = api.MdbExomol('.database/H2O/1H2-16O/POKAZATEL', nus, crit = 1e-50, gpu_transfer=True)#,crit = 1e-40)
    mdbTiO = api.MdbExomol('.database/TiO/48Ti-16O/Toto', nus, crit = 1e-50, gpu_transfer=True)#,crit = 1e-50)
    mdbOH = api.MdbExomol('.database/OH/16O-1H/MoLLIST', nus, gpu_transfer=True)
    mdbFeH = api.MdbExomol('.database/FeH/56Fe-1H/MoLLIST', nus, gpu_transfer=True)

    #CIA
    cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nus)

    #molecular mass
    molmassH2O = molinfo.molmass_major_isotope("H2O")
    molmassTiO = molinfo.molmass_major_isotope("TiO")
    molmassOH = molinfo.molmass_major_isotope("OH")
    molmassFeH = molinfo.molmass_major_isotope("FeH")
    molmassH = molinfo.molmass_major_isotope("H")
    molmassH2 = molinfo.molmass_major_isotope("H2")

    #Initialization of MODIT (for separate VALD species, and exomol molecules(e.g., FeH))
    cnuS, indexnuS, R, pmarray = initspec.init_modit_vald(asdb.nu_lines, nus, asdb.N_usp)
    cnu_FeH, indexnu_FeH, R, pmarray = initspec.init_modit(mdbFeH.nu_lines, nus)
    cnu_H2O, indexnu_H2O, R, pmarray = initspec.init_modit(mdbH2O.nu_lines, nus)
    cnu_OH, indexnu_OH, R, pmarray = initspec.init_modit(mdbOH.nu_lines, nus)
    cnu_TiO, indexnu_TiO, R, pmarray = initspec.init_modit(mdbTiO.nu_lines, nus)

    #sampling the max/min of temperature profiles
    fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
    T0_test=np.array([1500.0, 4000.0, 1500.0, 4000.0])
    alpha_test=np.array([0.2,0.2,0.05,0.05])
    res=0.2

    #Assume typical atmosphere
    H_He_HH_VMR_ref = [0.1, 0.15, 0.75]
    PH_ref = Parr* H_He_HH_VMR_ref[0]
    PHe_ref = Parr* H_He_HH_VMR_ref[1]
    PHH_ref = Parr* H_He_HH_VMR_ref[2]

    #Precomputing dgm_ngammaL
    dgm_ngammaL_VALD = setdgm_vald_all(asdb, PH_ref, PHe_ref, PHH_ref, R, fT, res, T0_test, alpha_test)
    dgm_ngammaL_FeH = setdgm_exomol(mdbFeH, fT, Parr, R, molmassFeH, res, T0_test, alpha_test)
    dgm_ngammaL_H2O = setdgm_exomol(mdbH2O, fT, Parr, R, molmassH2O, res, T0_test, alpha_test) 
    dgm_ngammaL_OH = setdgm_exomol(mdbOH, fT, Parr, R, molmassOH, res, T0_test, alpha_test) 
    dgm_ngammaL_TiO = setdgm_exomol(mdbTiO, fT, Parr, R, molmassTiO, res, T0_test, alpha_test) 


    T0 = 3000.
    alpha = 0.07
    Mp=0.155 *1.99e33/1.90e30
    Rp=0.186 *6.96e10/6.99e9
    u1=0.0
    u2=0.0
    RV=0.00
    vsini=2.0

    mmw=2.33*ONEARR #mean molecular weight
    log_e_H = -4.2
    VMR_H = 0.09 
    VMR_H2 = 0.77
    VMR_FeH = 10**-8
    VMR_H2O = 10**-4
    VMR_OH = 10**-4
    VMR_TiO = 10**-8
    A_Fe = 1.5
    A_Ti = 1.2

    adjust_continuum = 0.99

    ga=2478.57730044555*Mp/Rp**2
    Tarr = T0*(Parr/Pref)**alpha
    PH = Parr* VMR_H
    PHe = Parr* (1-VMR_H-VMR_H2)
    PHH = Parr* VMR_H2
    VMR_e = VMR_H*10**log_e_H

    #VMR of atoms and ions (+Abundance modification)
    mods_ID = jnp.array([[26,1], [22,1]])
    mods = jnp.array([A_Fe, A_Ti])
    VMR_uspecies = atomll.get_VMR_uspecies(asdb.uspecies, mods_ID, mods)
    VMR_uspecies = VMR_uspecies[:, None]*ONEARR

    #Compute delta tau

    #Atom & ions (VALD)
    SijMS, ngammaLMS, nsigmaDlS = vald_all(asdb, Tarr, PH, PHe, PHH, R)
    xsmS = xsmatrix_vald(cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS, nus, dgm_ngammaL_VALD)
    dtauatom = dtauVALD(dParr, xsmS, VMR_uspecies, mmw, ga)

    #FeH
    SijM_FeH, ngammaLM_FeH, nsigmaDl_FeH = exomol(mdbFeH, Tarr, Parr, R, molmassFeH)
    xsm_FeH = xsmatrix(cnu_FeH, indexnu_FeH, R, pmarray, nsigmaDl_FeH, ngammaLM_FeH, SijM_FeH, nus, dgm_ngammaL_FeH)
    dtaum_FeH = dtauM_mmwl(dParr, jnp.abs(xsm_FeH), VMR_FeH*ONEARR, mmw, ga)

    #H2O
    SijM_H2O, ngammaLM_H2O, nsigmaDl_H2O = exomol(mdbH2O, Tarr, Parr, R, molmassH2O)
    xsm_H2O = xsmatrix(cnu_H2O, indexnu_H2O, R, pmarray, nsigmaDl_H2O, ngammaLM_H2O, SijM_H2O, nus, dgm_ngammaL_H2O)
    dtaum_H2O = dtauM_mmwl(dParr, jnp.abs(xsm_H2O), VMR_H2O*ONEARR, mmw, ga) 

    #OH
    SijM_OH, ngammaLM_OH, nsigmaDl_OH = exomol(mdbOH, Tarr, Parr, R, molmassOH)
    xsm_OH = xsmatrix(cnu_OH, indexnu_OH, R, pmarray, nsigmaDl_OH, ngammaLM_OH, SijM_OH, nus, dgm_ngammaL_OH)
    dtaum_OH = dtauM_mmwl(dParr, jnp.abs(xsm_OH), VMR_OH*ONEARR, mmw, ga) 

    #TiO
    SijM_TiO, ngammaLM_TiO, nsigmaDl_TiO = exomol(mdbTiO, Tarr, Parr, R, molmassTiO)
    xsm_TiO = xsmatrix(cnu_TiO, indexnu_TiO, R, pmarray, nsigmaDl_TiO, ngammaLM_TiO, SijM_TiO, nus, dgm_ngammaL_TiO)
    dtaum_TiO = dtauM_mmwl(dParr, jnp.abs(xsm_TiO), VMR_TiO*ONEARR, mmw, ga) 

    #Hminus
    dtau_Hm = dtauHminus_mmwl(nus, Tarr, Parr, dParr, VMR_e*ONEARR, VMR_H*ONEARR, mmw, ga)

    #CIA
    dtauc_H2H2 = dtauCIA_mmwl(nus, Tarr, Parr, dParr, VMR_H2*ONEARR, VMR_H2*ONEARR, mmw, ga, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)

    #Summations
    dtau = dtauatom + dtaum_FeH + dtaum_H2O + dtaum_OH + dtaum_TiO + dtau_Hm + dtauc_H2H2

    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)
    Frot = response.rigidrot(nus, F0, vsini, u1, u2)
    wavd = jnp.linspace(wls, wll, 500)
    nusd = jnp.array(1.e8/wavd[::-1])
    mu = response.ipgauss_sampling(nusd, nus, Frot, beta_inst, RV)
    mu = mu/jnp.nanmax(mu)*adjust_continuum

    assert (np.all(~np.isnan(mu)) * \
            np.all(mu != 0) * \
            np.all(abs(mu) != np.inf))
                
if __name__ == "__main__":
    test_VALD_MODIT()
