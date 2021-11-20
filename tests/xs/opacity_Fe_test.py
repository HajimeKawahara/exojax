"""test for opacity calculation of metal lines

  - This test calculates Fe opacity from VALD3 line list. (Comparison with petitRADTRANS opacity is shown in examples/comparisons/opacity_Fe_VALD3.ipynb)
    The calculation of gamma is based on the van der Waals gamma in the line list (VALD or Kurucz), otherwise estimated according to the Unsoeld (1955)
    
  Note: The input line list needs to be obtained from VALD3 (http://vald.astro.uu.se/). VALD data access is free but requires registration through the Contact form (http://vald.astro.uu.se/~vald/php/vald.php?docpage=contact.html). After the registration, you can login and choose the "Extract Element" mode.
        For this test, the request form should be filled as:
            Starting wavelength :    1500
            Ending wavelength :    100000
            Element [ + ionization ] :    Fe 1
            Extraction format :    Long format
            Retrieve data via :    FTP
            Linelist configuration :    Default
            Unit selection:    Energy unit: eV - Medium: vacuum - Wavelength unit: angstrom - VdW syntax: default
        Please rename the file sent by VALD ([user_name_at_VALD].[request_number_at_VALD].gz) to "vald2600.gz" if you would like to use the code below without editing it.
   
"""

import pytest
import numpy as np
from exojax.spec import xsection, moldb, atomll
from exojax.spec.hitran import SijT, doppler_sigma
import matplotlib.pyplot as plt
from exojax.utils.constants import m_u

filepath_VALD3 = '.database/vald2600.gz'
path_fig = './'

#-------

out_suffix = '_pytest'
H_He_HH_VMR = [0.0, 0.16, 0.84] #H, He, H2 #pure[1.0, 0.0, 0.0] #test[0.05, 0.005, 0.1] #Solar[0.0, 0.16, 0.84]

nus = 1e8/np.arange(12200, 11800, -0.01, dtype=np.float64) #wavenumber range for opacity calculation (Covering whole wavelength ranges of both IRD and CARMENES)
nus4LL = 1e8/np.arange(1e5, 1500.0, -0.01, dtype=np.float64) #wavenumber range for LineList being taken into account (Taking all (except for 1e5–1e6) lines in the line lists (VALD3, Kurucz) into consideration)
pf_Irwin = False #if True, the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016


#Read line list
#$ cp [user_name_at_VALD].[request_number_at_VALD].gz vald2600.gz
adbFe = moldb.AdbVald(filepath_VALD3, nus4LL, Irwin=pf_Irwin)

Amol=np.float64( adbFe.atomicmass[0] ) #atomic mass [u]
ionE=np.float64( adbFe.ionE[0] ) #ionization energy [eV]
nu0=adbFe.nu_lines

#-------

@pytest.mark.parametrize("T", [2995,]) #[81, 110, 148, 200, 270, 365, 493, 666, 900, 1215, 1641, 2000, 2217, 2500, 2750, 2995, 3250, 3500, 3750, 4000]
@pytest.mark.parametrize("P", [0.100000,]) #[0.000001, 0.000010, 0.000100, 0.001000, 0.010000, 0.100000, 1.000000, 10.000000, 100.000000, 1000.000000]

def test_opacity_Fe(T, P):
    PH = P* H_He_HH_VMR[0]
    PHe = P* H_He_HH_VMR[1]
    PHH = P* H_He_HH_VMR[2]

    qt = np.ones_like(adbFe.A) * np.float32(adbFe.qr_interp("Fe 1", T))
    #↑Unlike the case of HITRAN (using Qr_HAPI), we ignored the isotopes.
    Sij = SijT(T, adbFe.logsij0, adbFe.nu_lines, adbFe.elower, qt)
    sigmaD = doppler_sigma(adbFe.nu_lines, T, Amol)
    
    gammaL_vald3 = atomll.gamma_vald3(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv_vald3 = xsection(nus, nu0, sigmaD, gammaL_vald3, Sij, memory_size=30) #←Bottleneck
    op_vald3 = np.array(xsv_vald3[::-1],dtype=np.float64)/(Amol*m_u)

    gammaL_uns = atomll.gamma_uns(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv_uns = xsection(nus, nu0, sigmaD, gammaL_uns, Sij, memory_size=30) #←Bottleneck
    op_uns = np.array(xsv_uns[::-1],dtype=np.float64)/(Amol*m_u)

    gammaL_KA3 = atomll.gamma_KA3(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv_KA3 = xsection(nus, nu0, sigmaD, gammaL_KA3, Sij, memory_size=30)
    op_KA3 = np.array(xsv_KA3[::-1],dtype=np.float64)/(Amol*m_u)

    gammaL_KA4 = atomll.gamma_KA4(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv_KA4 = xsection(nus, nu0, sigmaD, gammaL_KA4, Sij, memory_size=30)
    op_KA4 = np.array(xsv_KA4[::-1],dtype=np.float64)/(Amol*m_u)

    gammaL_KA3s = atomll.gamma_KA3s(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv_KA3s = xsection(nus, nu0, sigmaD, gammaL_KA3s, Sij, memory_size=30)
    op_KA3s = np.array(xsv_KA3s[::-1],dtype=np.float64)/(Amol*m_u)

    str_param = ("{:.0f}".format(T))+".K_"+("{:.6f}".format(P))+"bar"
    plt.figure()
    plt.plot(1.e8/nus[::-1],  op)
    plt.yscale("log")
    plt.xscale("log")
    plt.title(str_param)
    plt.savefig(path_fig+"opacity_Fe_test_"+str_param+".pdf")
    plt.clf()
    plt.close()

    assert((True in np.isnan(op_vald3)) == False)
    assert((True in np.isnan(op_uns)) == False)
    assert((True in np.isnan(op_KA3)) == False)
    assert((True in np.isnan(op_KA4)) == False)
    assert((True in np.isnan(op_KA3s)) == False)

if __name__ == "__main__":
    test_opacity_Fe()
