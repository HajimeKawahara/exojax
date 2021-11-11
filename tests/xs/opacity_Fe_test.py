"""test for opacity of metal lines

  - This test compares Fe opacity with petitRADTRANS
   
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

path_pRT = '/home/tako/work/pRT/'
path_VALD3 = '/home/tako/work/VALD3/'
path_fig = '/home/tako/Dropbox/tmpfig/tmp_211031/'
outdir = path_pRT + 'input_data/opacities/lines/line_by_line/Fe_exojax/'

#-------
out_suffix = '_pytest'
H_He_HH_VMR = [0.0, 0.16, 0.84] #H, He, H2 #pure[1.0, 0.0, 0.0] #test[0.05, 0.005, 0.1] #Solar[0.0, 0.16, 0.84]

nus = 1e8/np.arange(13000, 12000, -0.01, dtype=np.float64) #wavenumber range for opacity calculation (Covering whole wavelength ranges of both IRD and CARMENES)
nus4LL = 1e8/np.arange(1e5, 1500.0, -0.01, dtype=np.float64) #wavenumber range for LineList being taken into account (Taking all (except for 1e5–1e6) lines in the line lists (VALD3, Kurucz) into consideration)
pf_Irwin = False #if True, the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016


#Read line list
#$ cp [user_name_at_VALD].[request_number_at_VALD].gz vald2600.gz
adbFe = moldb.AdbVald(path_VALD3+'vald2600.gz', nus4LL, Irwin=pf_Irwin)

Amol=np.float64( adbFe.atomicmass[0] ) #atomic mass [u]
ionE=np.float64( adbFe.ionE[0] ) #ionization energy [eV]
nu0=adbFe.nu_lines

#Make files of wavelength [cm]
np.array(1.0/nus[::-1], dtype=np.float64).tofile(outdir+"wlen"+out_suffix+".dat") #wavelength
#-------

@pytest.mark.parametrize("T", [666, 900, 1215, 1641, 2000, 2217, 2500, 2750, 2995, 3250, 3500, 3750, 4000]) #81, 110, 148, 200, 270, 365, 493,
@pytest.mark.parametrize("P", [0.000001, 0.000010, 0.000100, 0.001000, 0.010000, 0.100000, 1.000000, 10.000000, 100.000000]) #, 1000.000000

def test_opacity_Fe(T, P):
    PH = P* H_He_HH_VMR[0]
    PHe = P* H_He_HH_VMR[1]
    PHH = P* H_He_HH_VMR[2]


    #Make files of opacity [cm^2]
    qt = np.ones_like(adbFe.A) * np.float32(adbFe.qr_interp("Fe 1", T, Irwin=pf_Irwin))
    #↑Unlike the case of HITRAN (using Qr_HAPI), we ignored the isotopes.
    Sij = SijT(T, adbFe.logsij0, adbFe.nu_lines, adbFe.elower, qt)
    sigmaD = doppler_sigma(adbFe.nu_lines, T, Amol)
    gammaL = atomll.gamma_vald3(T, PH, PHH, PHe, adbFe.ielem, adbFe.iion, \
            adbFe.dev_nu_lines, adbFe.elower, adbFe.eupper, adbFe.atomicmass, adbFe.ionE, \
            adbFe.gamRad, adbFe.gamSta, adbFe.vdWdamp, enh_damp=1.0)
    xsv = xsection(nus, nu0, sigmaD, gammaL, Sij, memory_size=30) #←Bottleneck
    out = "sigma_99_"+("{:.0f}".format(T))+".K_"+("{:.6f}".format(P))+"bar"+out_suffix+".dat"
    op = np.array(xsv[::-1],dtype=np.float64)/(Amol*m_u)
    op.tofile(outdir+out)


    #Read spectra of exojax
    petit_exojaxdir = path_pRT+"input_data/opacities/lines/line_by_line/Fe_exojax/"
    species_mass = Amol

    with open(petit_exojaxdir+"wlen"+out_suffix+".dat", 'rb') as w:
        wav_exo=np.array(  np.fromfile(w, dtype=np.float64)  , dtype=np.float64) #[cm]

    fn_exo = "sigma_99_"+("{:.0f}".format(T))+".K_"+("{:.6f}".format(P))+"bar"+out_suffix+".dat"
    with open(petit_exojaxdir+fn_exo, 'rb') as f:
        xs_exo = np.array(  np.fromfile(f, dtype=np.float64)  )*species_mass*m_u
        

    #Read spectra of petitRADTRANS
    petitdir = path_pRT+"input_data/opacities/lines/line_by_line/Fe/"
    species_mass = Amol

    with open (petitdir+"wlen.dat") as w:
        wav_prt=np.array(  np.fromfile(w, dtype=np.float64)  , dtype=np.float64) #[cm]
        
    fn_prt = "sigma_99_"+("{:.0f}".format(T))+".K_"+("{:.6f}".format(P))+"bar.dat"
    with open(petitdir+fn_prt, 'rb') as f:
        xs_prt = np.array(  np.fromfile(f, dtype=np.float64)  )*species_mass*m_u
        

    #Trim and Interpolate spectra (as preparation for taking residuals)
    import scipy as sp
    from matplotlib import gridspec

    x1, y1 = wav_exo*1.e8,  xs_exo
    x2, y2 = wav_prt*1.e8,  xs_prt

    x1t_min = max(min(x1), min(x2))
    x1t_max = min(max(x1), max(x2))
    x1t = x1[np.where(x1-x1t_min >= 0)[0][0]+10 : np.where(x1-x1t_max <= 0)[0][-1]-10]
    y1t = y1[np.where(x1-x1t_min >= 0)[0][0]+10 : np.where(x1-x1t_max <= 0)[0][-1]-10]
    x2t = x2[np.where(x2-x1t_min >= 0)[0][0] : np.where(x2-x1t_max <= 0)[0][-1]]
    y2t = y2[np.where(x2-x1t_min >= 0)[0][0] : np.where(x2-x1t_max <= 0)[0][-1]]

    y2t_interp = sp.interpolate.interp1d(x2t, y2t, kind="quadratic")(x1t)


    #Plot and Examine normalized residuals
    fig = plt.figure(figsize=(9, 5))
    gs = gridspec.GridSpec(5,1)
    ax1 = plt.subplot(gs[0:3])
    ax2 = plt.subplot(gs[3:], sharex=ax1)

    idparam = "_"+("{:.0f}".format(T))+".K_"+("{:.6f}".format(P))+"bar"

    ax1.plot(x1t, y1t, '-', lw=1, label='exojax' + idparam)
    ax1.plot(x1t, y2t_interp, ':', lw=1, label='petitRADTRANS') #, alpha=.7) #, color='mediumseagreen'
    ax1.set_yscale("log")

    #---------------------------------------------------------------------------------
    #case(1)
    #ax2.plot(x1t, (y1t-y2t_interp), '--', lw=1, color='k', label='residual')

    #case(2)
    #ax2.plot(x1t, (y1t/y2t_interp), '--', lw=1, color='k', label='ratio')
    #ax2.set_yscale("log")

    #case(3)
    ax2.plot(x1t, abs(y1t-y2t_interp)/y1t, '--', lw=1, color='k', label='abs(diff)/data')
    ax2.set_yscale("log")
    #---------------------------------------------------------------------------------

    ratio = y1t/y2t_interp
    diffmax = np.max(abs(y1t-y2t_interp))
    normalized_residual = abs(y1t-y2t_interp)/y1t
    ax1.set_title('mean(ratio) = ' + str("{:.1f}".format(np.mean(ratio))) + \
                ', std(ratio) = ' + str("{:.1f}".format(np.std(ratio))) + \
                ', median(abd(diff)/data) = '+str("{:.2e}".format(np.median(normalized_residual))) + \
                ', std(abd(diff)/data) = '+str("{:.2e}".format(np.std(normalized_residual))),  fontsize=9)
                #', max(abd(diff)) = '+str("{:.1e}".format(diffmax)) + \
                #', max(diff/data) = '+str("{:.1e}".format(np.max(normalized_residual))))
    ax1.axes.xaxis.set_visible(False)
    ax2.set_xlabel("wavelength ($\AA$)")
    plt.subplots_adjust(hspace=.05)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.show()
    plt.savefig(path_fig+'comp_diff_Fe_pRT'+idparam+'.pdf')#_ratio
    print('comp_diff_Fe_pRT'+idparam+'.pdf')


    assert np.median(normalized_residual) < 0.1
    assert np.std(normalized_residual) < 50. #5.

if __name__ == "__main__":
    test_opacity_Fe()
