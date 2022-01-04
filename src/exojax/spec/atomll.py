import numpy as np
from exojax.spec import atomllapi
from exojax.utils.constants import ccgs, m_u, kB, hcperk, ecgs, hcgs, Rcgs, a0, eV2wn
import jax.numpy as jnp

def Sij0(A, gupper, nu_lines, elower, QTref_284, QTmask, Irwin=False):
    """Reference Line Strength in Tref=296K, S0.

    Note:
       Tref=296K

    Args:
       A: Einstein coefficient (s-1)
       gupper: the upper state statistical weight
       nu_lines: line center wavenumber (cm-1)
       elower: elower
       QTref_284: partition function Q(Tref)
       QTmask: mask to identify a rows of QTref_284 to apply for each line
       Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016

    Returns:
       Sij(T): Line strength (cm)

    """
    Tref=296.0

    #Assign Q(Tref) for each line
    QTref = np.zeros_like(QTmask, dtype=float)
    for i, mask in enumerate(QTmask):
        QTref[i] = QTref_284[mask]
        
    #Use Irwin_1981 for Fe I (mask==76)  #test211013Tako
    if Irwin==True:
        QTref[jnp.where(QTmask == 76)[0]] = atomllapi.partfn_Fe(Tref)

    S0 = -A*gupper*np.exp(-hcperk*elower/Tref)*np.expm1(-hcperk*nu_lines/Tref)\
        /(8.0*np.pi*ccgs*nu_lines**2*QTref)

    return(S0)



def gamma_vald3(T, PH, PHH, PHe, ielem, iion, \
    nu_lines, elower, eupper, atomicmass, ionE, \
    gamRad, gamSta, vdWdamp, enh_damp=1.0): #, vdW_meth="V"):
    """HWHM of Lorentzian (cm-1) caluculated as gamma/(4*pi*c) [cm-1] for lines with the van der Waals gamma in the line list (VALD or Kurucz), otherwise estimated according to the Unsoeld (1955)

    Args:
      T: temperature (K)
      PH: hydrogen pressure (bar) #1 bar = 1e6 dyn/cm2
      PHH: H2 molecule pressure (bar)
      PHe: helium pressure (bar)
      ielem:  atomic number (e.g., Fe=26)
      iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      gamRad: log of gamma of radiation damping (s-1) (https://www.astro.uu.se/valdwiki/Vald3Format)
      gamSta: log of gamma of Stark damping (s-1)
      vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant cf.) This coefficient (enh_damp) depends on  each species in some codes such as Turbospectrum. #tako210917
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line (computed)
      C6: interaction constant (Eq.11.17 in Gray2005) (computed)
      logg6: log(gamma6) (Eq.11.29 in Gray2005) (computed)
      gam6H: 17*v**(0.6)*C6**(0.4)*N (computed) (v:relative velocity, N:number density of neutral perturber)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4)  (computed)

    Returns:
      gamma: pressure gamma factor (cm-1)

    Note:
       "/(4*np.pi*ccgs)" means:  damping constant -> HWHM of Lorentzian in [cm^-1]    


    * Reference of van der Waals damping constant (pressure/collision gamma):
    *   Unsöld1955: https://ui.adsabs.harvard.edu/abs/1955psmb.book.....U
    *   Kurucz+1981: https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K
    *   Barklem+1998: https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..863B
    *   Barklem+2000: https://ui.adsabs.harvard.edu/abs/2000A&AS..142..467B
    *   Gray+2005: https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G

    """
    gamRad = jnp.where(gamRad==0., -99, gamRad)
    gamSta = jnp.where(gamSta==0., -99, gamSta)
    chi_lam = nu_lines/eV2wn #[cm-1] -> [eV]
    chi = elower/eV2wn #[cm-1] -> [eV]

    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2)) #possibly with "ION**2" factor?
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)
    #Avoid nan (appeared by np.log10(negative C6))
    gamma_case1 = jnp.where(jnp.isnan(gamma_case1), 0., gamma_case1)

    Texp = 0.38 #Barklem+2000
    gam6H = 10**vdWdamp * (T/10000.)**Texp * PH*1e6 /(kB*T)
    gam6He = 10**vdWdamp * (T/10000.)**Texp * PHe*1e6*0.41336 /(kB*T)
    gam6HH = 10**vdWdamp * (T/10000.)**Texp * PHH*1e6*0.85 /(kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case2 = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)
    #Adopt case2 for lines with vdW in VALD, otherwise Case1
    
    gamma = (gamma_case1 * jnp.where(vdWdamp>=0., 1, 0) + gamma_case2 * jnp.where(vdWdamp<0., 1, 0))

    
    return(gamma)



def gamma_uns(T, PH, PHH, PHe, ielem, iion, \
    nu_lines, elower, eupper, atomicmass, ionE, \
    gamRad, gamSta, vdWdamp, enh_damp=1.0): #, vdW_meth="U"):
    """HWHM of Lorentzian (cm-1) estimated with the classical approximation by Unsoeld (1955)

    Args:
      T: temperature (K)
      PH: hydrogen pressure (bar)  #1 bar = 1e6 dyn/cm2
      PHH: H2 molecule pressure (bar)
      PHe: helium pressure (bar)
      ielem:  atomic number (e.g., Fe=26)
      iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
      gamSta: log of gamma of Stark damping (s-1)
      vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant cf.) This coefficient (enh_damp) depends on  each species in some codes such as Turbospectrum. #tako210917
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line (computed)
      C6: interaction constant (Eq.11.17 in Gray2005) (computed)
      logg6: log(gamma6) (Eq.11.29 in Gray2005) (computed)
      gam6H: 17*v**(0.6)*C6**(0.4)*N (v:relative velocity, N:number density of neutral perturber) (computed)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4)(computed)

    Returns:
      gamma: pressure gamma factor (cm-1)

    Note:
       "/(4*np.pi*ccgs)" means:  damping constant -> HWHM of Lorentzian in [cm^-1]
    
    * Reference of van der Waals damping constant (pressure/collision gamma):
    *  Unsöld1955: https://ui.adsabs.harvard.edu/abs/1955psmb.book.....U
    *  Kurucz+1981: https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K
    *  Barklem+1998: https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..863B
    *  Barklem+2000: https://ui.adsabs.harvard.edu/abs/2000A&AS..142..467B
    *  Gray+2005: https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G

    """
    gamRad = jnp.where(gamRad==0., -99, gamRad)
    gamSta = jnp.where(gamSta==0., -99, gamSta)
    chi_lam = nu_lines/eV2wn #[cm-1] -> [eV]
    chi = elower/eV2wn #[cm-1] -> [eV]
    
    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2)) #possibly with "ION**2" factor?
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)
    #Avoid nan (appeared by np.log10(negative C6))
    gamma = jnp.where(jnp.isnan(gamma_case1), 0., gamma_case1)

    return(gamma)



def gamma_KA3(T, PH, PHH, PHe, ielem, iion, \
    nu_lines, elower, eupper, atomicmass, ionE, \
    gamRad, gamSta, vdWdamp, enh_damp=1.0): #, vdW_meth="KA3"):
    """HWHM of Lorentzian (cm-1) caluculated with the 3rd equation in p.4 of Kurucz&Avrett1981

    Args:
      T: temperature (K)
      PH: hydrogen pressure (bar)  #1 bar = 1e6 dyn/cm2
      PHH: H2 molecule pressure (bar)
      PHe: helium pressure (bar)
      ielem:  atomic number (e.g., Fe=26)
      iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
      gamSta: log of gamma of Stark damping (s-1)
      vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant cf.) This coefficient (enh_damp) depends on  each species in some codes such as Turbospectrum. #tako210917
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line (computed)
      C6: interaction constant (Eq.11.17 in Gray2005) (computed)
      logg6: log(gamma6) (Eq.11.29 in Gray2005) (computed)
      gam6H: 17*v**(0.6)*C6**(0.4)*N (v:relative velocity, N:number density of neutral perturber) (computed)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4) (computed)

    Returns:
      gamma: pressure gamma factor (cm-1)

    Note:
      "/(4*np.pi*ccgs)" means:  damping constant -> HWHM of Lorentzian in [cm^-1]
    
    * Reference of van der Waals damping constant (pressure/collision gamma):
    *  Kurucz+1981: https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K
    *  Barklem+1998: https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..863B
    *  Barklem+2000: https://ui.adsabs.harvard.edu/abs/2000A&AS..142..467B
    *  Gray+2005: https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G

    """
    gamRad = jnp.where(gamRad==0., -99, gamRad)
    gamSta = jnp.where(gamSta==0., -99, gamSta)
    Zeff = iion #effective charge (=1 for Fe I, 2 for Fe II, etc.)
    
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper) #Square of effective quantum number of the upper state
    n_eff2_lower = Rcgs * Zeff**2 / (ionE*eV2wn - elower)
    #Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    msr_upper_iron = (45-ielem)/Zeff #for iron group elements (5th equation in Kurucz&Avrett1981)
    msr_upper_noiron = jnp.where(n_eff2_upper>0., (2.5 * (n_eff2_upper/Zeff)**2), 25) #for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper = jnp.where((ielem >= 26)  & (ielem <= 28), msr_upper_iron, msr_upper_noiron)
    msr_lower = 2.5 * (n_eff2_lower/Zeff)**2
    
    gap_msr = msr_upper - msr_lower
    gap_msr_rev = gap_msr * jnp.where(gap_msr < 0, -1., 1.) #Reverse upper and lower if necessary (TBC) #test2109\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev #[Bohr radius -> cm]
    gam6H = 17 * (8*kB*T*(1./atomicmass+1./1.)/(np.pi*m_u))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 /(kB*T)
    gam6He = 17 * (8*kB*T*(1./atomicmass+1./4.)/(np.pi*m_u))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 /(kB*T)
    gam6HH = 17 * (8*kB*T*(1./atomicmass+1./2.)/(np.pi*m_u))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 /(kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)
    
    return(gamma)



def gamma_KA4(T, PH, PHH, PHe, ielem, iion, \
    nu_lines, elower, eupper, atomicmass, ionE, \
    gamRad, gamSta, vdWdamp, enh_damp=1.0): #, vdW_meth="KA4"):
    """HWHM of Lorentzian (cm-1) caluculated with the 4rd equation in p.4 of Kurucz&Avrett1981
    
    Args:
      T: temperature (K)
      PH: hydrogen pressure (bar)  #1 bar = 1e6 dyn/cm2
      PHH: H2 molecule pressure (bar)
      PHe: helium pressure (bar)
      ielem:  atomic number (e.g., Fe=26)
      iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
      gamSta: log of gamma of Stark damping (s-1)
      vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant
          #cf.) This coefficient (enh_damp) depends on  each species in some codes such as Turbospectrum. #tako210917
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line (computed)
      C6: interaction constant (Eq.11.17 in Gray2005) (computed)
      logg6: log(gamma6) (Eq.11.29 in Gray2005) (computed)
      gam6H: 17*v**(0.6)*C6**(0.4)*N (v:relative velocity, N:number density of neutral perturber) (computed)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4) (computed)

    Returns:
      gamma: pressure gamma factor (cm-1)

    Note:
       Approximation of case4 assume "that the atomic weight A is much greater than 4, and that the mean-square-radius of the lower level <r^2>_lo is small compared to <r^2>_up".
    "/(4*np.pi*ccgs)" means:  damping constant -> HWHM of Lorentzian in [cm^-1]

    * Reference of van der Waals damping constant (pressure/collision gamma):
    * Kurucz+1981: https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K
    * Barklem+1998: https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..863B
    * Barklem+2000: https://ui.adsabs.harvard.edu/abs/2000A&AS..142..467B
    * Gray+2005: https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G

    """
    gamRad = jnp.where(gamRad==0., -99, gamRad)
    gamSta = jnp.where(gamSta==0., -99, gamSta)
    Zeff = iion #effective charge (=1 for Fe I, 2 for Fe II, etc.)
    
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper) #Square of effective quantum number of the upper state
    #Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    msr_upper_iron = (45-ielem)/Zeff #for iron group elements (5th equation in Kurucz&Avrett1981)
    msr_upper_noiron = jnp.where(n_eff2_upper>0., (2.5 * (n_eff2_upper/Zeff)**2), 25) #for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper = jnp.where((ielem >= 26)  & (ielem <= 28), msr_upper_iron, msr_upper_noiron)
                    
    gamma6 = 4.5e-9 * msr_upper**0.4 \
        * ((PH + 0.42*PHe + 0.85*PHH)*1e6/(kB*T)) * (T/10000.)**0.3
    gamma = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)

    return(gamma)



def gamma_KA3s(T, PH, PHH, PHe, ielem, iion, \
    nu_lines, elower, eupper, atomicmass, ionE, \
    gamRad, gamSta, vdWdamp, enh_damp=1.0): #, vdW_meth="KA3s"): (supplemetary)
    """(supplemetary:) HWHM of Lorentzian (cm-1) caluculated with the 3rd equation in p.4 of Kurucz&Avrett1981 but without discriminating iron group elements

    Args:
      T: temperature (K)
      PH: hydrogen pressure (bar)  #1 bar = 1e6 dyn/cm2
      PHH: H2 molecule pressure (bar)
      PHe: helium pressure (bar)
      ielem:  atomic number (e.g., Fe=26)
      iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
      gamSta: log of gamma of Stark damping (s-1)
      vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant cf.) This coefficient (enh_damp) depends on  each species in some codes such as Turbospectrum. #tako210917
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line (computed)
      C6: interaction constant (Eq.11.17 in Gray2005) (computed)
      logg6: log(gamma6) (Eq.11.29 in Gray2005) (computed)
      gam6H: 17*v**(0.6)*C6**(0.4)*N (v:relative velocity, N:number density of neutral perturber) (computed)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4)(computed)

    Returns:
      gamma: pressure gamma factor (cm-1)

    Note:
      "/(4*np.pi*ccgs)" means:  damping constant -> HWHM of Lorentzian in [cm^-1]
    
    * Reference of van der Waals damping constant (pressure/collision gamma):
    *  Kurucz+1981: https://ui.adsabs.harvard.edu/abs/1981SAOSR.391.....K
    *  Barklem+1998: https://ui.adsabs.harvard.edu/abs/1998MNRAS.300..863B
    *  Barklem+2000: https://ui.adsabs.harvard.edu/abs/2000A&AS..142..467B
    *  Gray+2005: https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G

    """
    gamRad = jnp.where(gamRad==0., -99, gamRad)
    gamSta = jnp.where(gamSta==0., -99, gamSta)
    Zeff = iion #effective charge (=1 for Fe I, 2 for Fe II, etc.)
    
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper) #Square of effective quantum number of the upper state
    n_eff2_lower = Rcgs * Zeff**2 / (ionE*eV2wn - elower)
    #Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    msr_upper_noiron = jnp.where(n_eff2_upper>0., (2.5 * (n_eff2_upper/Zeff)**2), 25) #for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper = msr_upper_noiron
    msr_lower = 2.5 * (n_eff2_lower/Zeff)**2

    gap_msr = msr_upper - msr_lower
    gap_msr_rev = gap_msr * jnp.where(gap_msr < 0, -1., 1.) #Reverse upper and lower if necessary (TBC) #test2109\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev #[Bohr radius -> cm]
    gam6H = 17 * (8*kB*T*(1./atomicmass+1./1.)/(np.pi*m_u))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 /(kB*T)
    gam6He = 17 * (8*kB*T*(1./atomicmass+1./4.)/(np.pi*m_u))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 /(kB*T)
    gam6HH = 17 * (8*kB*T*(1./atomicmass+1./2.)/(np.pi*m_u))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 /(kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma = (gamma6 + 10**gamRad + 10**gamSta) /(4*np.pi*ccgs)

    return(gamma)
