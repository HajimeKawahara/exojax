import numpy as np
from exojax.spec import atomllapi
from exojax.utils.constants import ccgs, m_u, kB, hcperk, ecgs, hcgs, Rcgs, a0, eV2wn
import jax.numpy as jnp
from jax.lax import scan
import warnings


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
    Tref = 296.0

    # Assign Q(Tref) for each line
    QTref = np.zeros_like(QTmask, dtype=float)
    for i, mask in enumerate(QTmask):
        QTref[i] = QTref_284[mask]

    # Use Irwin_1981 for Fe I (mask==76)  #test211013Tako
    if Irwin == True:
        QTref[jnp.where(QTmask == 76)[0]] = atomllapi.partfn_Fe(Tref)

    S0 = -A*gupper*np.exp(-hcperk*elower/Tref)*np.expm1(-hcperk*nu_lines/Tref)\
        / (8.0*np.pi*ccgs*nu_lines**2*QTref)

    return S0


def gamma_vald3(T, PH, PHH, PHe, ielem, iion,
                nu_lines, elower, eupper, atomicmass, ionE,
                gamRad, gamSta, vdWdamp, enh_damp=1.0):  # , vdW_meth="V"):
    """HWHM of Lorentzian (cm-1) caluculated as gamma/(4*pi*c) [cm-1] for lines
    with the van der Waals gamma in the line list (VALD or Kurucz), otherwise
    estimated according to the Unsoeld (1955)

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
    gamRad = jnp.where(gamRad == 0., -99, gamRad)
    gamSta = jnp.where(gamSta == 0., -99, gamSta)
    chi_lam = nu_lines/eV2wn  # [cm-1] -> [eV]
    chi = elower/eV2wn  # [cm-1] -> [eV]

    # possibly with "ION**2" factor?
    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2))
    C6 = jnp.abs(C6)  # test2202
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)
    # Avoid nan (appeared by np.log10(negative C6))
    # (Note: if statements is NOT compatible with JAX)
    # if len(jnp.where(jnp.isnan(gamma_case1))[0]) > 0:
    #     warnings.warn('nan were generated in gamma_case1 (), so they were replaced by 0.0 \n\t'+'The number of the lines with the nan: '+str(int(len(jnp.where(jnp.isnan(gamma_case1))[0]))))
    gamma_case1 = jnp.where(jnp.isnan(gamma_case1), 0., gamma_case1)

    Texp = 0.38  # Barklem+2000
    gam6H = 10**vdWdamp * (T/10000.)**Texp * PH*1e6 / (kB*T)
    gam6He = 10**vdWdamp * (T/10000.)**Texp * PHe*1e6*0.41336 / (kB*T)
    gam6HH = 10**vdWdamp * (T/10000.)**Texp * PHH*1e6*0.85 / (kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case2 = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)
    # Adopt case2 for lines with vdW in VALD, otherwise Case1

    gamma = (gamma_case1 * jnp.where(vdWdamp >= 0., 1, 0) +
             gamma_case2 * jnp.where(vdWdamp < 0., 1, 0))

    return gamma


def gamma_uns(T, PH, PHH, PHe, ielem, iion,
              nu_lines, elower, eupper, atomicmass, ionE,
              gamRad, gamSta, vdWdamp, enh_damp=1.0):  # , vdW_meth="U"):
    """HWHM of Lorentzian (cm-1) estimated with the classical approximation by
    Unsoeld (1955)

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
    gamRad = jnp.where(gamRad == 0., -99, gamRad)
    gamSta = jnp.where(gamSta == 0., -99, gamSta)
    chi_lam = nu_lines/eV2wn  # [cm-1] -> [eV]
    chi = elower/eV2wn  # [cm-1] -> [eV]

    # possibly with "ION**2" factor?
    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2))
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)
    # Avoid nan (appeared by np.log10(negative C6))
    gamma = jnp.where(jnp.isnan(gamma_case1), 0., gamma_case1)

    return gamma


def gamma_KA3(T, PH, PHH, PHe, ielem, iion,
              nu_lines, elower, eupper, atomicmass, ionE,
              gamRad, gamSta, vdWdamp, enh_damp=1.0):  # , vdW_meth="KA3"):
    """HWHM of Lorentzian (cm-1) caluculated with the 3rd equation in p.4 of
    Kurucz&Avrett1981.

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
    gamRad = jnp.where(gamRad == 0., -99, gamRad)
    gamSta = jnp.where(gamSta == 0., -99, gamSta)
    Zeff = iion  # effective charge (=1 for Fe I, 2 for Fe II, etc.)

    # Square of effective quantum number of the upper state
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper)
    n_eff2_lower = Rcgs * Zeff**2 / (ionE*eV2wn - elower)
    # Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    # for iron group elements (5th equation in Kurucz&Avrett1981)
    msr_upper_iron = (45-ielem)/Zeff
    # for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper_noiron = jnp.where(
        n_eff2_upper > 0., (2.5 * (n_eff2_upper/Zeff)**2), 25)
    msr_upper = jnp.where((ielem >= 26) & (ielem <= 28),
                          msr_upper_iron, msr_upper_noiron)
    msr_lower = 2.5 * (n_eff2_lower/Zeff)**2

    gap_msr = msr_upper - msr_lower
    gap_msr_rev = gap_msr * \
        jnp.where(
            gap_msr < 0, -1., 1.)  # Reverse upper and lower if necessary (TBC) #test2109\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev  # [Bohr radius -> cm]
    gam6H = 17 * (8*kB*T*(1./atomicmass+1./1.)/(np.pi*m_u))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 / (kB*T)
    gam6He = 17 * (8*kB*T*(1./atomicmass+1./4.)/(np.pi*m_u))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 / (kB*T)
    gam6HH = 17 * (8*kB*T*(1./atomicmass+1./2.)/(np.pi*m_u))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 / (kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)

    return gamma


def gamma_KA4(T, PH, PHH, PHe, ielem, iion,
              nu_lines, elower, eupper, atomicmass, ionE,
              gamRad, gamSta, vdWdamp, enh_damp=1.0):  # , vdW_meth="KA4"):
    """HWHM of Lorentzian (cm-1) caluculated with the 4rd equation in p.4 of
    Kurucz&Avrett1981.

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
    gamRad = jnp.where(gamRad == 0., -99, gamRad)
    gamSta = jnp.where(gamSta == 0., -99, gamSta)
    Zeff = iion  # effective charge (=1 for Fe I, 2 for Fe II, etc.)

    # Square of effective quantum number of the upper state
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper)
    # Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    # for iron group elements (5th equation in Kurucz&Avrett1981)
    msr_upper_iron = (45-ielem)/Zeff
    # for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper_noiron = jnp.where(
        n_eff2_upper > 0., (2.5 * (n_eff2_upper/Zeff)**2), 25)
    msr_upper = jnp.where((ielem >= 26) & (ielem <= 28),
                          msr_upper_iron, msr_upper_noiron)

    gamma6 = 4.5e-9 * msr_upper**0.4 \
        * ((PH + 0.42*PHe + 0.85*PHH)*1e6/(kB*T)) * (T/10000.)**0.3
    gamma = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)

    return gamma


def gamma_KA3s(T, PH, PHH, PHe, ielem, iion,
               nu_lines, elower, eupper, atomicmass, ionE,
               gamRad, gamSta, vdWdamp, enh_damp=1.0):  # , vdW_meth="KA3s"): (supplemetary)
    """(supplemetary:) HWHM of Lorentzian (cm-1) caluculated with the 3rd
    equation in p.4 of Kurucz&Avrett1981 but without discriminating iron group
    elements.

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
    gamRad = jnp.where(gamRad == 0., -99, gamRad)
    gamSta = jnp.where(gamSta == 0., -99, gamSta)
    Zeff = iion  # effective charge (=1 for Fe I, 2 for Fe II, etc.)

    # Square of effective quantum number of the upper state
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*eV2wn - eupper)
    n_eff2_lower = Rcgs * Zeff**2 / (ionE*eV2wn - elower)
    # Mean of square of radius (in units of a0, the radius of the first Bohr orbit; p.320 in Aller (1963); https://ui.adsabs.harvard.edu/abs/1963aass.book.....A)
    # for other elements (6th equation in Kurucz&Avrett1981)
    msr_upper_noiron = jnp.where(
        n_eff2_upper > 0., (2.5 * (n_eff2_upper/Zeff)**2), 25)
    msr_upper = msr_upper_noiron
    msr_lower = 2.5 * (n_eff2_lower/Zeff)**2

    gap_msr = msr_upper - msr_lower
    gap_msr_rev = gap_msr * \
        jnp.where(
            gap_msr < 0, -1., 1.)  # Reverse upper and lower if necessary (TBC) #test2109\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev  # [Bohr radius -> cm]
    gam6H = 17 * (8*kB*T*(1./atomicmass+1./1.)/(np.pi*m_u))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 / (kB*T)
    gam6He = 17 * (8*kB*T*(1./atomicmass+1./4.)/(np.pi*m_u))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 / (kB*T)
    gam6HH = 17 * (8*kB*T*(1./atomicmass+1./2.)/(np.pi*m_u))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 / (kB*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma = (gamma6 + 10**gamRad + 10**gamSta) / (4*np.pi*ccgs)

    return gamma


def get_unique_species(adb):
    """Extract a unique list of line contributing species from VALD atomic
    database (adb)

    Args:
       adb: adb instance made by the AdbVald class in moldb.py

    Returns:
       uspecies: unique elements of the combination of ielem and iion (jnp.array with a shape of N_UniqueSpecies x 2(ielem and iion))
    """
    seen = []
    def get_unique_list(seq): return [
        x for x in seq if x not in seen and not seen.append(x)]
    uspecies = jnp.array(get_unique_list(
        jnp.vstack([adb.ielem, adb.iion]).T.tolist()))
    return uspecies


def ielemion_to_FastChemSymbol(ielem, iion):
    """Translate atomic number and ionization level into SpeciesSymbol in
    FastChem.

    Args:
        ielem:  atomic number (int) (e.g., Fe=26)
        iion:  ionized level (int) (e.g., neutral=1, singly)

    Returns:
        SpeciesSymbol in FastChem (str) (cf. https://github.com/exoclime/FastChem/blob/master/input/logK_ext.dat)
    """
    return ((atomllapi.PeriodicTable[ielem] + '1' + '+'*(iion-1)).rstrip('1'))


def get_VMR_uspecies(uspecies, mods_ID, mods):
    """Extract VMR arrays of the species that contribute the opacity
    ("uspecies" made with "get_unique_species")

    Args:
        uspecies: jnp.array of unique list of the species contributing the opacity [N_species x 2(ielem and iion)]
        mods_ID: jnp.array listing the species whose abundances are different from the solar [N_modified_species x 2(ielem and iion)]
        mods: jnp.array of each abundance deviation from the Sun [dex] for each modified species listed in mods_ID [N_modified_species]

    Returns:
        VMR_uspecies: jnp.array of volume mixing ratio [N_species]
    """
    mods_ID_uspecies = jnp.zeros(len(mods_ID), dtype=int)

    def f_miu(i_and_arr, sp):
        i, arr = i_and_arr
        i_and_arr = i + \
            1, jnp.where(((mods_ID[:, 0] == sp[0]) & (
                mods_ID[:, 1] == sp[1])), i, arr)
        return (i_and_arr, sp)
    mods_ID_uspecies = scan(f_miu, (0, mods_ID_uspecies), uspecies)[0][1]

    ipccd = atomllapi.load_atomicdata()
    ItIoI = atomllapi.ielem_to_index_of_ipccd
    Narr = jnp.array(10**(ipccd['solarA']))  # number density in the Sun

    def f_vmr(i, sp): return (i, jnp.where(sp[1] == 1,
                                           Narr[ItIoI[sp[0]]] / jnp.sum(Narr),
                                           Narr[ItIoI[sp[0]]] / jnp.sum(Narr) * 1e-10))
    VMR_uspecies = scan(f_vmr, 0, uspecies)[1]

    def f_mod(i_and_VMR, i_MIU):
        i, VMR_uspecies = i_and_VMR
        i_and_VMR = i + \
            1, VMR_uspecies.at[i_MIU].set(VMR_uspecies[i_MIU]*10**mods[i])
        return (i_and_VMR, i_MIU)
    VMR_uspecies = scan(f_mod, (0, VMR_uspecies), mods_ID_uspecies)[0][1]

    return VMR_uspecies


def get_VMR_uspecies_FC(FCSpIndex_uspecies, mixing_ratios):
    """By using FastChem, extract volume mixing ratio (VMR) of the species that
    contribute the opacity ("uspecies" made with "get_unique_species")

    Args:
        FCSpIndex_uspecies: SpeciesIndex in FastChem for each species of interest [N_species]
        mixing_ratios: volume mixing ratios of all available gases calculated using fastchem2_call.run_fastchem [N_layer x N_species]

    Returns:
        VMR_uspecies: VMR of each species in each atmospheric layer [N_species x N_layer]
    """
    def floop(i_sp, VMR_sp):
        VMR_sp = mixing_ratios[:, FCSpIndex_uspecies[i_sp]]
        i_sp = i_sp + 1
        return (i_sp, VMR_sp)

    i, VMR_uspecies = scan(floop, 0, jnp.zeros(len(FCSpIndex_uspecies)))
    return VMR_uspecies


def uspecies_info(uspecies, ielem_to_index_of_ipccd, mods_ID=jnp.array([[0, 0], ]), mods=jnp.array([0, ]), mods_id_trans=jnp.array([])):
    """Provide arrays of information of the species that contribute the opacity
    ("uspecies" made with "get_unique_species")

    Args:
       uspecies: jnp.array of unique list of the species contributing the opacity
       ielem_to_index_of_ipccd: jnp.array for conversion from ielem to the index of ipccd
       mods_ID: jnp.array listing the species whose abundances are different from the solar
       mods: jnp.array of each abundance deviation from the Sun [dex] for each modified species in mods_ID
       mods_id_trans: jnp.array for converting index in "mods_ID" of each species into index in uspecies

    Returns:
       MMR_uspecies_list: jnp.array of mass mixing ratio in the Sun of each species in "uspecies"
       atomicmass_uspecies_list: jnp.array of atomic mass [amu] of each species in "uspecies"
       mods_uspecies_list: jnp.array of abundance deviation from the Sun [dex] for each species in "uspecies"
    """
    ipccd = atomllapi.load_atomicdata()
    Narr = jnp.array(10**(ipccd['solarA']))  # number density
    # mass of each neutral atom per particle [amu]
    massarr = jnp.array(ipccd['mass'])
    Nmassarr = Narr * massarr  # mass density of each neutral species

    def floopMMR(i, arr):
        arr = Nmassarr[ielem_to_index_of_ipccd[uspecies[i][0]]
                       ] / jnp.sum(Nmassarr)
        i = i + 1
        return (i, arr)
    MMR_uspecies_list = scan(floopMMR, 0, np.zeros(len(uspecies)))[1]

    def floopAM(i, arr):
        arr = massarr[ielem_to_index_of_ipccd[uspecies[i][0]]]
        i = i + 1
        return (i, arr)
    atomicmass_uspecies_list = scan(
        floopAM, 0, np.zeros(len(uspecies)))[1]  # [amu]

    # for i, mit in enumerate(mods_id_trans):
    # mods_uspecies_list[mit] = mods[i]
    def f_Mmul(msi, null):
        ms, i = msi
        mit = mods_id_trans[i]
        ms = (ms.at[mit].set(mods[i]))
        i = i + 1
        msi = [ms, i]
        return (msi, null)
    length = len(mods)

    def g_Mmul(msi0):
        msi, null = scan(f_Mmul, msi0, None, length)
        return msi[0]

    mods_uspecies_list = jnp.zeros(len(uspecies))
    mods_uspecies_list = g_Mmul([mods_uspecies_list, 0])

    return (MMR_uspecies_list, atomicmass_uspecies_list, mods_uspecies_list)


def sep_arr_of_sp(arr, adb, trans_jnp=True, inttype=False):
    """Separate by species (atoms or ions) the jnp.array stored as an instance variable in adb, and pad with zeros to adjust the length

    Args:
        arr: array of a parameter (one of the attributes of adb below) to be separated [N_line]
        adb: adb instance made by the AdbVald class in moldb.py
        trans_jnp: if True, the output is converted to jnp.array (dtype='float32')
        inttype: if True (along with trans_jnp = True), the output is converted to jnp.array of dtype='int32'

    Returns:
        arr_stacksp: species-separated array [N_species x N_line_max]
    """
    uspecies = get_unique_species(adb)
    N_usp = len(uspecies)
    len_of_eachsp = np.zeros(N_usp, dtype='int')
    for i, sp in enumerate(uspecies):
        len_of_eachsp[i] = len(
            np.where((adb.ielem == sp[0]) * (adb.iion == sp[1]))[0])
    L_max = np.max(len_of_eachsp)

    arr_stacksp = np.zeros([N_usp, L_max])
    def pad0(arr, L): return np.pad(arr, ((0, L-len(arr))))
    for i, sp in enumerate(uspecies):
        index_sp = np.where((adb.ielem == sp[0]) * (adb.iion == sp[1]))[0]
        arr_t = jnp.take(arr, index_sp)
        arr_tp = pad0(arr_t, L_max)
        arr_stacksp[i] = arr_tp
    if trans_jnp:
        if inttype:
            arr_stacksp = jnp.array(arr_stacksp, dtype='int32')
        else:
            arr_stacksp = jnp.array(arr_stacksp)

    return arr_stacksp


def padding_2Darray_for_each_atom(orig_arr, adb, sp):
    """Extract only data of the species of interest from 2D-array and pad with
    zeros to adjust the length.

    Args:
        orig_arr: array [N_any (e.g., N_nu or N_layer), N_line]
            Note that if your ARRAY is 1D, it must be broadcasted with ARRAY[None,:], and the output must be also reshaped with OUTPUTARRAY.reshape(ARRAY.shape)
        adb: adb instance made by the AdbVald class in moldb.py
        sp: array of [ielem, iion]

    Returns:
       padded_valid_arr
    """
    orig_arr = orig_arr.T
    valid_indices = jnp.where(
        (adb.ielem == sp[0]) * (adb.iion == sp[1]), jnp.arange(adb.ielem.shape[0]), adb.ielem.shape[0])
    padding_zero = jnp.zeros([1, orig_arr.shape[1]])
    padded_arr = jnp.concatenate([orig_arr, padding_zero])
    padded_valid_arr = padded_arr[jnp.sort(valid_indices)]
    padded_valid_arr = padded_valid_arr.T
    return padded_valid_arr


def interp_QT284(T, T_gQT, gQT_284species):
    """interpolated partition function of all 284 species.

    Args:
        T: temperature
        T_gQT: temperature in the grid obtained from the adb instance [N_grid(42)]
        gQT_284species: partition function in the grid from the adb instance [N_species(284) x N_grid(42)]

    Returns:
        QT_284: interpolated partition function at T Q(T) for all 284 Atomic Species [284]
    """
    list_gQT_eachspecies = gQT_284species.tolist()
    listofDA_gQT_eachspecies = list(
        map(lambda x: jnp.array(x), list_gQT_eachspecies))
    listofQT = list(map(lambda x: jnp.interp(
        T, T_gQT, x), listofDA_gQT_eachspecies))
    QT_284 = jnp.array(listofQT)
    return QT_284
