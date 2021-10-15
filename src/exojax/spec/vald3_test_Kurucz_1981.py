import numpy as np
import jax.numpy as jnp

def Sij0(A, gupper, nu_lines, elower, QTref_284, QTmask):
    """Reference Line Strength in Tref=296K, S0.

    Note:
       Tref=296K

    Args:.
       A: Einstein coefficient (s-1)
       gupper: the upper state statistical weight
       nu_lines: line center wavenumber (cm-1)
       elower: elower
       QTref_284: partition function Q(Tref)
       QTmask: mask to identify a rows of QTref_284 to apply for each line
       Mmol: molecular mass (normalized by m_u)

    Returns:
       Sij(T): Line strength (cm)

    """
    ccgs=29979245800.0 #[cm/s]
    hcperk=1.4387773538277202 #hc/kB in cgs
    Tref=296.0

    #Assign Q(Tref) for each line
    QTref = np.zeros_like(QTmask, dtype=float)
    for i, mask in enumerate(QTmask):
        QTref[i] = QTref_284[mask]

    S0 = -A*gupper*np.exp(-hcperk*elower/Tref)*np.expm1(-hcperk*nu_lines/Tref)\
        /(8.0*np.pi*ccgs*nu_lines**2*QTref)

    return(S0)


def gamma_natural(A):
    """gamma factor by natural broadning
    
    1/(4 pi c) = 2.6544188e-12 (cm-1 s)

    Args:
       A: Einstein A-factor (1/s)

    Returns:
       gamma_natural: natural width (cm-1)

    """
    return(2.6544188e-12*A)
    

def gamma_vald3(P, T, PH, PHH, PHe, \
    nu_lines, elower, ionE, gamRad, vdWdamp, enh_damp=1.0):
    #%\\\\20210804 #tako 原子ごとにenh_dampも場合分け（Turbospectrumに倣う？）するならielemが入力(Args)に必要.
    """gamma factor by a pressure broadening
      based on Gray+2005(2005oasp.book.....G)

    Args(inputs):
      T: temperature (K) (array)
      P: pressure (bar)  #1 bar = 1e6 dyn/cm2 (array)
      PH: hydrogen pressure (bar) (array)
      PHH: H2 molecule pressure (bar) (array)
      PHe: helium pressure (bar) (array)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      ionE: ionization potential [eV]
      vdWdamp:  van der Waals damping parameters
      gamRad: log of gamma(HWHM of Lorentzian) of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant

    Args(calculated):
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line
      C6: interaction constant (Eq.11.17 in Gray)
      logg6: log(gamma6) (Eq.11.29 in Gray)
      gam6H: 17*v**(0.6)*C6**(0.4)*N
           #(v:relative velocity, N:number density of neutral perturber)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4)

    Returns:
      gamma: pressure gamma factor (cm-1)

    """
    hcgs = 6.62607015e-27 #[erg*s]
    ccgs = 2.99792458e10 #[cm/s]
    kcgs = 1.38064852e-16 #[erg/K]

    #CASE1 (classical approximation by Unsoeld (1955))
    chi_lam = nu_lines/8065.54 #[cm-1] -> [eV]
    chi = elower/8065.54 #[cm-1] -> [eV]
    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2)) #possibly need "ION**2" factor as turbospectrum?
    #logg6 = 20 + 0.4*jnp.log10(C6) + jnp.log10(PH*1e6) - 0.7*jnp.log10(T)
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7  # = 10**logg6
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = gamma6 + 10**gamRad
    gamma_case1 = np.where(np.isnan(gamma_case1), 0., gamma_case1) #avoid nan (appeared by jnp.log10(negative C6))

    #CASE2 (van der Waars broadening based on gamma6 at 10000 K)
    Texp = 0.38 #Barklem+2000
    gam6H = 10**vdWdamp * (T/10000.)**Texp * PH*1e6 /(kcgs*T)
    gam6He = 10**vdWdamp * (T/10000.)**Texp * PHe*1e6*0.41336 /(kcgs*T)
    gam6HH = 10**vdWdamp * (T/10000.)**Texp * PHH*1e6*0.85 /(kcgs*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case2 = gamma6 + 10**gamRad

    #Prioritize Case2 (Case1 if w/o vdW)
    gamma = (gamma_case1 * jnp.where(vdWdamp>=0., 1, 0) + gamma_case2 * jnp.where(vdWdamp<0., 1, 0))\
        /ccgs  #Note that gamRad in VALD is in [s-1] (https://www.astro.uu.se/valdwiki/Vald3Format)

    return(gamma)


def gamma_vald3_Kurucz1981(T, PH, PHH, PHe, \
    nu_lines, elower, eupper, atomicmass, ionE, gamRad, vdWdamp, ielem, enh_damp=1.0):
    #%\\\\20210917 #tako 原子ごとにenh_dampも場合分け（Turbospectrumに倣う？）するならielemが入力(Args)に必要.
    """
      gamma factor by a pressure broadening
      based on Gray+2005(2005oasp.book.....G)

    Args(inputs):
      T: temperature (K) (array)
      PH: hydrogen pressure (bar) (array)  #1 bar = 1e6 dyn/cm2 (array)
      PHH: H2 molecule pressure (bar) (array)
      PHe: helium pressure (bar) (array)
      nu_lines:  transition waveNUMBER in [cm-1] (NOT frequency in [s-1])
      elower: excitation potential (lower level) [cm-1]
      eupper: excitation potential (upper level) [cm-1]
      atomicmass: atomic mass [amu]
      ionE: ionization potential [eV]
      vdWdamp:  van der Waals damping parameters
      gamRad: gamma(HWHM of Lorentzian) of radiation damping
      enh_damp: empirical "enhancement factor" for classical Unsoeld's damping constant

    Args(calculated):
      chi_lam (=h*nu=1.2398e4/wvl[AA]): energy of a photon in the line
      C6: interaction constant (Eq.11.17 in Gray)
      logg6: log(gamma6) (Eq.11.29 in Gray)
      gam6H: 17*v**(0.6)*C6**(0.4)*N
           #(v:relative velocity, N:number density of neutral perturber)
      Texp: temperature dependency (gamma6 \sim T**((1-α)/2) ranging 0.3–0.4)

    Returns:
      gamma: pressure gamma factor (cm-1)

    """
    hcgs = 6.62607015e-27 #[erg*s]
    ccgs = 2.99792458e10 #[cm/s]
    kcgs = 1.38064852e-16 #[erg/K]

    #CASE1 (classical approximation by Unsoeld (1955))
    chi_lam = nu_lines/8065.54 #[cm-1] -> [eV]
    chi = elower/8065.54 #[cm-1] -> [eV]
    C6 = 0.3e-30 * ((1/(ionE-chi-chi_lam)**2) - (1/(ionE-chi)**2)) #possibly need "ION**2" factor as turbospectrum?
    #logg6 = 20 + 0.4*jnp.log10(C6) + jnp.log10(PH*1e6) - 0.7*jnp.log10(T)
    gam6H = 1e20 * C6**0.4 * PH*1e6 / T**0.7  # = 10**logg6
    gam6He = 1e20 * C6**0.4 * PHe*1e6*0.41336 / T**0.7
    gam6HH = 1e20 * C6**0.4 * PHH*1e6*0.85 / T**0.7
    gamma6 = enh_damp * (gam6H + gam6He + gam6HH)
    gamma_case1 = (gamma6 )/ccgs # + 10**gamRad)/ccgs
    gamma_case1 = np.where(np.isnan(gamma_case1), 0., gamma_case1) #avoid nan (appeared by jnp.log10(negative C6))

    #CASE2 (van der Waars broadening based on gamma6 at 10000 K)
    Texp = 0.38 #Barklem+2000
    gam6H = 10**vdWdamp * (T/10000.)**Texp * PH*1e6 /(kcgs*T)
    gam6He = 10**vdWdamp * (T/10000.)**Texp * PHe*1e6*0.41336 /(kcgs*T)
    gam6HH = 10**vdWdamp * (T/10000.)**Texp * PHH*1e6*0.85 /(kcgs*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case2 = (gamma6 ) /ccgs /(4*np.pi) # + 10**gamRad)/ccgs




    #CASE3 (3rd equation in p.4 of Kurucz_1981_solar_spectrum_synthesis_SAOSR_391_____K.pdf)
    ##test210917
    
    #gamma_case3 = 10**vdWdamp * ((PH+0.42*PHe+0.85*PHH)*1e6/(kcgs*T)) * (T/10000.)**0.3    + 10**gamRad
    #ってこれ結局CASE2のTexpが0.30になっただけ…
    
    eupper = elower+nu_lines #tesTako\\\\202109
    
    Rcgs = 1.0973731568e5 #[cm-1]
    ucgs = 1.660539067e-24 #unified atomic mass unit [g]
    a0 = 5.2917720859e-9 #[cm]
    ecgs = 4.803204e-10 #elementary charge [esu]
    Zeff = 1.
    n_eff2_upper = Rcgs * Zeff**2 / (ionE*8065.54 - eupper) #Square of effective quantum number of the upper state
    n_eff2_lower = Rcgs * Zeff**2 / (ionE*8065.54 - elower)
    msr_upper = np.where(n_eff2_upper>0., (2.5 * (n_eff2_upper/Zeff)**2), 25) #Mean of square of radius of the upper level
    msr_lower = 2.5 * (n_eff2_lower/Zeff)**2 #Mean of square of radius of the upper level
    msr_upper_anothereq = (45-ielem)/Zeff #5ht equation in Kurucz_1981
    #in units of a0, the radius of the first Bohr orbit (noticed by Kawashima-san (ref. p.320 in Aller (1963))

    """gamma_case3 = 17 * (8*kcgs*T*(1/atomicmass+1/1)/(jnp.pi*1.008000))**0.3 * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev))**0.4 * PH*1e6 /(kcgs*T) + \
                    17 * (8*kcgs*T*(1/atomicmass+1/4)/(jnp.pi*4.002600))**0.3 * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev))**0.4 * PHe*1e6 /(kcgs*T) + \
                    17 * (8*kcgs*T*(1/atomicmass+1/2)/(jnp.pi*1.008000*2))**0.3 * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev))**0.4 * PHH*1e6 /(kcgs*T) #"""
                  
    gap_msr = msr_upper - msr_lower
    gap_msr_rev = gap_msr * np.where(gap_msr < 0, -1., 1.) #Reverse upper and lower if necessary(TBC)_\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev #[Bohr radius -> cm]
    gam6H = 17 * (8*kcgs*T*(1./atomicmass+1./1.)/(jnp.pi*ucgs))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 /(kcgs*T)
    gam6He = 17 * (8*kcgs*T*(1./atomicmass+1./4.)/(jnp.pi*ucgs))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 /(kcgs*T)
    gam6HH = 17 * (8*kcgs*T*(1./atomicmass+1./2.)/(jnp.pi*ucgs))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 /(kcgs*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case3 = (gamma6 ) /ccgs /(4*np.pi) # + 10**gamRad)/ccgs



    
    #CASE3.5: minor change from CASE3
    gap_msr = msr_upper_anothereq - msr_lower
    gap_msr_rev = gap_msr * np.where(gap_msr < 0, -1., 1.) #Reverse upper and lower if necessary(TBC)_\\\\
    gap_msr_rev_cm = a0**2 * gap_msr_rev #[Bohr radius -> cm]
    gam6H = 17 * (8*kcgs*T*(1./atomicmass+1./1.)/(jnp.pi*ucgs))**0.3 \
        * (6.63e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PH*1e6 /(kcgs*T)
    gam6He = 17 * (8*kcgs*T*(1./atomicmass+1./4.)/(jnp.pi*ucgs))**0.3 \
        * (2.07e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHe*1e6 /(kcgs*T)
    gam6HH = 17 * (8*kcgs*T*(1./atomicmass+1./2.)/(jnp.pi*ucgs))**0.3 \
        * (8.04e-25*ecgs**2/hcgs*(gap_msr_rev_cm))**0.4 \
        * PHH*1e6 /(kcgs*T)
    gamma6 = gam6H + gam6He + gam6HH
    gamma_case3half = (gamma6 ) /ccgs /(4*np.pi) # + 10**gamRad)/ccgs


    #CASE4 (4th equation in p.4 of Kurucz_1981)
    gamma6 = 4.5e-9 * msr_upper**0.4 \
        * ((PH + 0.42*PHe + 0.85*PHH)*1e6/(kcgs*T)) * (T/10000.)**0.3
    gamma_case4 = (gamma6 ) /ccgs /(4*np.pi) # + 10**gamRad)/ccgs

    #CASE5: minor change from CASE4 (4th equation in p.4 of Kurucz_1981)
    gamma6 = 4.5e-9 * msr_upper_anothereq**0.4 \
        * ((PH + 0.42*PHe + 0.85*PHH)*1e6/(kcgs*T)) * (T/10000.)**0.3
    gamma_case5 = (gamma6 ) /ccgs /(4*np.pi) # + 10**gamRad)/ccgs




    #Prioritize Case2 (Case1 if w/o vdW)
    #gamma = (gamma_case1 * jnp.where(vdWdamp>=0., 1, 0) + gamma_case2 * jnp.where(vdWdamp<0., 1, 0))
    #return(gamma)
    
    
    return(gamma_case2, gamma_case3, gamma_case4, gamma_case5, gamma_case3half, msr_upper, msr_lower, msr_upper_anothereq)
    #return(gamma, n_eff2_upper, n_eff2_lower, msr_upper, msr_lower, gam6H, gam6He, gam6HH, gammatest)
