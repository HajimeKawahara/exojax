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
    

def gamma_vald3(T, PH, PHH, PHe, \
    nu_lines, elower, ionE, gamRad, vdWdamp, enh_damp=1.0):
    #%\\\\20210804 #tako 原子ごとにenh_dampも場合分け（Turbospectrumに倣う？）するならielemが入力(Args)に必要.
    """gamma factor by a pressure broadening
      based on Gray+2005(2005oasp.book.....G)

    Args(inputs):
      T: temperature (K) (array)
      PH: hydrogen pressure (bar) (array)  #1 bar = 1e6 dyn/cm2 (array)
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
    #hcgs = 6.62607015e-27 #[erg*s]
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
