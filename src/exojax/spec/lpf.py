#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Custom JVP version of the line profile functions used in exospectral
analysis."""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.special.faddeeva import rewofz, imwofz
from exojax.special.faddeeva import asymptotic_wofz
from jax import custom_jvp

# exomol
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import line_strength, doppler_sigma, gamma_natural

# vald
from exojax.spec.atomll import gamma_vald3

import warnings



def exomol(mdb, Tarr, Parr, molmass):
    """compute molecular line information required for MODIT using Exomol mdb.

    Args:
       mdb: mdb instance
       Tarr: Temperature array
       Parr: Pressure array
       molmass: molecular mass

    Returns:
       line intensity matrix,
       gammaL matrix,
       sigmaD matrix
    """

    qt = vmap(mdb.qr_interp)(Tarr)
    SijM = jit(vmap(line_strength, (0, None, None, None, 0)))(Tarr, mdb.logsij0,
                                                     mdb.dev_nu_lines,
                                                     mdb.elower, qt)
    gammaLMP = jit(vmap(gamma_exomol,
                        (0, 0, None, None)))(Parr, Tarr, mdb.n_Texp,
                                             mdb.alpha_ref)
    gammaLMN = gamma_natural(mdb.A)
    gammaLM = gammaLMP + gammaLMN[None, :]
    sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(mdb.nu_lines, Tarr,
                                                        molmass)
    return SijM, gammaLM, sigmaDM


def vald(adb, Tarr, PH, PHe, PHH):
    """Compute VALD line information required for LPF using VALD atomic database (adb)
    
    Args:
       adb: adb instance made by the AdbVald class in moldb.py
       Tarr: Temperature array
       PH: Partial pressure array of neutral hydrogen (H)
       PHe: Partial pressure array of neutral helium (He)
       PHH: Partial pressure array of molecular hydrogen (H2)

    Returns:
       SijM: line intensity matrix
       gammaLM: gammaL matrix
       sigmaDM: sigmaD matrix
    
    """
    # Compute normalized partition function for each species
    qt_284 = vmap(adb.QT_interp_284)(Tarr)
    qt = qt_284[:, adb.QTmask]

    # Compute line strength matrix
    SijM = jit(vmap(line_strength,(0,None,None,None,0)))\
        (Tarr, adb.logsij0, adb.nu_lines, adb.elower, qt)

    # Compute gamma parameters for the pressure and natural broadenings
    gammaLM = jit(vmap(gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
            (Tarr, PH, PHH, PHe, adb.ielem, adb.iion, adb.dev_nu_lines, adb.elower, adb.eupper, adb.atomicmass, adb.ionE, adb.gamRad, adb.gamSta, adb.vdWdamp, 1.0)

    # Compute doppler broadening
    sigmaDM = jit(vmap(doppler_sigma,(None,0,None)))\
        (adb.nu_lines, Tarr, adb.atomicmass)

    return SijM, gammaLM, sigmaDM


def vald_each(Tarr, PH, PHe, PHH, \
            qt_284_T, QTmask, \
             logsij0, nu_lines, ielem, iion, dev_nu_lines, elower, eupper, atomicmass, ionE, gamRad, gamSta, vdWdamp, ):
    """Compute VALD line information required for LPF for separated each species
    
    Args:
        Tarr:  temperature array [N_layer]
        PH:  partial pressure array of neutral hydrogen (H) [N_layer]
        PHe:  partial pressure array of neutral helium (He) [N_layer]
        PHH:  partial pressure array of molecular hydrogen (H2) [N_layer]
        qt_284_T:  partition function at the temperature T Q(T), for 284 species
        QTmask:  array of index of Q(Tref) grid (gQT) for each line
        logsij0:  log line strength at T=Tref
        nu_lines:  line center (cm-1) in np.array (float64)
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
        dev_nu_lines:  line center (cm-1) in device (float32)
        elower:  the lower state energy (cm-1)
        eupper:  the upper state energy (cm-1)
        atomicmass:  atomic mass (amu)
        ionE:  ionization potential (eV)
        gamRad:  log of gamma of radiation damping (s-1)
        gamSta:  log of gamma of Stark damping (s-1)
        vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)

    Returns:
       SijM: line intensity matrix [N_layer x N_line]
       gammaLM: gammaL matrix [N_layer x N_line]
       sigmaDM: sigmaD matrix [N_layer x N_line]
    
    """
    # Compute normalized partition function for each species
    qt = qt_284_T[:, QTmask]

    # Compute line strength matrix
    SijM = jit(vmap(line_strength,(0,None,None,None,0)))\
        (Tarr, logsij0, nu_lines, elower, qt)
    SijM = jnp.nan_to_num(SijM, nan=0.0)

    # Compute gamma parameters for the pressure and natural broadenings
    gammaLM = jit(vmap(gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
            (Tarr, PH, PHH, PHe, ielem, iion, dev_nu_lines, elower, eupper, atomicmass, ionE, gamRad, gamSta, vdWdamp, 1.0)

    # Compute doppler broadening
    sigmaDMn=jit(vmap(doppler_sigma,(None,0,None)))\
        (nu_lines, Tarr, atomicmass)
    sigmaDM = jnp.where(sigmaDMn != 0, sigmaDMn, 1.)

    return SijM, gammaLM, sigmaDM


@jit
def ljert(x, a):
    """ljert function, consisting of a combination of imwofz and imag(asymptiotic wofz).

    Args:
        x:
        a:

    Returns:
        L(x,a) or Imag(wofz(x+ia))

    Note:
        ljert provides a L(x,a) function. This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.
    """
    r2 = x * x + a * a
    return jnp.where(r2 < 111., imwofz(x, a), jnp.imag(asymptotic_wofz(x, a)))


@custom_jvp
def hjert(x, a):
    """custom JVP version of the Voigt-Hjerting function, consisting of a
    combination of rewofz and real(asymptotic wofz).

    Args:
        x: 
        a:

    Returns:
        H(x,a) or Real(wofz(x+ia))

    Examples:

       hjert provides a Voigt-Hjerting function w/ custom JVP. 

       >>> hjert(1.0,1.0)
          DeviceArray(0.30474418, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)
    """
    r2 = x * x + a * a
    return jnp.where(r2 < 111., rewofz(x, a), jnp.real(asymptotic_wofz(x, a)))


@hjert.defjvp
def hjert_jvp(primals, tangents):
    x, a = primals
    ux, ua = tangents
    dHdx = 2.0 * a * ljert(x, a) - 2.0 * x * hjert(x, a)
    dHda = 2.0 * x * ljert(x, a) + 2.0 * a * hjert(x, a) - 2.0 / jnp.sqrt(
        jnp.pi)
    primal_out = hjert(x, a)
    tangent_out = dHdx * ux + dHda * ua
    return primal_out, tangent_out


@jit
def voigtone(nu, sigmaD, gammaL):
    """Custom JVP version of (non-vmapped) Voigt function using Voigt-Hjerting
    function.

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt funtion
    """

    sfac = 1.0 / (jnp.sqrt(2) * sigmaD)
    v = sfac * hjert(sfac * nu, sfac * gammaL) / jnp.sqrt(jnp.pi)
    return v


@jit
def voigt(nuvector, sigmaD, gammaL):
    """Custom JVP version of Voigt profile using Voigt-Hjerting function.

    Args:
       nu: wavenumber array
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile
    """

    sfac = 1.0 / (jnp.sqrt(2.0) * sigmaD)
    vhjert = vmap(hjert, (0, None), 0)
    v = sfac * vhjert(sfac * nuvector, sfac * gammaL) / jnp.sqrt(jnp.pi)
    return v


@jit
def vvoigt(numatrix, sigmaD, gammas):
    """Custom JVP version of vmaped voigt profile.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline

    Return:
       Voigt profile vector in R^Nwav
    """
    vmap_voigt = vmap(voigt, (0, 0, 0), 0)
    return vmap_voigt(numatrix, sigmaD, gammas)


@jit
def xsvector(numatrix, sigmaD, gammaL, Sij):
    """Custom JVP version of cross section vector.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline
       Sij: line strength vector in R^Nline

    Return:
       cross section vector in R^Nwav
    """
    return jnp.dot((vvoigt(numatrix, sigmaD, gammaL)).T, Sij)


@jit
def xsmatrix(numatrix, sigmaDM, gammaLM, SijM):
    """Custom JVP version of cross section matrix.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    return vmap(xsvector, (None, 0, 0, 0))(numatrix, sigmaDM, gammaLM, SijM)


from exojax.spec.make_numatrix import make_numatrix0
import numpy as np
import tqdm


def auto_xsection(nu, nu_lines, sigmaD, gammaL, Sij, memory_size=15.):
    """compute cross section.

    Warning:
       This is NOT auto-differentiable function.

    Args:
       nu: wavenumber array
       nu_lines: line center
       sigmaD: sigma parameter in Doppler profile 
       gammaL:  broadening coefficient in Lorentz profile 
       Sij: line strength
       memory_size: memory size for numatrix0 (MB)

    Returns:
       numpy.array: cross section (xsv)

    Example:
       >>> from exojax.spec.lpf import auto_xsection
       >>> from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
       >>> from exojax.spec import moldb
       >>> import numpy as np
       >>> nus=np.linspace(1000.0,10000.0,900000,dtype=np.float64) #cm-1
       >>> mdbCO=moldb.MdbHit('~/exojax/data/CO','05_hit12',nus)
       >>> Mmol=28.010446441149536 # molecular weight
       >>> Tfix=1000.0 # we assume T=1000K
       >>> Pfix=1.e-3 # we compute P=1.e-3 bar
       >>> Ppart=Pfix #partial pressure of CO. here we assume a 100% CO atmosphere. 
       >>> qt=mdbCO.qr_interp_lines(Tfix)
       >>> Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
       >>> gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self) + gamma_natural(mdbCO.A) 
       >>> sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
       >>> nu_lines=mdbCO.nu_lines
       >>> xsv=auto_xsection(nus,nu_lines,sigmaD,gammaL,Sij,memory_size=30)
        100%|████████████████████████████████████████████████████| 456/456 [00:03<00:00, 80.59it/s]
    """
    NL = len(nu_lines)
    d = int(memory_size/(NL*4/1024./1024.))
    if d > 0:
        Ni = int(len(nu)/d)
        xsv = []
        for i in tqdm.tqdm(range(0, Ni+1)):
            s = int(i*d)
            e = int((i+1)*d)
            e = min(e, len(nu))
            numatrix = make_numatrix0(nu[s:e], nu_lines, warning=False)
            xsv = np.concatenate(
                [xsv, xsvector(numatrix, sigmaD, gammaL, Sij)])
    else:
        NP = int((NL*4/1024./1024.)/memory_size)+1
        d = int(memory_size/(int(NL/NP)*4/1024./1024.))
        Ni = int(len(nu)/d)
        dd = int(NL/NP)
        xsv = []
        for i in tqdm.tqdm(range(0, Ni+1)):
            s = int(i*d)
            e = int((i+1)*d)
            e = min(e, len(nu))
            xsvtmp = np.zeros_like(nu[s:e])
            for j in range(0, NP+1):
                ss = int(j*dd)
                ee = int((j+1)*dd)
                ee = min(ee, NL)
                numatrix = make_numatrix0(
                    nu[s:e], nu_lines[ss:ee], warning=False)
                xsvtmp = xsvtmp + \
                    xsvector(numatrix, sigmaD[ss:ee],
                             gammaL[ss:ee], Sij[ss:ee])
            xsv = np.concatenate([xsv, xsvtmp])

    if(nu.dtype != np.float64):
        warnings.warn('The wavenumber grid is not np.float64 but '+str(nu.dtype),UserWarning)
    if(nu_lines.dtype != np.float64):
        warnings.warn('The line centers (nu_lines) are not np.float64 but '+str(nu.dtype),UserWarning)


    return xsv

