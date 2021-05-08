#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""radiative transfer module used in exospectral analysis.

"""
from jax import jit
import jax.numpy as jnp
import numpy as np
from exojax.special.expn import E1
from exojax.spec.hitrancia import logacia

def nugrid(x0,x1,N,unit="cm-1"):
    """generating wavenumber grid

    Args:
       x0: start wavenumber (cm-1) or wavelength (nm) or (AA)
       x1: end wavenumber (cm-1) or wavelength (nm) or (AA)
       N: the number of the wavenumber grid
       unit: unit of the input grid
    
    Returns:
       wavenumber grid evenly spaced in log space
       corresponding wavelength grid
       resolution

    """
    if unit=="cm-1":
        nus=np.logspace(np.log10(x0),np.log10(x1),N,dtype=np.float64)#AA
        wav=1.e8/nus[::-1]
    elif unit=="nm":
        wav=np.logspace(np.log10(x0),np.log10(x1),N,dtype=np.float64)#AA
        nus=1.e7/wav[::-1]
    elif unit=="AA":
        wav=np.logspace(np.log10(x0),np.log10(x1),N,dtype=np.float64)#AA
        nus=1.e8/wav[::-1]
        
    dlognu=(np.log10(nus[-1])-np.log10(nus[0]))/N
    resolution=1.0/dlognu
    if resolution<300000.0:
        print("WARNING: resolution may be too small. R=",resolution)
        
    return nus, wav, resolution

def check_nugrid(nus,crit1=1.e-5,crit2=1.e-14):
    """checking if nugrid is evenly spaced in a logarithm scale (ESLOG)

    Args:
       nus: wavenumber grid
       crit1: criterion for the maximum deviation of log10(nu)/median(log10(nu)) from ESLOG 
       crit2: criterion for the maximum deviation of log10(nu) from ESLOG 

    Returns:
       True (nugrid is ESLOG) or False (not)

    """
    q=np.log10(nus)
    p=q[1:]-q[:-1]
    w=(p-np.mean(p))
    val1=np.max(np.abs(w))/np.median(p)
    val2=np.max(np.abs(w))
    if val1<crit1 and val2 < crit2:
        return True
    else:
        return False
    

def pressure_layer(logPtop=-8.,logPbtm=2.,NP=20,mode="ascending"):
    """generating the pressure layer
    
    Args: 
       logPtop: log10(P[bar]) at the top layer
       logPbtm: log10(P[bar]) at the bottom layer
       NP: the number of the layers

    Returns: 
         Parr: pressure layer
         dParr: delta pressure layer 
         k: k-factor, P[i-1] = k*P[i]

    Note:
        dParr[i] = Parr[i] - Parr[i-1], dParr[0] = (1-k) Parr[0] for ascending mode
    
    """
    dlogP=(logPbtm-logPtop)/(NP-1)
    k=10**-dlogP
    Parr=jnp.logspace(logPtop,logPbtm,NP)
    dParr = (1.0-k)*Parr
    if mode=="descending":
        Parr=Parr[::-1] 
        dParr=dParr[::-1]
    
    return jnp.array(Parr), jnp.array(dParr), k


def dtauCIA(nus,Tarr,Parr,dParr,vmr1,vmr2,mmw,g,nucia,tcia,logac):
    """dtau of the CIA continuum

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
       vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
       mmw: mean molecular weight of atmosphere
       g: gravity (cm2/s)
       nucia: wavenumber array for CIA
       tcia: temperature array for CIA
       logac: log10(absorption coefficient of CIA)

    Returns:
       optical depth matrix  [N_layer, N_nus] 

    Note:
       logm_ucgs=np.log10(m_u*1.e3) where m_u = scipy.constants.m_u.

    """
    kB=1.380649e-16
    logm_ucgs=-23.779750909492115

    narr=(Parr*1.e6)/(kB*Tarr)
    lognarr1=jnp.log10(vmr1*narr) #log number density
    lognarr2=jnp.log10(vmr2*narr) #log number density
    
    logkb=np.log10(kB)    
    logg=jnp.log10(g)
    ddParr=dParr/Parr
    
    dtauc=(10**(logacia(Tarr,nus,nucia,tcia,logac)\
            +lognarr1[:,None]+lognarr2[:,None]+logkb-logg-logm_ucgs)\
            *Tarr[:,None]/mmw*ddParr[:,None])

    return dtauc
    
def dtauM(dParr,xsm,MR,mass,g):
    """dtau of the molecular cross section

    Note:
       fac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_layer, N_nus]
       MR: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
       mass: mean molecular weight for VMR or molecular mass for MMR
       g: gravity (cm/s2)

    Returns:
       optical depth matrix [N_layer, N_nus]

    """

    fac=6.022140858549162e+29
    return fac*xsm*dParr[:,None]*MR[:,None]/(mass*g)


@jit
def trans2E3(x):
    """transmission function 2E3 (two-stream approximation with no scattering) expressed by 2 E3(x)

    Note:
       The exponetial integral of the third order E3(x) is computed using Abramowitz Stegun (1970) approximation of E1 (exojax.special.E1).

    Args:
       x: input variable

    Returns:
       Transmission function T=2 E3(x)
    
    """
    return (1.0-x)*jnp.exp(-x) + x**2*E1(x)

@jit
def rtrun(dtau,S):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)

    Args:
        dtau: opacity matrix 
        S: source matrix [N_layer, N_nus]
 
    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.

    """
    Nnus=jnp.shape(dtau)[1]
    TransM=jnp.where(dtau==0, 1.0, trans2E3(dtau))   
    QN=jnp.zeros(Nnus)
    Qv=(1-TransM)*S
    Qv=jnp.vstack([Qv,QN])
    onev=jnp.ones(Nnus)
    TransM=jnp.vstack([onev,TransM])
    Fx=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
    return Fx

@jit
def rtrun_direct(dtau,S):
    """Radiative Transfer using direct integration

    Note: 
        Use dtau/mu instead of dtau when you want to use non-unity, where mu=cos(theta)

    Args:
        dtau: opacity matrix 
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    taupmu=jnp.cumsum(dtau,axis=0)
    Fx=jnp.sum(S*jnp.exp(-taupmu)*dtau,axis=0)
    return Fx


@jit
def rtrun_surface(dtau,S,Sb):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type) with a planetary surface

    Args:
        dtau: opacity matrix 
        S: source matrix [N_layer, N_nus]
        Sb: source from the surface [N_nus]
 
    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus=jnp.shape(dtau)[1]
    TransM=jnp.where(dtau==0, 1.0, trans2E3(dtau))   
    QN=Sb
    Qv=(1-TransM)*S
    Qv=jnp.vstack([Qv,QN])
    onev=jnp.ones(Nnus)
    TransM=jnp.vstack([onev,TransM])
    Fx=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
    return Fx



if __name__ == "__main__":
    
    nus,wav,res=nugrid(22999,23000,1000,"AA")
    print(check_nugrid(nus))
    nus,wav,res=nugrid(22999,23000,10000,"AA")
    print(check_nugrid(nus))
    nus,wav,res=nugrid(22999,23000,100000,"AA")
    print(check_nugrid(nus))
    nus=np.linspace(1.e8/23000.,1.e8/22999.,1000)
    print(check_nugrid(nus))
    nus=np.linspace(1.e8/23000.,1.e8/22999.,10000)
    print(check_nugrid(nus))
    nus=np.linspace(1.e8/23000.,1.e8/22999.,100000)
    print(check_nugrid(nus))

    
