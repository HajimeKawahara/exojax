#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""radiative transfer module used in exospectral analysis.

"""
from jax import jit
import jax.numpy as jnp
import numpy as np
from exojax.special.expn import E1
from exojax.spec.hitrancia import logacia

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
    logg=np.log10(g)
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



