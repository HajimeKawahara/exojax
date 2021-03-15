#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""radiative transfer module used in exospectral analysis.

"""
from jax import jit
import jax.numpy as jnp
from exojax.special.expn import E1

def dtaux(dParr,xsm,MR,mass,g):
    """dtau from the molecular cross section

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
def rtrun(dtauM,S):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)
    Args:
        dtauM: opacity matrix 
        S: source matrix [N_layer, N_nus]
 
    Returns:
        flux in the unit of [erg/cm2/s/Hz]
    """
    ccgs=29979245800.0 #c (cgs)
    Nnus=jnp.shape(dtauM)[1]
    TransM=jnp.where(dtauM==0, 1.0, trans2E3(dtauM))   
    QN=jnp.zeros(Nnus)
    Qv=(1-TransM)*S
    Qv=jnp.vstack([Qv,QN])
    onev=jnp.ones(Nnus)
    TransM=jnp.vstack([onev,TransM])
    Fx=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
    return Fx/ccgs

@jit
def rtrun_direct(dtauM,S):
    """Radiative Transfer using direct integration

    Note: 
        Use dtauM/mu instead of dtauM when you want to use non-unity, where mu=cos(theta)

    Args:
        dtauM: opacity matrix 
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/Hz]
    """
    ccgs=29979245800.0 #c (cgs)
    taupmu=jnp.cumsum(dtauM,axis=0)
    Fx=jnp.sum(S*jnp.exp(-taupmu)*dtauM,axis=0)
    return Fx/ccgs


@jit
def rtrun_surface(dtauM,S,Sb):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type) with a planetary surface

    Args:
        dtauM: opacity matrix 
        S: source matrix [N_layer, N_nus]
        Sb: source from the surface [N_nus]
 
    Returns:
        flux in the unit of [erg/cm2/s/Hz]
    """
    ccgs=29979245800.0 #c (cgs)
    Nnus=jnp.shape(dtauM)[1]
    TransM=jnp.where(dtauM==0, 1.0, trans2E3(dtauM))   
    QN=Sb
    Qv=(1-TransM)*S
    Qv=jnp.vstack([Qv,QN])
    onev=jnp.ones(Nnus)
    TransM=jnp.vstack([onev,TransM])
    Fx=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
    return Fx/ccgs

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



