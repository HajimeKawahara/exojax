#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""radiative transfer module used in exospectral analysis.

"""
from jax import jit
from jax.lax import scan
from exojax.spec import lpf
import jax.numpy as jnp
from exojax.spec import planck
from functools import partial
from exojax.spec.clpf import cxsmatrix


@jit
def trans2E3(x):
    """transmission function given by 2 E3(x)

    Notes:
       The exponetial integral of the third order E3(x) is computed using Abramowitz Stegun (1970) approximation of E1

    Args:
       x: input

    Returns:
       2 E3(x)
    
    """
    from exojax.special.expn import E1
    return ((1.0-x)*jnp.exp(-x) + x**2*E1(x))

@jit
def rtrun(xsm,tfac,gi,dParr,epsilon=1.e-20):
    """Radiative Transfer using 2 stream+AS (Helios-R1 type)
    Args:
        xsm: cross section matrix (cm2)
        tfac: conversion factor pressure x cross section to tau
        gi: blackbody emission layer
        dParr: delta P 
        epsilon: small number to avoid zero tau layer
 
    Returns:
        flux in the unit of [erg/cm2/s/Hz]
    """
    Nnus=jnp.shape(xsm)[1]
    dtauM=dParr[:,None]*xsm*tfac[:,None]+epsilon
    TransMx=trans2E3(dtauM)
    TransM=jnp.where(dtauM==0, 1.0, TransMx)   
    QN=jnp.zeros(Nnus)
    Qv=(1-TransM)*gi
    Qv=jnp.vstack([Qv,QN])
    onev=jnp.ones(Nnus)
    TransM=jnp.vstack([onev,TransM])
    Fx=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
    ccgs=29979245800.0 #c (cgs)
    return Fx/ccgs



def pressure_layer(logPtop=-8.,logPbtm=2.,NP=20,mode="ascending"):
    """constructing the pressure layer
    
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
    
    return Parr, dParr, k

def ASfactors():
    A0=-0.57721566
    A1= 0.99999193
    A2=-0.24991055
    A3= 0.05519968
    A4=-0.00976004
    A5= 0.00107857
    B1=8.5733287401
    B2=18.059016973
    B3=8.6347608925
    B4=0.2677737343
    C1=9.5733223454
    C2=25.6329561486
    C3=21.0996530827
    C4=3.9584969228
    return A0,A1,A2,A3,A4,A5,B1,B2,B3,B4,C1,C2,C3,C4



