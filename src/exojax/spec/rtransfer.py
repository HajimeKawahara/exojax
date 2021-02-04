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

__all__ = ['JaxRT']

class JaxRT(object):
    """Jax Radiative Transfer class
    
    """
    def __init__(self):
        self.nuarr = []
        self.numic = 0.5 # 0.5 micron for planck
        self.Sfix = []
        self.Parr = []
#        self.dParr = []
        

    @partial(jit, static_argnums=(0,))
    def run(self,nu0,sigmaD,gammaL,source):
        """Running RT by linear algebra radiative transfer using vmap

        Note: 

        Args: 
           nu0: reference wavenumber
           sigmaD: STD of a Gaussian profile
           gammaL: gamma factor of Lorentzian
           source: source vector in the atmospheric layers
           
        Returns:
           F: upward flux

        """
        numatrix=lpf.make_numatrix(self.nuarr,self.hatnufix,nu0)
        
        xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM)
        xsv = 1.e-1*crossx(numatrix,sigmaD,gammaL,self.Sfix)
        dtauM=self.dParr[:,None]*xsv[None,:]
        TransM=(1.0-dtauM)*jnp.exp(-dtauM)

        #QN=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
        QN=jnp.zeros(len(self.nuarr))
        Qv=(1-TransM)*source[:,None]
        Qv=jnp.vstack([Qv,QN])
    
        onev=jnp.ones(len(self.nuarr))
    
        TransM=jnp.vstack([onev,TransM])
        F=(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
        F=F*3.e7
   
        return F

    
    @partial(jit, static_argnums=(0,))        
    def add_layer(self,carry,x):
        """adding an atmospheric layer (old)

        Args:
           carry: F[i], P[i], nu0, sigmaD, gammaL
           x: free parameters, T
        
        Returns:
           carry: F[i+1], P[i+1]=k*P[i]
           dtaui: dtau of this layer

        """
        F,Pi,nu0,sigmaD,gammaL = carry        
        Ti = x
        gi = planck.nB(Ti,self.numic)
        numatrix=lpf.make_numatrix(self.nuarr,self.hatnufix,nu0)
        cs=cross(numatrix,sigmaD,gammaL,self.Sfix)
        dtaui = 1.e-1*cs*(1.0-self.k)*Pi # delta P = (1.0-k)*Pi
        Trans=(1.0-dtaui)*jnp.exp(-dtaui)
        F = F*Trans + gi*(1.0-Trans)
        carry=[F,self.k*Pi,nu0,sigmaD,gammaL] #carryover 
        return carry,dtaui

    @partial(jit, static_argnums=(0,))
    def layerscan(self,init):
        """Runnin RT by scanning layers (old)

        Args: 
           init: initial parameters
           Tarr: temperature array        
        
        Returns:
           F: upward flux

        """
        FP,null=(scan(self.add_layer,init,self.Tarr,self.NP))
        return FP[0]*3.e4 #TODO: 

    
@jit
def cross(numatrix,sigmaD,gammaL,S):
    """cross section

    Note:
       This routine was replaced by lpf.xsvector or lpf.xsmatrix. Will be removed.

    Args:
       numatrix: jnp array
                 wavenumber matrix
       sigmaD: float
               sigma parameter in Voigt profile
       gammaL: float
               gamma parameter in Voigt profile
       S: jnp array
          line strength array
    
    Returns:
       cs: cross section

    """
#    cs = jnp.dot(lpf.VoigtTc(numatrix,sigmaD,gammaL).T,S)
    cs = jnp.dot((lpf.voigt(numatrix.flatten(),sigmaD,gammaL)).reshape(jnp.shape(numatrix)).T,S)
    return cs



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
