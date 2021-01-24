#!/usr/bin/env python
# coding: utf-8

# # MCMC Fitting the emission profile to an emission spectrum  
# 
# HMC-NUTS: Hamiltonian Monte Carlo No-U-Turn Sample using numpyro

# In[1]:


# importing lpf modile in exojax.spec
from exojax.spec import rtransfer as rt
from exojax.spec import planck
from exojax.spec import make_numatrix0,xsvector
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import arviz
import numpy as np
import jax.numpy as jnp
from jax import random
from jax.lax import map, scan
from jax import vmap, jit
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

plt.style.use('bmh')
#numpyro.set_platform("cpu")
#numpyro.set_platform("gpu")


# TP
alpha_in=0.02
NP=70
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr = 1500.*(Parr/Parr[-1])**alpha_in

#here we used 05_hit12.par in /home/kawahara/exojax/data/CO 
import hapi
hapi.db_begin('/home/kawahara/exojax/data/CO')
# Setting wavenumber bins
wav=np.linspace(23000,23500,5000,dtype=np.float64)#AA
nus=1.e8/wav[::-1]


molec='05_hit12'
A_all = hapi.getColumn(molec, 'a')
n_air_all = hapi.getColumn(molec, 'n_air')
isoid_all = hapi.getColumn(molec,'local_iso_id')
gamma_air_all = hapi.getColumn(molec, 'gamma_air')
gamma_self_all = hapi.getColumn(molec, 'gamma_self')

nu_lines_all = hapi.getColumn(molec, 'nu')
delta_air_all = hapi.getColumn(molec, 'delta_air')
S_ij_all = hapi.getColumn(molec, 'sw')
elower_all = hapi.getColumn(molec, 'elower')
gpp_all = hapi.getColumn(molec, 'gpp')


margin=10
crit=1.e-98
#crit=1.e-300
#mask=(S_ij_all>crit)
mask=(nu_lines_all>nus[0]-margin)*(nu_lines_all<nus[-1]+margin)*(S_ij_all>crit)
#mask=(nu_lines_all>nus[0]-margin)*(nu_lines_all<nus[-1]+margin)#*(S_ij_all>crit)

A = A_all[mask]
n_air=n_air_all[mask]
isoid=isoid_all[mask]
gamma_air=gamma_air_all[mask]
gamma_self=gamma_self_all[mask] 

nu_lines=nu_lines_all[mask] 
delta_air=delta_air_all[mask]
S_ij0=S_ij_all[mask]
elower=elower_all[mask]
gpp=gpp_all[mask]

logsij0=np.log(S_ij0) #use numpy not jnp


Tref=296.
#isotope
uniqiso=np.unique(isoid)


#USE HAPI partition function for T-P
allT=list(np.concatenate([[Tref],Tarr]))
Qr=[]
for iso in uniqiso:
    Qr.append(hapi.partitionSum(5,iso, allT))
Qr=np.array(Qr)
qr=Qr[:,0]/Qr[:,1:].T #Q(Tref)/Q(T)
np.shape(qr) #qr(T, iso)

#partitioning Q(T) for each line
qt=np.zeros((NP,len(isoid)))
for idx,iso in enumerate(uniqiso):
    mask=isoid==iso
    for ilayer in range(NP):
        qt[ilayer,mask]=qr[ilayer,idx]

#Mmol=28.010446441149536
Mmol=3.86 #mean molecular weight
Xco=0.065 #mixing ratio
Tref=296.0
bar2atm=1.0/1.01325
#Pfix=1.e-3*bar2atm#atm

SijM=vmap(SijT,(0,None,None,None,None,0))(Tarr,logsij0,nu_lines,gpp,elower,qt)
gammaLM = vmap(gamma_hitran,(0,0,0,None,None,None))(Parr,Tarr,Parr, n_air, gamma_air, gamma_self)
#+ gamma_natural(A) #uncomment if you inclide a natural width
sigmaDM=vmap(doppler_sigma,(None,0,None))(nu_lines,Tarr,Mmol)

numatrix=make_numatrix0(nus,nu_lines)
xsmatrix=jit(vmap(xsvector,(None,0,0,0)))

sigv=sigmaDM[0,:]
gamv=gammaLM[0,:]
sv=SijM[0,:]

jxsvector=jit(xsvector)

import time 
ts=time.time()
xsv=jxsvector(numatrix,sigv,gamv,sv).block_until_ready()
te=time.time()
print(te-ts,"sec")

# In[111]:

ts=time.time()
xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM).block_until_ready()
te=time.time()
print(te-ts,"sec")
