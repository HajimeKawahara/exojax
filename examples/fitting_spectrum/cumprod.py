#!/usr/bin/env python
# coding: utf-8

# # MCMC Fitting the emission profile to an emission spectrum  
# 
# This example conducts HMC-NUTS fitting to a mock absorption line. We use Schwarzchild equation of the absorption model based on Tepper 
# approximation of Voigt profile, lpf.FAbsVTc in exojax.lpf (line profile functions) module. 
# 
# HMC-NUTS: Hamiltonian Monte Carlo No-U-Turn Sample using numpyro

# In[16]:


# importing lpf modile in exojax.spec
from exojax.spec import lpf,planck
from exojax.spec import rtransfer as rt

import seaborn as sns
import matplotlib.pyplot as plt
import arviz

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.lax import map, scan
from jax import vmap
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

#layers
NP=30
Parr, k=rt.const_p_layer(NP=NP)
dParr = (1.0-k)*Parr
Parr=Parr[::-1] ### define from upper layer
dParr=dParr[::-1]
Tarr = 1000.*(Parr/Parr[0])**-0.1

#nu
N=50
nuarr=jnp.linspace(-10,10,N)

#data
numic=0.5
gi = planck.nB(Tarr,numic)
dParr = (1.0-k)*Parr
nu0=0.7
sigmaD=1.0
gammaL=0.5
xsv=1.e-1*lpf.VoigtHjert(nuarr-nu0,sigmaD,gammaL)
dtauM=dParr[None,:]*xsv[:,None]
TransM=(1.0-dtauM)*jnp.exp(-dtauM)
#Q0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
Qv=(1-TransM)*gi
F=(jnp.sum(Qv*jnp.cumprod(TransM,axis=1),axis=1))
F=F*3.e7

sigin=0.8
data=F+np.random.normal(0,sigin,size=N)

#probabilistic model using numpyro
def model(nu,y):
    #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    alpha = numpyro.sample('alpha', dist.Uniform(-0.3,0.3)) #

    #model
    Tarr = 1000.*(Parr/Parr[0])**alpha #
    numic=0.5
    gi = planck.nB(Tarr,numic)
    xsv=1.e-1*lpf.VoigtHjert(nuarr-nu0,sD,gL)
    dtauM=dParr[None,:]*xsv[:,None]
    TransM=(1.0-dtauM)*jnp.exp(-dtauM)
    Qv=(1-TransM)*gi
    F=(jnp.sum(Qv*jnp.cumprod(TransM,axis=1),axis=1))
    mu=F*3.e7
    
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nuarr, y=data)
mcmc.print_summary()



