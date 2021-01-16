#!/usr/bin/env python
# coding: utf-8

# # MCMC Fitting the absorption profile to an absorption line  
# 
# This example conducts HMC-NUTS fitting to a mock absorption line. We use the absorption model based on Tepper 
# approximation of Voigt profile, lpf.FAbsVTc in exojax.lpf (line profile functions) module. 
# 
# HMC-NUTS: Hamiltonian Monte Carlo No-U-Turn Sample using numpyro
# importing lpf and absorption modiles in exojax.spec

from exojax.spec import lpf
from exojax.spec import absorption

import seaborn as sns
import matplotlib.pyplot as plt
import arviz

import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jit
from jax.lax import map

import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

plt.style.use('bmh')
#numpyro.set_platform("cpu")
numpyro.set_platform("gpu")

# generating mock absorption data
np.random.seed(38)
N=1000
nur=1000
nuarr=jnp.linspace(-nur,nur,N)
sigin=0.01
sDfix = jnp.array(1.0)
gLfix = jnp.array(0.5)

Nmol=100
hatnufix = (np.random.rand(Nmol)-0.5)*nur*2
Sfix=np.random.rand(Nmol)
Afix=jnp.array(0.03)
nu0fix = 0.7
#f = lambda nu: lpf.MultiAbsVTc(nu-nu0fix,sDfix,gLfix,Afix,Sfix,hatnufix)

numatrix=lpf.make_numatrix(nuarr,hatnufix,nu0fix)
spec=absorption.MultiAbsVHjert(numatrix,sDfix,gLfix,Afix,Sfix)
data=spec+np.random.normal(0,sigin,size=N)



# In[20]:


#probabilistic model using numpyro
def model(nu,y):
    A = numpyro.sample('A', dist.Uniform(0.0,1.0))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    #numatrix=lpf.make_numatrix(nu,hatnufix,nu0)
    numatrix=nu-nu0
    mu = absorption.MultiAbsVHjert(numatrix,sD,gL,A,Sfix)
    #mu = lpf.MultiAbsVTc(nu-nu0,sD,gL,A,Sfix,hatnufix)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=lpf.make_numatrix(nuarr,hatnufix,0), y=data)
mcmc.print_summary()

