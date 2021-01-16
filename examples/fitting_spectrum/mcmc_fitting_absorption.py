#!/usr/bin/env python
# coding: utf-8

# # MCMC Fitting the absorption profile to an absorption line  
# 
# This example conducts HMC-NUTS fitting to a mock absorption line. We use the absorption model based on Tepper 
# approximation of Voigt profile, lpf.FAbsVRewofz in exojax.lpf (line profile functions) module. 
# 
# HMC-NUTS: Hamiltonian Monte Carlo No-U-Turn Sample using numpyro

# In[1]:


# importing lpf modile in exojax.spec
from exojax.spec import lpf
from exojax.spec import absorption


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import arviz

import numpy as np
import jax.numpy as jnp
from jax import random
from jax.lax import map

import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

plt.style.use('bmh')
#numpyro.set_platform("cpu")
#numpyro.set_platform("gpu")


# In[4]:


# generating mock absorption data
np.random.seed(34)
N=20
nuarr=jnp.linspace(-10,10,N)
sigin=0.01
sDfix = jnp.array(1.0)
gLfix = jnp.array(0.5)
Afix = jnp.array(1.0)
nu0fix = 0.7
data=lpf.FAbsVHjert(nuarr-nu0fix,sDfix,gLfix,Afix)+np.random.normal(0,sigin,size=N)


#probabilistic model using numpyro
def model(nu,y):
    A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    mu = lpf.FAbsVHjert(nu-nu0,sD,gL,A)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

# OK, a HMC-NUTS!
# Our model, lpf.FAbsVRewofz, is compatible to jax. Autograd works. We can perform a HMC-NUTS.
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nuarr, y=data)
mcmc.print_summary()
import sys
sys.exit()

# In[21]:


# In[11]:


#cool and flexible corner plot in arviz. You can also try kind='hexbin' instead of 'kde' for instance.
#arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=True)
refs={}
refs["A"]=Afix
refs["sD"]=sDfix
refs["gL"]=gLfix
refs["sigma"]=sigin
refs["nu0"]=nu0fix
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.savefig("corner.png")

# generating predictions
# hpdi is "highest posterior density interval"
posterior_sample = mcmc.get_samples()
pred = Predictive(model,posterior_sample)
nu_ = jnp.linspace(-12,12,100)
predictions = pred(rng_key_,nu=nu_,y=None)
median_mu = jnp.median(predictions["y"],axis=0)
hpdi_mu = hpdi(predictions["y"], 0.9)


# In[14]:


# final plot of median and 90% credible area of the prediction

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
ax.plot(nu_,median_mu,color="C0")
ax.plot(nuarr,data,"+",color="C1",label="data")
ax.fill_between(nu_, hpdi_mu[0], hpdi_mu[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("$\\nu$")
plt.legend()
plt.savefig("../../documents/figures/mcmc_fitting_absorption.png")

