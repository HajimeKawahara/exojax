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

plt.style.use('bmh')
#numpyro.set_platform("cpu")
#numpyro.set_platform("gpu")


# In[3]:


NP=17
Parr, k=rt.const_p_layer(NP=NP)
Tarr = 1000.*(Parr/Parr[0])**-0.1

N=50
nuarr=jnp.linspace(-10,10,N)
def add_layer(carry,x):
    """
    Params:
      carry: F[i], P[i]
      x: free parameters, T
      
    Returns:
      FP: F[i+1], P[i+1]=k*P[i]
      dtaui: dtau of this layer
    """
    F,Pi,nu0,sigmaD,gammaL = carry
    Ti = x
    numic=0.5
    #dP = k*Pi
    gi = planck.nB(Ti,numic)
    #dtaui = 1.e-1*lpf.VoigtTc(nuarr-nu0,sigmaD,gammaL)*(1.0-k)*Pi
    dtaui = 1.e-1*lpf.VoigtHjert(nuarr-nu0,sigmaD,gammaL)*(1.0-k)*Pi
    Trans=(1.0-dtaui)*jnp.exp(-dtaui)
    F = F*Trans + gi*(1.0-Trans)
    carry=[F,k*Pi,nu0,sigmaD,gammaL] #carryover 
    return carry,dtaui



def g(xs):
    """
    Params: 
      xs: free parameters
    """
    Tarr=xs
    numic=0.5
    F0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
    #F0=0.0
    init=[F0,Parr[0],0.7,1.0,0.5]
    FP,null=scan(add_layer,init,Tarr,NP)
    return FP[0]*3.e4


# In[12]:


F0=jnp.zeros(len(nuarr))
init=[F0,Parr[0],0.7,1.0,0.5]
FP,tauarr=scan(add_layer,init,Tarr.T,NP)


from jax import grad
F0=jnp.zeros(len(nuarr))
#F0=0.0
init=[F0,Parr[0],0.7,1.0,0.5]
#scan(add_layer,init,Tarr,NP)


# In[15]:


sigin=0.8
data=g(Tarr)+np.random.normal(0,sigin,size=N)

#probabilistic model using numpyro
def model(nu,y):
    #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    alpha = numpyro.sample('alpha', dist.Uniform(-0.3,0.3)) #
    Tarr = 1000.*(Parr/Parr[0])**alpha #
    numic=0.5
    nuarr=nu
    F0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
    init=[F0,Parr[0],nu0,sD,gL]
    FP,null=scan(add_layer,init,Tarr,NP)
    mu = FP[0]*3.e4
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# In[18]:


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nuarr, y=data)
mcmc.print_summary()



