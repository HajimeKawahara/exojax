import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import vmap, jit
import matplotlib.pyplot as plt
from exojax.dynamics.rvfunc import rvf
from exojax.dynamics.getE import getE

##
N=100
t=np.random.rand(N)*100
T0=0.0
P=10.0
e=0.3
omegaA=0.5
Ksini=10.0
Vsys=5.0
model=rvf(t,T0,P,e,omegaA,Ksini,Vsys)
sigma=3.0
noise=np.random.normal(0.0,sigma,N)
rv=model+noise
err=sigma*np.ones(N)/2.0

import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

def model_c(t1,y1,e1):
    P=numpyro.sample("P", dist.Uniform(8.0,12.0))
    Ksini=numpyro.sample('Ksini', dist.Exponential(0.1)) #should be modified Jeffery later
    T0 = numpyro.sample('T0', dist.Uniform(-6.0,6.0))
    sesinw = numpyro.sample('sesinw', dist.Uniform(-1.0,1.0))
    secosw = numpyro.sample('secosw', dist.Uniform(-1.0,1.0))
    etmp=sesinw**2+secosw**2
    e=jnp.where(etmp>1.0,1.0,etmp)
    omegaA=jnp.arctan2(sesinw,secosw) #
#    sigmajit=numpyro.sample('sigmajit', dist.Uniform(0.1,100.0))
    sigmajit=numpyro.sample('sigmajit', dist.Exponential(1.0))
    Vsys = numpyro.sample('Vsys', dist.Uniform(-10,10.0))
    mu=rvf(t1,T0,P,e,omegaA,Ksini,Vsys)
    errall=jnp.sqrt(e1**2+sigmajit**2)
    numpyro.sample("y1", dist.Normal(mu, errall), obs=y1) #-
        

#Running a HMC-NUTS
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000
kernel = NUTS(model_c)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, t1=t, y1=rv, e1=err)
mcmc.print_summary()
print("end HMC")

#Post-processing
posterior_sample = mcmc.get_samples()
np.savez("savepos.npz",[posterior_sample])

