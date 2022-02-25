# Example of RV curve fitting
from numpyro.diagnostics import hpdi
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
import pandas as pd
import numpy as np
import jax.numpy as jnp
import tqdm
from jax import random
from jax import vmap, jit
import matplotlib.pyplot as plt
from exojax.dynamics.rvfunc import rvf
from exojax.dynamics.getE import getE

##
N = 100
t = np.random.rand(N)*100
T0 = 0.0
P = 10.0
e = 0.3
omegaA = 0.5
Ksini = 10.0
Vsys = 5.0
model = rvf(t, T0, P, e, omegaA, Ksini, Vsys)
sigma = 3.0
np.random.seed(1)
noise = np.random.normal(0.0, sigma, N)
rv = model+noise
err = sigma*np.ones(N)/2.0


def model_c(t1, y1, e1):
    P = numpyro.sample('P', dist.Uniform(8.0, 12.0))
    # should be modified Jeffery later
    Ksini = numpyro.sample('Ksini', dist.Exponential(0.1))
    T0 = numpyro.sample('T0', dist.Uniform(-6.0, 6.0))
    sesinw = numpyro.sample('sesinw', dist.Uniform(-1.0, 1.0))
    secosw = numpyro.sample('secosw', dist.Uniform(-1.0, 1.0))
    etmp = sesinw**2+secosw**2
    e = jnp.where(etmp > 1.0, 1.0, etmp)
    omegaA = jnp.arctan2(sesinw, secosw)
#    sigmajit=numpyro.sample('sigmajit', dist.Uniform(0.1,100.0))
    sigmajit = numpyro.sample('sigmajit', dist.Exponential(1.0))
    Vsys = numpyro.sample('Vsys', dist.Uniform(-10, 10.0))
    mu = rvf(t1, T0, P, e, omegaA, Ksini, Vsys)
    errall = jnp.sqrt(e1**2+sigmajit**2)
    numpyro.sample('y1', dist.Normal(mu, errall), obs=y1)  # -


# Running a HMC-NUTS
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 2000, 4000
kernel = NUTS(model_c)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, t1=t, y1=rv, e1=err)
mcmc.print_summary()
print('end HMC')

# Post-processing
posterior_sample = mcmc.get_samples()
np.savez('savepos.npz', [posterior_sample])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
ax.errorbar(t, rv, yerr=err, ls='none')
ax.plot(t, rv, 'o')

sesinw = posterior_sample['sesinw']
secosw = posterior_sample['secosw']
eps = sesinw**2+secosw**2
omegaAps = jnp.arctan2(sesinw, secosw)

tpre = jnp.linspace(np.min(t), np.max(t), 3600)
for i in tqdm.tqdm(range(0, len(posterior_sample['P'][::10]))):
    e = eps[i]
    T0 = posterior_sample['T0'][i]
    P = posterior_sample['P'][i]
    omegaA = omegaAps[i]
    Ksini = posterior_sample['Ksini'][i]
    Vsys = posterior_sample['Vsys'][i]
    model = rvf(tpre, T0, P, e, omegaA, Ksini, Vsys)
    ax.plot(tpre, model, alpha=0.05, color='gray')

plt.savefig('npz/results.png', bbox_inches='tight', pad_inches=0.0)
plt.show()
