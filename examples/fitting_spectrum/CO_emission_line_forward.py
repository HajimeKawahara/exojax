#!/usr/bin/env python
from exojax.spec import make_numatrix0, voigt, xsvector
from exojax.spec import rtransfer as rt
from exojax.spec import planck
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural

import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.lax import map, scan


import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('bmh')
#numpyro.set_platform("cpu")
#numpyro.set_platform("gpu")

# ## INPUT MODEL

# In[28]:


# TP
alpha_in=0.02
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr = 1500.*(Parr/Parr[-1])**alpha_in


# In[30]:


#here we used 05_hit12.par in /home/kawahara/exojax/data/CO 
import hapi
hapi.db_begin('/home/kawahara/exojax/data/CO')
# Setting wavenumber bins
N=5000
wav=np.linspace(23000,23500,N,dtype=np.float64)#AA
nus=1.e8/wav[::-1]
nus.dtype, nus[1]-nus[0]


# In[31]:


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


# In[32]:


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


# In[33]:


Tref=296.
#isotope
uniqiso=np.unique(isoid)


# In[34]:


#USE HAPI partition function for T-P
allT=list(np.concatenate([[Tref],Tarr]))
Qr=[]
for iso in uniqiso:
    Qr.append(hapi.partitionSum(5,iso, allT))
Qr=np.array(Qr)
qr=Qr[:,0]/Qr[:,1:].T #Q(Tref)/Q(T)
np.shape(qr) #qr(T, iso)


# In[35]:


#partitioning Q(T) for each line
qt=np.zeros((NP,len(isoid)))
for idx,iso in enumerate(uniqiso):
    mask=isoid==iso
    for ilayer in range(NP):
        qt[ilayer,mask]=qr[ilayer,idx]


# In[36]:


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



numatrix0=make_numatrix0(nus,nu_lines)
xsmatrix=jit(vmap(xsvector,(None,0,0,0)))


# In[39]:



sigv=sigmaDM[0,:]
gamv=gammaLM[0,:]
sv=SijM[0,:]


#xsv=xsvector(numatrix0,sigv,gamv,sv).block_until_ready()


# In[42]:


Nmol=215
nur=500
nuarr=np.linspace(-nur,nur,N,dtype=np.float64)
hatnufix = (np.random.rand(Nmol)-0.5)*nur*2
numatrix=make_numatrix0(nuarr,hatnufix)

sigmaD=3.0*np.ones(Nmol)
gammaL=0.5*np.ones(Nmol)
Sfix=np.random.rand(Nmol)*1.e12
xsigmaD=jnp.array(sigmaD)
xgammaL=jnp.array(gammaL)
xSfix=jnp.array(Sfix)


# In[43]:


#%timeit -n 100 xsv=xsvector(numatrix,xsigmaD,xgammaL,xSfix).block_until_ready()


# In[44]:

import time
ts=time.time()
xsm=xsmatrix(numatrix0,sigmaDM,gammaLM,SijM).block_until_ready()
te=time.time()
print(te-ts,"sec")

import sys
sys.exit()

from scipy.constants import  m_u

g=980.0 # cm/s2
Mmol*m_u*1.e3 #g
tfac=Xco/(Mmol*m_u*g)


# In[50]:


#1 bar = 10^5 Pa = 10^6 dyn/cm2
#Pa 


# In[51]:


numic=1.e4/np.mean(nus)
gi = planck.nB(Tarr,numic)

dtauM=dParr[:,None]*xsm*tfac
TransM=(1.0-dtauM)*jnp.exp(-dtauM)
#QN=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
QN=jnp.zeros(len(nus))
Qv=(1-TransM)*gi[:,None]
Qv=jnp.vstack([Qv,QN])
onev=jnp.ones(len(nus))
TransM=jnp.vstack([onev,TransM])
F=1.e10*(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))


# In[52]:


#for i in range(0,NP):
#    plt.plot(dtauM[i,:])

fig=plt.figure()
ax=fig.add_subplot(111)
c=ax.imshow(np.log10(dtauM),vmin=-5,vmax=3,cmap="magma")
plt.colorbar(c)
ax.set_aspect(0.7/ax.get_data_ratio())


# In[53]:


fig=plt.figure(figsize=(10,5))
plt.plot(wav,F,lw=1)


# In[38]:


sigin=3.0
N=len(F)
data=F+np.random.normal(0,sigin,size=N)

plt.plot(data,".")


# In[39]:


vgamma_hitran=jit(vmap(gamma_hitran,(0,0,0,None,None,None)))
vdoppler_sigma=jit(vmap(doppler_sigma,(None,0,None)))
vSij=jit(vmap(SijT,(0,None,None,None,None,0)))


# In[54]:

#########
#DO NOT CALL NUMPYRO BEFORE YOU NEED

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
import arviz


def model(nu,y):
  #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
  #sD = numpyro.sample('sD', dist.Exponential(1.))
  #gL = numpyro.sample('gL', dist.Exponential(1.))
  sigma = numpyro.sample('sigma', dist.Exponential(4.))
  #nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
  alpha = numpyro.sample('alpha', dist.Uniform(0.01,0.04)) #
  
  #model
  #numatrix0=make_numatrix0(nu,nu_lines)
  Tarr = 1500.*(Parr/Parr[-1])**alpha    
  #line computation
  SijM=vSij(Tarr,logsij0,nu_lines,gpp,elower,qt)
  gammaLM = vgamma_hitran(Parr,Tarr,Parr, n_air, gamma_air, gamma_self)
  #+ gamma_natural(A) #uncomment if you inclide a natural width
  sigmaDM=vdoppler_sigma(nu_lines,Tarr,Mmol)
  
  gi = planck.nB(Tarr,numic)
  xsm=xsmatrix(numatrix0,sigmaDM,gammaLM,SijM)
  
  dtauM=dParr[:,None]*xsm*tfac
  TransM=(1.0-dtauM)*jnp.exp(-dtauM)
  #QN=jnp.zeros(len(nus))
  Qv=(1-TransM)*gi[:,None]
  Qv=jnp.vstack([Qv,QN])
  #onev=jnp.ones(len(nus))
  TransM=jnp.vstack([onev,TransM])
  mu=1.e10*(jnp.sum(Qv*jnp.cumprod(TransM,axis=0),axis=0))
  print(mu)
  #numic=0.5
  #nuarr=nu
  #F0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
  #init=[F0,Parr[0],nu0,sD,gL]
  #FP,null=scan(add_layer,init,Tarr,NP)
  #mu = FP[0]*3.e4
  numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# In[55]:


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 100, 200

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nus, y=data)
mcmc.print_summary()


# In[27]:


refs={}
#refs["A"]=Afix
refs["sD"]=sDfix
refs["gL"]=gLfix
refs["sigma"]=sigin
refs["nu0"]=nu0fix
refs["alpha"]=-0.1
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.show()


# ## LA RT (class)

# In[17]:


jaxrt=rt.JaxRT()
jaxrt.nuarr=nuarr
jaxrt.hatnufix=hatnufix
jaxrt.Sfix=Sfix
jaxrt.Parr=Parr
jaxrt.dParr=dParr
jaxrt.NP=NP
jaxrt.k=k


# In[21]:


#run using flatten() and reshape()
def model(nu,y):
    #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    alpha = numpyro.sample('alpha', dist.Uniform(-0.3,0.3)) #
    
    #model
    Tarr = 1000.*(Parr/Parr[0])**alpha #
    source = planck.nB(Tarr,jaxrt.numic)
    mu=jaxrt.run(nu0,sigmaD,gammaL,source)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# In[20]:


#runx (using vmap)
def model(nu,y):
    #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    alpha = numpyro.sample('alpha', dist.Uniform(-0.3,0.3)) #
    
    #model
    Tarr = 1000.*(Parr/Parr[0])**alpha #
    source = planck.nB(Tarr,jaxrt.numic)
    mu=jaxrt.runx(nu0,sigmaD,gammaL,source)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# In[24]:


np.shape(numatrix)


# In[22]:


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nuarr, y=data)
mcmc.print_summary()


# ## Note (Jan 16 2021): 
# currently runx (using vmap) is ~ 2 times slower than run (using flatten() and reshape)

# In[23]:


refs={}
#refs["A"]=Afix
refs["sD"]=sDfix
refs["gL"]=gLfix
refs["sigma"]=10.0
refs["nu0"]=nu0fix
refs["alpha"]=-0.1
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.show()


# In[23]:


refs={}
#refs["A"]=Afix
refs["sD"]=sDfix
refs["gL"]=gLfix
refs["sigma"]=sigin
refs["nu0"]=nu0fix
refs["alpha"]=-0.1
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.show()


# In[36]:


# generating predictions
# hpdi is "highest posterior density interval"
posterior_sample = mcmc.get_samples()
pred = Predictive(model,posterior_sample)
nu_ = nuarr
predictions = pred(rng_key_,nu=nu_,y=None)
median_mu = jnp.median(predictions["y"],axis=0)
hpdi_mu = hpdi(predictions["y"], 0.9)


# In[37]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
ax.plot(nu_,median_mu,color="C0")
ax.plot(nuarr,data,"+",color="C1",label="data")
ax.fill_between(nu_, hpdi_mu[0], hpdi_mu[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("$\\nu$")
plt.legend()
plt.savefig("../../documents/figures/mcmc_fitting_emission.png")
plt.show()


# ## Layer scan

# In[9]:


jaxrt=rt.JaxRT()
jaxrt.nuarr=nuarr
jaxrt.hatnufix=hatnufix
jaxrt.Sfix=Sfix
jaxrt.Parr=Parr
jaxrt.NP=NP
jaxrt.k=k


# In[10]:


numic=0.5
F0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
init=[F0,Parr[0],0.7,1.0,0.5]
jaxrt.Tarr=Tarr
get_ipython().run_line_magic('timeit', 'jaxrt.layerscan(init)')


# In[11]:


F0=jnp.zeros(len(nuarr))
init=[F0,Parr[0],0.7,1.0,0.5]
FP,tauarr=scan(jaxrt.add_layer,init,Tarr.T,NP)

fig=plt.figure()
ax=fig.add_subplot(111)
c=ax.imshow(tauarr)
plt.colorbar(c,shrink=0.7)
ax.set_aspect(0.7/ax.get_data_ratio())
plt.gca().invert_yaxis()


# In[12]:


from jax import grad
F0=jnp.zeros(len(nuarr))
#F0=0.0
init=[F0,Parr[0],0.7,1.0,0.5]
#scan(add_layer,init,Tarr,NP)


# In[16]:


@jit
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
    FP,null=scan(jaxrt.add_layer,init,Tarr,NP)
    return FP[0]*3.e4

sigin=5.0
data=g(Tarr)+np.random.normal(0,sigin,size=N)
plt.plot(data,".")


# In[17]:


#jaxrt.Tarr=Tarr

#probabilistic model using numpyro
def model(nu,y):
    #A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sD = numpyro.sample('sD', dist.Exponential(1.))
    gL = numpyro.sample('gL', dist.Exponential(1.))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    nu0 = numpyro.sample('nu0', dist.Uniform(-5,5))
    numic=0.5
    nuarr=nu
    F0=jnp.ones(len(nuarr))*planck.nB(Tarr[0],numic)
    init=[F0,Parr[0],nu0,sD,gL]
    jaxrt.Tarr=Tarr
    mu=jaxrt.layerscan(init)
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# In[18]:


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 1000, 2000

kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nuarr, y=data)
mcmc.print_summary()


# In[22]:


import arviz
arviz.plot_trace(mcmc, var_names=["gL", "sD","nu0","sigma"])
plt.show()


# In[24]:


refs={}
#refs["A"]=Afix
refs["sD"]=sDfix
refs["gL"]=gLfix
refs["sigma"]=sigin
refs["nu0"]=nu0fix
refs["alpha"]=-0.1
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.show()


# In[25]:


# generating predictions
# hpdi is "highest posterior density interval"
posterior_sample = mcmc.get_samples()
pred = Predictive(model,posterior_sample)
nu_ = nuarr
predictions = pred(rng_key_,nu=nu_,y=None)
median_mu = jnp.median(predictions["y"],axis=0)
hpdi_mu = hpdi(predictions["y"], 0.9)


# In[26]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
ax.plot(nu_,median_mu,color="C0")
ax.plot(nuarr,data,"+",color="C1",label="data")
ax.fill_between(nu_, hpdi_mu[0], hpdi_mu[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("$\\nu$")
plt.legend()
plt.savefig("../../documents/figures/mcmc_fitting_emission.png")
plt.show()


# In[ ]:




