#!/usr/bin/env python
from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo
from exojax.spec import make_numatrix0,xsvector
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.plot.atmplot import plottau, plotcf
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import vmap, jit

#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#grid for F0
N=1000
wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
nus=1.e8/wav[::-1]

#grid for F
M=1000
wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA        
nusd=1.e8/wavd[::-1]

#dv matrix
c=299792.458
dvmat=jnp.array(c*np.log(nusd[:,None]/nus[None,:]))

#Atmospheric parameters
alpha_in=0.02
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr = 1500.*(Parr/Parr[-1])**alpha_in
mmw=2.33 #mean molecular weight
molmassCO=molinfo.molmass("CO") #molecular mass (CO)
MMR=0.01*np.ones_like(Tarr) #mass mixing ratio
g=1.e5 # gravity cm/s2

#Macro response model
RV=0.0
vsini_in=15.0 #rotational vsini
beta=3.0 #IP sigma

#--------------------------------------------------
#ExoMol 

mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) #loading molecular database 

nu0=mdbCO.nu_lines
qt=vmap(mdbCO.qr_interp)(Tarr)
gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
    (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
gammaLMN=gamma_natural(mdbCO.A)
gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
SijM=jit(vmap(SijT,(0,None,None,None,0)))\
    (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
    (mdbCO.nu_lines,Tarr,molmassCO)
numatrix=make_numatrix0(nus,nu0)
xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM)

dtaumCO=dtauM(dParr,xsm,MMR,molmassCO,g)
#plottau(nus,dtauMx,Tarr,Parr,unit="AA") #tau
#plotcf(nus,dtauMx,Tarr,Parr,dParr,unit="AA")

#--------------------------------------------------
#CIA
mmrH2=0.74
mmrHe=0.25
molmassH2=molinfo.molmass("H2")
molmassHe=molinfo.molmass("He")
vmrH2=(mmrH2*mmw/molmassH2)
vmrHe=(mmrHe*mmw/molmassHe)
##H2-H2
#cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
#dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,\
#              mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
##H2-He
#cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)
#dtaucH2He=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrHe,\
#              mmw,g,cdbH2He.nucia,cdbH2He.tcia,cdbH2He.logac)

#------------------------------------------------------
#Running Radiative Transfer
dtau=dtaumCO#+dtaucH2H2+dtaucH2He
sourcef=planck.piBarr(Tarr,nus)
F0=rtrun(dtau,sourcef)

#Applying a Response and Noise
F=response.response(dvmat,F0,vsini_in,beta,RV)

intfac=1.e7
sigin=0.25
data=F*intfac+np.random.normal(0,sigin,size=M)

#------------------------------------------------------
#some figures for checking
fig=plt.figure(figsize=(20,6.0))
plt.plot(wav[::-1],F0,lw=1,color="C1",alpha=0.5,label="F0")
plt.plot(wavd[::-1],F,lw=1,color="C2",label="F")
plt.savefig("fig/highredCO.png")
plt.clf()

plt.plot(data,".")
plt.savefig("fig/data.png")

#######################################################
#HMC-NUTS FITTING PART
#######################################################

import arviz
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi


#Model
def model_c(nu,y):
    A = numpyro.sample('A', dist.Uniform(0.5,1.5))
    sigma = numpyro.sample('sigma', dist.Exponential(0.3))
#    nu0 = numpyro.sample('nu0', dist.Uniform(-0.3,0.3))
    alpha = numpyro.sample('alpha', dist.Uniform(0.019,0.021))
    vsini = numpyro.sample('vsini', dist.Uniform(1.0,30.0))
    #T-P model
    Tarr = 1500.*(Parr/Parr[-1])**alpha 
    
    #line computation
    qt=vmap(mdbCO.qr_interp)(Tarr)
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCO.logsij0,mdbCO.dev_nu_lines,mdbCO.elower,qt)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
        (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
        (mdbCO.dev_nu_lines,Tarr,molmassCO)
    sourcef = planck.piBarr(Tarr,nus)
    
    xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM) 
    dtaumCO=dtauM(dParr,xsm,MMR,mmw,g)
    #dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,\
    #          mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
    #dtaucH2He=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrHe,\
    #          mmw,g,cdbH2He.nucia,cdbH2He.tcia,cdbH2He.logac)
    dtau=dtaumCO#+dtaucH2H2+dtaucH2He

    F0=rtrun(dtau,sourcef)
    mu=response.response(dvmat,F0,vsini,beta,RV)
    mu=intfac*A*mu
    numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

#--------------------------------------------------------
#Running a HMC-NUTS

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 100, 200

kernel = NUTS(model_c,forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu=nus, y=data)

#mcmc.print_summary()

#--------------------------------------------------------
#Post-processing

posterior_sample = mcmc.get_samples()
pred = Predictive(model_c,posterior_sample)
nu_ = nus
predictions = pred(rng_key_,nu=nu_,y=None)
median_mu = jnp.median(predictions["y"],axis=0)
hpdi_mu = hpdi(predictions["y"], 0.9)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6.0))
#plt.plot(wav[::-1],Fx0,lw=1,color="C1",label="in")
ax.plot(wav[::-1],median_mu,color="C0")
ax.plot(wav[::-1],data,"+",color="C1",label="data")
ax.fill_between(wav[::-1], hpdi_mu[0], hpdi_mu[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("wavelength ($\AA$)",fontsize=16)
plt.legend()
plt.savefig("fig/results.png")
plt.show()

arviz.plot_trace(mcmc, var_names=["A","sigma","alpha","vsini"])
#arviz.plot_trace(mcmc, var_names=["A","sigma","nu0","alpha","vsini"])
plt.savefig("fig/trace.png")


refs={}
refs["A"]=1.0
refs["sigma"]=sigin
#refs["nu0"]=0.0
refs["alpha"]=0.02
refs["vsini"]=vsini_in
#refs["alpha"]=-0.1
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,                marginals=True,
                reference_values=refs,
               reference_values_kwargs={'color':"red", "marker":"o", "markersize":12}) 
plt.savefig("fig/corner.png")

