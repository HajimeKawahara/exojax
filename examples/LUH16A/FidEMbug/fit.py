# Basic modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# JAX
import jax.numpy as jnp
from jax import random

# ExoJAX
from exojax.spec import initspec, planck, moldb, contdb, response, molinfo
from exojax.spec.lpf import xsvector, xsmatrix, exomol
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, wavenumber_grid, pressure_layer
from exojax.plot.atmplot import  plot_maxpoint
from exojax.spec.evalline import reduceline_exomol
from exojax.spec.limb_darkening import ld_kipping
from exojax.utils.astrofunc import getjov_gravity
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.constants import RJ, pc
from exojax.utils.gpkernel import gpkernel_RBF

# FLUX reference
Fabs_REF2=2.7e-12 #absolute flux (i.e. flux@10pc) erg/s/cm2/um Burgasser+ 1303.7283 @2.29um
fac0=RJ**2/((10.0*pc)**2)  #nomralize by RJ
Fref=(2.29**2)*Fabs_REF2/fac0/1.e4 #erg/cm2/s/cm-1 @ 2.3um

# Loading spectrum
dat=pd.read_csv("../data/luhman16a_spectra_detector1.csv",delimiter=",")
wavd=(dat["wavelength_micron"].values)*1.e4 #AA
nusd=1.e8/wavd[::-1]
fobs=(dat["normalized_flux"].values)[::-1]
err=(dat["err_normalized_flux"].values)[::-1]

# ATMOSPHERIC LAYER
Pref=1.0 # Reference pressure for a T-P model (bar)
NP=100
Parr, dParr, k=pressure_layer(NP=NP)
mmw=2.33 #mean molecular weight
ONEARR=np.ones_like(Parr) #ones_array for MMR
molmassCO=molinfo.mean_molmass("CO") #molecular mass (CO)
molmassH2O=molinfo.mean_molmass("H2O") #molecular mass (H2O)

# Instrument
beta=resolution_to_gaussian_std(100000.) #std of gaussian from R=100000.

# Loading Molecular datanase and  Reducing Molecular Lines
Nx=4500    # number of wavenumber bins (nugrid) for fit
ws=22876.0 # AA
we=23010.0 # AA
nus,wav,res=nugrid(ws-5.0,we+5.0,Nx,unit="AA")

# Masking data
mask=(ws<wavd[::-1])*(wavd[::-1]<we) # data fitting range
mask=mask*((22898.5>wavd[::-1])+(wavd[::-1]>22899.5))  # Additional mask to remove a strong telluric
fobsx=fobs[mask]
nusdx=nusd[mask]
wavdx=1.e8/nusdx[::-1]
errx=err[mask]

# Loading molecular database 
mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) 
mdbH2O=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-46) 

# LOADING CIA
mmrH2=0.74
mmrHe=0.25
molmassH2=molinfo.mean_molmass("H2")
molmassHe=molinfo.mean_molmass("He")
vmrH2=(mmrH2*mmw/molmassH2)
vmrHe=(mmrHe*mmw/molmassHe)
cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)

# Reducing Molecular Lines
def Tmodel(Parr,T0):
    """ Constant T model
    """
    return T0*np.ones_like(Parr)

# Reference physical quantities
g=10**(5.0)
maxMMR_CO=0.01
maxMMR_H2O=0.005

# CO 
mask_CO,maxcf,maxcia=reduceline_exomol(mdbCO,Parr,dParr,mmw,g,vmrH2,cdbH2H2,maxMMR_CO,molmassCO,Tmodel,[1700.0]) #only 1700K
plot_maxpoint(mask_CO,Parr,maxcf,maxcia,mol="CO")
plt.savefig("maxpoint_CO.pdf", bbox_inches="tight", pad_inches=0.0)

# H2O
T0xarr=list(range(500,1800,100))
mask_H2O,maxcf,maxcia=reduceline_exomol(mdbH2O,Parr,dParr,mmw,g,vmrH2,cdbH2H2,maxMMR_H2O,molmassH2O,Tmodel,T0xarr) #only 1700K
plot_maxpoint(mask_H2O,Parr,maxcf,maxcia,mol="H2O")
plt.savefig("maxpoint_H2O.pdf", bbox_inches="tight", pad_inches=0.0)

# Initialization of direct LPF
numatrix_CO=initspec.init_lpf(mdbCO.nu_lines,nus)    
numatrix_H2O=initspec.init_lpf(mdbH2O.nu_lines,nus)

# HMC-NUTS FITTING PART
from numpyro import sample
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

# Some constants for fitting
baseline=1.07 #(baseline for a CIA photosphere in the observed (normaized) spectrum)
maxMMR_CO=0.01
maxMMR_H2O=0.005

# Model
def model_c(nu1,y1,e1):
    Rp = sample('Rp', dist.Uniform(0.5,1.5))
    Mp = sample('Mp', dist.Normal(33.5,0.3))
    RV = sample('RV', dist.Uniform(26.0,30.0))
    MMR_CO = sample('MMR_CO', dist.Uniform(0.0,maxMMR_CO))
    MMR_H2O = sample('MMR_H2O', dist.Uniform(0.0,maxMMR_H2O))
    T0 = sample('T0', dist.Uniform(1000.0,1700.0))
    alpha = sample('alpha', dist.Uniform(0.05,0.15))
    vsini = sample('vsini', dist.Uniform(10.0,20.0))    

    # Kipping Limb Darkening Prior
    q1 = sample('q1', dist.Uniform(0.0,1.0))
    q2 = sample('q2', dist.Uniform(0.0,1.0))
    u1,u2=ld_kipping(q1,q2)
    
    #GP
    logtau = sample('logtau', dist.Uniform(-1.5,0.5)) #tau=1 <=> 5A
    tau=10**(logtau)
    loga = sample('loga', dist.Uniform(-4.0,-2.0))
    a=10**(loga)

    #gravity
    g=getjov_gravity(Rp,Mp)
        
    #T-P model//
    Tarr = T0*(Parr/Pref)**alpha 
        
    #CO
    SijM_CO,gammaLM_CO,sigmaDM_CO=exomol(mdbCO,Tarr,Parr,molmassCO)
    xsm_CO=xsmatrix(numatrix_CO,sigmaDM_CO,gammaLM_CO,SijM_CO) 
    dtaumCO=dtauM(dParr,xsm_CO,MMR_CO*ONEARR,molmassCO,g)
    
    #H2O
    SijM_H2O,gammaLM_H2O,sigmaDM_H2O=exomol(mdbH2O,Tarr,Parr,molmassH2O)
    xsm_H2O=xsmatrix(numatrix_H2O,sigmaDM_H2O,gammaLM_H2O,SijM_H2O) 
    dtaumH2O=dtauM(dParr,xsm_H2O,MMR_H2O*ONEARR,molmassH2O,g)
    
    #CIA
    dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,\
                      mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
    dtaucH2He=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrHe,\
                      mmw,g,cdbH2He.nucia,cdbH2He.tcia,cdbH2He.logac)
    
    dtau=dtaumCO+dtaumH2O+dtaucH2H2+dtaucH2He    
    sourcef = planck.piBarr(Tarr,nus)
    Ftoa=Fref/Rp**2
    F0=rtrun(dtau,sourcef)/baseline/Ftoa
    
    Frot=response.rigidrot(nus,F0,vsini,u1,u2)
    mu=response.ipgauss_sampling(nu1,nus,Frot,beta,RV)
    cov=gpkernel_RBF(nu1,tau,a,e1)
    sample("y1", dist.MultivariateNormal(loc=mu, covariance_matrix=cov), obs=y1)
        
#Running a HMC-NUTS
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 500, 1000
kernel = NUTS(model_c,forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
mcmc.run(rng_key_, nu1=nusdx, y1=fobsx, e1=errx)
print("end HMC")

# Post-processing
posterior_sample = mcmc.get_samples()
np.savez("npz/savepos.npz",[posterior_sample])

pred = Predictive(model_c,posterior_sample,return_sites=["y1"])
nu = nus
predictions = pred(rng_key_,nu1=nu,y1=None,e1=errx)
median_mu = jnp.median(predictions["y1"],axis=0)
hpdi_mu = hpdi(predictions["y1"], 0.9)
np.savez("npz/saveplotpred.npz",[wavdx,fobsx,errx,median_mu,hpdi_mu])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6.0))
ax.plot(wavdx[::-1],median_mu,color="C0")
ax.plot(wavdx[::-1],fobsx,"+",color="C1",label="data")

# Annotation for some lines
red=(1.0+28.07/300000.0) #for annotation
ax.plot([22913.3*red,22913.3*red],[0.6,0.75],color="C0",lw=1)
ax.plot([22918.07*red,22918.07*red],[0.6,0.77],color="C1",lw=1)
ax.plot([22955.67*red,22955.67*red],[0.6,0.68],color="C2",lw=1)
plt.text(22913.3*red,0.55,"A",color="C0",fontsize=12,horizontalalignment="center")
plt.text(22918.07*red,0.55,"B",color="C1",fontsize=12,horizontalalignment="center")
plt.text(22955.67*red,0.55,"C",color="C2",fontsize=12,horizontalalignment="center")
ax.fill_between(wavdx[::-1], hpdi_mu[0], hpdi_mu[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("wavelength ($\AA$)",fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.savefig("npz/results.pdf", bbox_inches="tight", pad_inches=0.0)
plt.savefig("npz/results.png", bbox_inches="tight", pad_inches=0.0)

# ARVIZ part
import arviz
rc = {
    "plot.max_subplots": 1024,
}

try:
    arviz.rcParams.update(rc)
    arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
    plt.savefig("npz/cornerall.png")
except:
    print("failed corner")

try:
    pararr=["Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","vsini","RV","q1","q2","logtau","loga"]
    arviz.plot_trace(mcmc, var_names=pararr)
    plt.savefig("npz/trace.png")
except:
    print("failed trace")

    
