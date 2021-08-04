import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import jax.numpy as jnp
from jax import random
from jax import vmap, jit

from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo, make_numatrix0
from exojax.spec.lpf import xsvector
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
from exojax.plot.atmplot import plottau, plotcf, plot_maxpoint
from exojax.utils.afunc import getjov_logg
from exojax.utils.constants import RJ, pc, Rs, c
from exojax.spec.evalline import mask_weakline

#reference pressure for a T-P model
Pref=1.0 #bar

#FLUX reference
Fabs_REF2=2.7e-12 #absolute flux (i.e. flux@10pc) erg/s/cm2/um Burgasser+ 1303.7283 @2.29um
fac0=RJ**2/((10.0*pc)**2)  #nomralize by RJ
Fref=(2.29**2)*Fabs_REF2/fac0/1.e4 #erg/cm2/s/cm-1 @ 2.3um

#loading spectrum
dat=pd.read_csv("../data/luhman16a_spectra_detector1.csv",delimiter=",")
wavd=(dat["wavelength_micron"].values)*1.e4 #AA
nusd=1.e8/wavd[::-1]
fobs=(dat["normalized_flux"].values)[::-1]
err=(dat["err_normalized_flux"].values)[::-1]

#ATMOSPHERE
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
mmw=2.33 #mean molecular weight
R=100000.
beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R) #IP sigma need check
ONEARR=np.ones_like(Parr) #ones_array for MMR
molmassCO=molinfo.molmass("CO") #molecular mass (CO)
molmassH2O=molinfo.molmass("H2O") #molecular mass (H2O)

#LOADING CIA
mmrH2=0.74
mmrHe=0.25
molmassH2=molinfo.molmass("H2")
molmassHe=molinfo.molmass("He")
vmrH2=(mmrH2*mmw/molmassH2)
vmrHe=(mmrHe*mmw/molmassHe)

#LINES
g=10**(5.0)
T0c=1700.0
Tarr = T0c*np.ones_like(Parr)    
maxMMR_CO=0.01
maxMMR_H2O=0.005


###########################################################
#Loading Molecular datanase and  Reducing Molecular Lines
###########################################################

def ap(fobs,nusd,ws,we,Nx):
    mask=(ws<wavd[::-1])*(wavd[::-1]<we)
    #additional mask to remove a strong telluric
    mask=mask*((22898.5>wavd[::-1])+(wavd[::-1]>22899.5))  
    fobsx=fobs[mask]
    nusdx=nusd[mask]
    wavdx=1.e8/nusdx[::-1]
    errx=err[mask]
    nus,wav,res=nugrid(ws-5.0,we+5.0,Nx,unit="AA")
    #loading molecular database 
    mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) 
    mdbH2O=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-46) 
    #LOADING CIA
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)

    #REDUCING UNNECESSARY LINES
    #1. CO
    Tarr = T0c*np.ones_like(Parr)    
    qt=vmap(mdbCO.qr_interp)(Tarr)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
        (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP+gammaLMN[None,:]
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
        (mdbCO.nu_lines,Tarr,molmassCO)        
    mask_CO,maxcf,maxcia=mask_weakline(mdbCO,Parr,dParr,Tarr,SijM,gammaLM,sigmaDM,maxMMR_CO*ONEARR,molmassCO,mmw,g,vmrH2,cdbH2H2)
    mdbCO.masking(mask_CO)

    plot_maxpoint(mask_CO,Parr,maxcf,maxcia,mol="CO")
    plt.savefig("maxpoint_CO.pdf", bbox_inches="tight", pad_inches=0.0)
        
    #2. H2O
    T0xarr=list(range(500,1800,100))
    for k,T0x in enumerate(T0xarr):
        Tarr = T0x*np.ones_like(Parr)    
        qt=vmap(mdbH2O.qr_interp)(Tarr)
        gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbH2O.n_Texp,mdbH2O.alpha_ref)
        gammaLMN=gamma_natural(mdbH2O.A)
        gammaLM=gammaLMP+gammaLMN[None,:]
        SijM=jit(vmap(SijT,(0,None,None,None,0)))\
            (Tarr,mdbH2O.logsij0,mdbH2O.nu_lines,mdbH2O.elower,qt)
        sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbH2O.nu_lines,Tarr,molmassH2O)    
        mask_H2O_tmp,maxcf,maxcia=mask_weakline(mdbH2O,Parr,dParr,Tarr,SijM,gammaLM,sigmaDM,maxMMR_H2O*ONEARR,molmassH2O,mmw,g,vmrH2,cdbH2H2)
        if k==0:
            mask_H2O=np.copy(mask_H2O_tmp)
        else:
            mask_H2O=mask_H2O+mask_H2O_tmp

        if k==len(T0xarr)-1:
            print("H2O ")
            plot_maxpoint(mask_H2O_tmp,Parr,maxcf,maxcia,mol="H2O")
            plt.savefig("maxpoint_H2O.pdf", bbox_inches="tight", pad_inches=0.0)
            print("H2O saved")
            
    mdbH2O.masking(mask_H2O)
    print("Final:",len(mask_H2O),"->",np.sum(mask_H2O))
    #nu matrix
    numatrix_CO=make_numatrix0(nus,mdbCO.nu_lines)    
    numatrix_H2O=make_numatrix0(nus,mdbH2O.nu_lines)

    return fobsx,nusdx,wavdx,errx,nus,wav,res,mdbCO,mdbH2O,numatrix_CO,numatrix_H2O,cdbH2H2,cdbH2He
    
N=1500
fobs1,nusd1,wavd1,err1,nus1,wav1,res1,mdbCO1,mdbH2O1,numatrix_CO1,numatrix_H2O1,cdbH2H21,cdbH2He1=ap(fobs,nusd,22876.0,23010.0,N)

#######################################################
#HMC-NUTS FITTING PART
#######################################################
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

baseline=1.07 #(baseline for a CIA photosphere in the observed (normaized) spectrum)
# Model
def model_c(nu1,y1,e1):
    Rp = numpyro.sample('Rp', dist.Uniform(0.5,1.5))
    Mp = numpyro.sample('Mp', dist.Normal(33.5,0.3))
    sigma = numpyro.sample('sigma', dist.Exponential(10.0))
    RV = numpyro.sample('RV', dist.Uniform(26.0,30.0))
    MMR_CO = numpyro.sample('MMR_CO', dist.Uniform(0.0,maxMMR_CO))
    MMR_H2O = numpyro.sample('MMR_H2O', dist.Uniform(0.0,maxMMR_H2O))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0,1700.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05,0.15))
    vsini = numpyro.sample('vsini', dist.Uniform(10.0,20.0))

    g=2478.57730044555*Mp/Rp**2 #gravity
    
    #Limb Darkening from 2013A&A...552A..16C (1500K, logg=5, K)
    #0.5969 	0.1125
    u1=0.6
    u2=0.1
    #T-P model//
    Tarr = T0*(Parr/Pref)**alpha 
    
    #line computation CO
    qt_CO=vmap(mdbCO1.qr_interp)(Tarr)
    qt_H2O=vmap(mdbH2O1.qr_interp)(Tarr)
    
    def obyo(y,tag,nusd,nus,numatrix_CO,numatrix_H2O,mdbCO,mdbH2O,cdbH2H2,cdbH2He):
        #CO
        SijM_CO=jit(vmap(SijT,(0,None,None,None,0)))\
            (Tarr,mdbCO.logsij0,mdbCO.dev_nu_lines,mdbCO.elower,qt_CO)
        gammaLMP_CO = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
        gammaLMN_CO=gamma_natural(mdbCO.A)
        gammaLM_CO=gammaLMP_CO+gammaLMN_CO[None,:]
        sigmaDM_CO=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbCO.dev_nu_lines,Tarr,molmassCO)    
        xsm_CO=xsmatrix(numatrix_CO,sigmaDM_CO,gammaLM_CO,SijM_CO) 
        dtaumCO=dtauM(dParr,xsm_CO,MMR_CO*ONEARR,molmassCO,g)
        #H2O
        SijM_H2O=jit(vmap(SijT,(0,None,None,None,0)))\
            (Tarr,mdbH2O.logsij0,mdbH2O.dev_nu_lines,mdbH2O.elower,qt_H2O)
        gammaLMP_H2O = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbH2O.n_Texp,mdbH2O.alpha_ref)
        gammaLMN_H2O=gamma_natural(mdbH2O.A)
        gammaLM_H2O=gammaLMP_H2O+gammaLMN_H2O[None,:]
        sigmaDM_H2O=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbH2O.dev_nu_lines,Tarr,molmassH2O)
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
        mu=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)
        
        errall=jnp.sqrt(e1**2+sigma**2)
        numpyro.sample(tag, dist.Normal(mu, errall), obs=y)

    obyo(y1,"y1",nusd1,nus1,numatrix_CO1,numatrix_H2O1,mdbCO1,mdbH2O1,cdbH2H21,cdbH2He1)

#Running a HMC-NUTS
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 500, 1000
kernel = NUTS(model_c,forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu1=nusd1, y1=fobs1, e1=err1)
print("end HMC")

#Post-processing
posterior_sample = mcmc.get_samples()
np.savez("npz/savepos.npz",[posterior_sample])

pred = Predictive(model_c,posterior_sample,return_sites=["y1"])
nu_1 = nus1
predictions = pred(rng_key_,nu1=nu_1,y1=None,e1=err1)
median_mu1 = jnp.median(predictions["y1"],axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.9)
np.savez("npz/saveplotpred.npz",[wavd1,fobs1,err1,median_mu1,hpdi_mu1])

red=(1.0+28.07/300000.0) #for annotation
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6.0))
ax.plot(wavd1[::-1],median_mu1,color="C0")
ax.plot(wavd1[::-1],fobs1,"+",color="C1",label="data")

#annotation for some lines
ax.plot([22913.3*red,22913.3*red],[0.6,0.75],color="C0",lw=1)
ax.plot([22918.07*red,22918.07*red],[0.6,0.77],color="C1",lw=1)
ax.plot([22955.67*red,22955.67*red],[0.6,0.68],color="C2",lw=1)
plt.text(22913.3*red,0.55,"A",color="C0",fontsize=12,horizontalalignment="center")
plt.text(22918.07*red,0.55,"B",color="C1",fontsize=12,horizontalalignment="center")
plt.text(22955.67*red,0.55,"C",color="C2",fontsize=12,horizontalalignment="center")
#

ax.fill_between(wavd1[::-1], hpdi_mu1[0], hpdi_mu1[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
plt.xlabel("wavelength ($\AA$)",fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)

plt.savefig("npz/results.pdf", bbox_inches="tight", pad_inches=0.0)
plt.savefig("npz/results.png", bbox_inches="tight", pad_inches=0.0)

#ARVIZ part
import arviz
rc = {
    "plot.max_subplots": 250,
}

arviz.rcParams.update(rc)
pararr=["Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","vsini","RV","sigma"]
arviz.plot_trace(mcmc, var_names=pararr)
plt.savefig("npz/trace.png")
pararr=["Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","vsini","RV","sigma"]
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
plt.savefig("npz/cornerall.png")
