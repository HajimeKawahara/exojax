from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo
from exojax.spec import make_numatrix0,xsvector
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
from exojax.plot.atmplot import plottau, plotcf, plot_maxpoint
from exojax.utils.afunc import getjov_logg
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import vmap, jit
import pandas as pd
from exojax.utils.constants import RJ, pc, Rs, c
import sys
from exojax.spec.evalline import mask_weakline


#FLUX reference
Fabs_REF2=2.7e-12 #absolute flux (i.e. flux@10pc) erg/s/cm2/um Burgasser+ 1303.7283 @2.3um
Rp=0.85*RJ #BD radius
fac0=Rp**2/((10.0*pc)**2) 
Ftoa=Fabs_REF2/fac0/1.e4 #erg/cm2/s/cm-1 @ 2.3um

#loading spectrum
dat=pd.read_csv("data/luhman16a_spectra.csv",delimiter=",")
wavd=(dat["wavelength_micron"].values)*1.e4 #AA
nusd=1.e8/wavd[::-1]
fobs=(dat["normalized_flux"].values)[::-1]
err=(dat["err_normalized_flux"].values)[::-1]
plt.plot(wavd[::-1],fobs)


#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#ATMOSPHERE
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
icia=87#CIA layer
mmw=2.33 #mean molecular weight
R=100000.
beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R) #IP sigma need check
print("beta=",beta,"km/s")
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
g=1.e5
T0c=1700.0
Tarr = T0c*np.ones_like(Parr)    
maxMMR_CO=0.02
maxMMR_H2O=0.01

#masking
def ap(fobs,nusd,ws,we,Nx):
    mask=(ws<wavd[::-1])*(wavd[::-1]<we)
    fobsx=fobs[mask]
    nusdx=nusd[mask]
    wavdx=1.e8/nusdx[::-1]
    errx=err[mask]
    N=4500
    nus,wav,res=nugrid(ws-5.0,we+5.0,Nx,unit="AA")
    #loading molecular database 
    mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) 
    mdbH2O=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-45) 
    print("resolution=",res)

    #LOADING CIA
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)

    ### REDUCING UNNECESSARY LINES
    #######################################################
    
    #1. CO
    qt=vmap(mdbCO.qr_interp)(Tarr)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
        (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
        (mdbCO.nu_lines,Tarr,molmassCO)    
    
    mask_CO,maxcf,maxcia=mask_weakline(mdbCO,Parr,dParr,Tarr,SijM,gammaLM,sigmaDM,maxMMR_CO*ONEARR,molmassCO,mmw,g,vmrH2,cdbH2H2)
    mdbCO.masking(mask_CO)

    #plot_maxpoint(mask_CO,Parr,maxcf,maxcia,mol="CO")
    #plt.savefig("maxpoint_CO.pdf", bbox_inches="tight", pad_inches=0.0)
    
    
    #2. H2O
    qt=vmap(mdbH2O.qr_interp)(Tarr)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
        (Parr,Tarr,mdbH2O.n_Texp,mdbH2O.alpha_ref)
    gammaLMN=gamma_natural(mdbH2O.A)
    gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbH2O.logsij0,mdbH2O.nu_lines,mdbH2O.elower,qt)
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
    (mdbH2O.nu_lines,Tarr,molmassH2O)    
    
    mask_H2O,maxcf,maxcia=mask_weakline(mdbH2O,Parr,dParr,Tarr,SijM,gammaLM,sigmaDM,maxMMR_H2O*ONEARR,molmassH2O,mmw,g,vmrH2,cdbH2H2)
    
    mdbH2O.masking(mask_H2O)

    #nu matrix
    numatrix_CO=make_numatrix0(nus,mdbCO.nu_lines)    
    numatrix_H2O=make_numatrix0(nus,mdbH2O.nu_lines)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)

    
    return fobsx,nusdx,wavdx,errx,nus,wav,res,mdbCO,mdbH2O,numatrix_CO,numatrix_H2O,cdbH2H2,cdbH2He
    
#plot_maxpoint(mask_H2O,Parr,maxcf,maxcia,mol="H2O")
#plt.savefig("maxpoint_H2O.pdf", bbox_inches="tight", pad_inches=0.0)

N=1000
fobs1,nusd1,wavd1,err1,nus1,wav1,res1,mdbCO1,mdbH2O1,numatrix_CO1,numatrix_H2O1,cdbH2H21,cdbH2He1=ap(fobs,nusd,22876.0,23010.0,N)
fobs2,nusd2,wavd2,err2,nus2,wav2,res2,mdbCO2,mdbH2O2,numatrix_CO2,numatrix_H2O2,cdbH2H22,cdbH2He2=ap(fobs,nusd,23038.6,23159.3,N)
fobs3,nusd3,wavd3,err3,nus3,wav3,res3,mdbCO3,mdbH2O3,numatrix_CO3,numatrix_H2O3,cdbH2H23,cdbH2He3=ap(fobs,nusd,23193.5,23310.0,N)
fobs4,nusd4,wavd4,err4,nus4,wav4,res4,mdbCO4,mdbH2O4,numatrix_CO4,numatrix_H2O4,cdbH2H24,cdbH2He4=ap(fobs,nusd,23341.6,23453.2,N)


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
def model_c(nu1,y1,nu2,y2,nu3,y3,nu4,y4):
    f = numpyro.sample('f', dist.Uniform(0.0,1.0))
    A1 = numpyro.sample('A1', dist.Normal(1.0,0.1))
    A2 = numpyro.sample('A2', dist.Normal(1.0,0.1))
    A3 = numpyro.sample('A3', dist.Normal(1.0,0.1))
    A4 = numpyro.sample('A4', dist.Normal(1.0,0.1))    
    Rp = numpyro.sample('Rp', dist.Uniform(0.5,1.1))
    Mp = numpyro.sample('Mp', dist.Normal(34.2,1.2))
    sigma = numpyro.sample('sigma', dist.Exponential(0.5))
    RV = numpyro.sample('RV', dist.Uniform(27.0,29.0))
    MMR_CO = numpyro.sample('MMR_CO', dist.Uniform(0.0,maxMMR_CO))
    MMR_H2O = numpyro.sample('MMR_H2O', dist.Uniform(0.0,maxMMR_H2O))
    T0 = numpyro.sample('T0', dist.Uniform(1000.0,1700.0))
    alpha = numpyro.sample('alpha', dist.Uniform(0.05,0.15))
    vsini = numpyro.sample('vsini', dist.Uniform(10.0,15.0))
#    logg = numpyro.sample('logg', dist.Uniform(4.0,6.0))

    g=2478.57730044555*Mp/Rp**2 #gravity
    u1=0.0
    u2=0.0
    #T-P model//
    Tarr = T0*(Parr/Parr[icia])**alpha 
    
    #line computation CO
    qt_CO=vmap(mdbCO1.qr_interp)(Tarr)
    qt_H2O=vmap(mdbH2O1.qr_interp)(Tarr)
    
    def obyo(y,tag,Ax,nusd,nus,numatrix_CO,numatrix_H2O,mdbCO,mdbH2O,cdbH2H2,cdbH2He):
        #CO
        SijM_CO=jit(vmap(SijT,(0,None,None,None,0)))\
            (Tarr,mdbCO.logsij0,mdbCO.dev_nu_lines,mdbCO.elower,qt_CO)
        gammaLMP_CO = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
        gammaLMN_CO=gamma_natural(mdbCO.A)
        gammaLM_CO=gammaLMP_CO[:,None]+gammaLMN_CO[None,:]
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
        gammaLM_H2O=gammaLMP_H2O[:,None]+gammaLMN_H2O[None,:]
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

        F0=rtrun(dtau,sourcef)/(Ftoa*Ax)
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        mu=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)
        numpyro.sample(tag, dist.Normal(mu, sigma), obs=y)

    obyo(y1,"y1",A1,nusd1,nus1,numatrix_CO1,numatrix_H2O1,mdbCO1,mdbH2O1,cdbH2H21,cdbH2He1)
    obyo(y2,"y2",A2,nusd2,nus2,numatrix_CO2,numatrix_H2O2,mdbCO2,mdbH2O2,cdbH2H22,cdbH2He2)
    obyo(y3,"y3",A3,nusd3,nus3,numatrix_CO3,numatrix_H2O3,mdbCO3,mdbH2O3,cdbH2H23,cdbH2He3)
    obyo(y4,"y4",A4,nusd4,nus4,numatrix_CO4,numatrix_H2O4,mdbCO4,mdbH2O4,cdbH2H24,cdbH2He4)
#--------------------------------------------------------
#Running a HMC-NUTS

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 100, 200

kernel = NUTS(model_c,forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_, nu1=nusd1, y1=fobs1, nu2=nusd2, y2=fobs2, nu3=nusd3, y3=fobs3, nu4=nusd4, y4=fobs4)
print("end")
#mcmc.print_summary()

#--------------------------------------------------------
#Post-processing

posterior_sample = mcmc.get_samples()
np.savez("savepos.npz",[posterior_sample])

pred = Predictive(model_c,posterior_sample,return_sites=["y1","y2","y3","y4"])
nu_1 = nus1
nu_2 = nus2
nu_3 = nus3
nu_4 = nus4

predictions = pred(rng_key_,nu1=nu_1,y1=None,nu2=nu_2,y2=None,nu3=nu_3,y3=None,nu4=nu_4,y4=None)
#------------------
np.savez("saveres.npz",[pred,predictions,mcmc])
np.savez("savenus.npz",[nus1,nus2,nus3,nus4])
np.savez("savedata.npz",[nusd1, fobs1, nusd2, fobs2, nusd3, fobs3, nusd4, fobs4])
#------------------
median_mu1 = jnp.median(predictions["y1"],axis=0)
hpdi_mu1 = hpdi(predictions["y1"], 0.9)
median_mu2 = jnp.median(predictions["y2"],axis=0)
hpdi_mu2 = hpdi(predictions["y2"], 0.9)
median_mu3 = jnp.median(predictions["y3"],axis=0)
hpdi_mu3 = hpdi(predictions["y3"], 0.9)
median_mu4 = jnp.median(predictions["y4"],axis=0)
hpdi_mu4 = hpdi(predictions["y4"], 0.9)

#------------------
np.savez("saveall0.npz",[median_mu1,hpdi_mu1,median_mu2,hpdi_mu2,median_mu3,hpdi_mu3,median_mu4,hpdi_mu4])
#------------------


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6.0))
#plt.plot(wav[::-1],Fx0,lw=1,color="C1",label="in")
ax.plot(wavd1[::-1],median_mu1,color="C0")
ax.plot(wavd1[::-1],fobs1,"+",color="C1",label="data")
ax.fill_between(wavd1[::-1], hpdi_mu1[0], hpdi_mu1[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")
ax.plot(wavd2[::-1],median_mu2,color="C0")
ax.plot(wavd2[::-1],fobs2,"+",color="C1")
ax.fill_between(wavd2[::-1], hpdi_mu2[0], hpdi_mu2[1], alpha=0.3, interpolate=True,color="C0")
ax.plot(wavd3[::-1],median_mu3,color="C0")
ax.plot(wavd3[::-1],fobs3,"+",color="C1")
ax.fill_between(wavd3[::-1], hpdi_mu3[0], hpdi_mu3[1], alpha=0.3, interpolate=True,color="C0")
ax.plot(wavd4[::-1],median_mu4,color="C0")
ax.plot(wavd4[::-1],fobs4,"+",color="C1")
ax.fill_between(wavd4[::-1], hpdi_mu4[0], hpdi_mu4[1], alpha=0.3, interpolate=True,color="C0")

plt.xlabel("wavelength ($\AA$)",fontsize=16)
plt.legend()
plt.savefig("fig/results.png")

pararr=["f","Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","vsini","RV","sigma","A1","A2","A3","A4"]
arviz.plot_trace(mcmc, var_names=pararr)
plt.savefig("fig/trace.png")

pararr=["f","Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","sigma"]
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
plt.savefig("fig/corner1.png")

pararr=["f","Mp","Rp","T0","A1","A2","A3","A4"]
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
plt.savefig("fig/corner2.png")

pararr=["f","Mp","Rp","T0","alpha","vsini","RV"]
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
plt.savefig("fig/corner3.png")

pararr=["f","Mp","Rp","T0","alpha","MMR_CO","MMR_H2O","vsini","RV","sigma","A1","A2","A3","A4"]
arviz.plot_pair(arviz.from_numpyro(mcmc),kind='kde',divergences=False,marginals=True) 
plt.savefig("fig/cornerall.png")
