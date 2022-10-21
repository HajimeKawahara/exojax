import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import jax.numpy as jnp
from jax import random
from jax import vmap, jit

from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, wavenumber_grid
from exojax.plot.atmplot import plottau, plotcf, plot_maxpoint
from exojax.utils.afunc import getjov_logg
from exojax.utils.constants import RJ, pc, Rs, c
from exojax.spec.evalline import mask_weakline

from exojax.spec import dit, modit

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
Nx=3000
ws=22876.0
we=23010.0
mask=(ws<wavd[::-1])*(wavd[::-1]<we)
#additional mask to remove a strong telluric
mask=mask*((22898.5>wavd[::-1])+(wavd[::-1]>22899.5))  
fobsx=fobs[mask]
nusdx=nusd[mask]
wavdx=1.e8/nusdx[::-1]
errx=err[mask]

print("data masked",len(nusd),"->",len(nusdx))

nus,wav,res=nugrid(ws-5.0,we+5.0,Nx,unit="AA",xsmode="modit")
#loading molecular database 
mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) 
mdbH2O=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-46) 
#LOADING CIA
cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)

### MODIT settings
from exojax.spec import initspec
from exojax.spec.modit import  minmax_dgmatrix

cnu_CO, indexnu_CO, R_CO, pmarray_CO=initspec.init_modit(mdbCO.nu_lines,nus)
cnu_H2O, indexnu_H2O, R_H2O, pmarray_H2O=initspec.init_modit(mdbH2O.nu_lines,nus)

# Precomputing gdm_ngammaL                                                                                              
from exojax.spec.modit import setdgm_exomol
from jax import jit, vmap

fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
T0_test=np.array([1000.0,1700.0,1000.0,1700.0])
alpha_test=np.array([0.15,0.15,0.05,0.05])
res=0.05
dgm_ngammaL_CO=setdgm_exomol(mdbCO,fT,Parr,R_CO,molmassCO,res,T0_test,alpha_test)
dgm_ngammaL_H2O=setdgm_exomol(mdbH2O,fT,Parr,R_H2O,molmassH2O,res,T0_test,alpha_test)

#######################################################
#HMC-NUTS FITTING PART
#######################################################
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from exojax.spec.modit import exomol,xsmatrix

baseline=1.07 #(baseline for a CIA photosphere in the observed (normaized) spectrum)
# Model
def predmod(nu1,y1,e1):    
    Rp = 0.88
    Mp = 33.2
    RV = 28.07
    sigma=0.0135
    MMR_CO = 0.0059
    MMR_H2O = 0.0023
    T0 = 1295
    alpha = 0.097
    vsini = 16.0
    q1 = 0.5
    q2 = 0.5
    sqrtq1=jnp.sqrt(q1)
    u1=2.0*sqrtq1*q2
    u2=sqrtq1*(1.0-2.0*q2)
    
    g=2478.57730044555*Mp/Rp**2 #gravity
    
    #T-P model//
    Tarr = T0*(Parr/Pref)**alpha 
    
    #line computation CO
    qt_CO=vmap(mdbCO.qr_interp)(Tarr)
    qt_H2O=vmap(mdbH2O.qr_interp)(Tarr)
    
    def obyo(y,tag,nusdx,nus,mdbCO,mdbH2O,cdbH2H2,cdbH2He):
        #CO
        SijM_CO,ngammaLM_CO,nsigmaDl_CO=exomol(mdbCO,Tarr,Parr,R_CO,molmassCO)
        xsm_CO=xsmatrix(cnu_CO,indexnu_CO,R_CO,pmarray_CO,nsigmaDl_CO,ngammaLM_CO,SijM_CO,nus,dgm_ngammaL_CO)
        dtaumCO=dtauM(dParr,jnp.abs(xsm_CO),MMR_CO*ONEARR,molmassCO,g)
        
        #H2O
        SijM_H2O,ngammaLM_H2O,nsigmaDl_H2O=exomol(mdbH2O,Tarr,Parr,R_H2O,molmassH2O)
        xsm_H2O=xsmatrix(cnu_H2O,indexnu_H2O,R_H2O,pmarray_H2O,nsigmaDl_H2O,ngammaLM_H2O,SijM_H2O,nus,dgm_ngammaL_H2O)
        dtaumH2O=dtauM(dParr,jnp.abs(xsm_H2O),MMR_H2O*ONEARR,molmassH2O,g)

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
        mu=response.ipgauss_sampling(nusdx,nus,Frot,beta,RV)
        return mu

    mu=obyo(y1,"y1",nusdx,nus,mdbCO,mdbH2O,cdbH2H2,cdbH2He)
    return mu
    
#mcmc.run(rng_key_, nu1=nusd1, y1=fobs1, e1=err1)
mu=predmod(nusdx,fobsx,errx)
print(mu)
np.savez("cmodit.npz",[nusdx,mu])
