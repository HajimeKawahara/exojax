"""Test plotting of Luhman16A


"""
from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo
from exojax.spec import make_numatrix0,xsvector
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.plot.atmplot import plottau, plotcf
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import vmap, jit
import pandas as pd
from exojax.utils.constants import RJ, pc, Rs
import sys

#FLUX reference
Fabs_REF2=2.7e-12 #absolute flux (i.e. flux@10pc) erg/s/cm2/um Burgasser+ 1303.7283 @2.3um
Rp=RJ #BD radius
fac0=Rp**2/((10.0*pc)**2) 
Ftoa=Fabs_REF2/fac0/1.e4 #erg/cm2/s/cm @ 2.3um

#loading spectrum
dat=pd.read_csv("data/luhman16a_spectra.csv",delimiter=",")
wavd=(dat["wavelength_micron"].values)*1.e4 #AA
nusd=1.e8/wavd[::-1]
fobs=(dat["normalized_flux"].values)[::-1]
err=(dat["err_normalized_flux"].values)[::-1]
plt.plot(wavd[::-1],fobs)

#masking
mask=(22876.0<wavd[::-1])*(wavd[::-1]<23010.0)
#mask=mask*((22980>wavd[::-1])+(wavd[::-1]>22990.0)) #other line masking
fobs=fobs[mask]
nusd=nusd[mask]
err=err[mask]
wavd=1.e8/nusd[::-1]
M=len(nusd)
plt.plot(wavd[::-1],fobs)
plt.savefig("fig/spec0.png")

#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#grid for F0
N=1500
nus,wav,res=nugrid(22850,23030,N,unit="AA")
print("resolution=",res)

#dv matrix
c=299792.458
dvmat=jnp.array(c*np.log(nusd[:,jnp.newaxis]/nus[jnp.newaxis,:]))

#Atmospheric parameters
alpha_in=0.05
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr = 1200.*(Parr/Parr[-1])**alpha_in

mmw=2.33 #mean molecular weight
molmassCO=molinfo.molmass("CO") #molecular mass (CO)
MMR=0.02*np.ones_like(Tarr) #mass mixing ratio
g=1.e5 # gravity cm/s2

#Macro response model
RV=30.0 #km/s
vsini_in=10.0 #rotational vsini km/s
beta=3.0 #IP sigma km/s

#--------------------------------------------------
#ExoMol 
#LOADING CO
mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) #loading molecular database 
molmassCO=molinfo.molmass("CO") #molecular mass (CO)

#LOADING H2O
mdbH2O=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-45) #loading molecular dat
molmassH2O=molinfo.molmass("H2O") #molecular mass (H2O)


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

#--------------------------------------------------
#CIA
mmrH2=0.74
mmrHe=0.25
molmassH2=molinfo.molmass("H2")
molmassHe=molinfo.molmass("He")
vmrH2=(mmrH2*mmw/molmassH2)
vmrHe=(mmrHe*mmw/molmassHe)
##H2-H2
cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,\
              mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
##H2-He
cdbH2He=contdb.CdbCIA('.database/H2-He_2011.cia',nus)
dtaucH2He=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrHe,\
              mmw,g,cdbH2He.nucia,cdbH2He.tcia,cdbH2He.logac)

#Running Radiative Transfer
dtau=dtaumCO+dtaucH2H2+dtaucH2He

#cf=plotcf(nus,dtau,Tarr,Parr,dParr,unit="AA")
#print(Parr[np.argmax(cf,axis=0)])
#plt.savefig("fig/cf.png")

sourcef=planck.piBarr(Tarr,nus)
F0=rtrun(dtau,sourcef)/Ftoa #divided by the normalization.

u1=0.0
u2=0.0
Frot=response.rigidrot(nus,F0,vsini_in,u1,u2)
F=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)


#------------------------------------------------------
#some figures for checking
fig=plt.figure(figsize=(20,6.0))
plt.plot(wav[::-1],F0,lw=1,color="C1",label="F0")
plt.plot(wavd[::-1],F,lw=1,color="C2",label="F")
plt.plot(wavd[::-1],fobs)
plt.savefig("fig/spec.png")
