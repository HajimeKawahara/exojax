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
import pandas as pd

########################
FCIA=2.5e-16 #lambda F_lambda erg/s/cm2
nu0=1.e7/2300.0
print(nu0*planck.piBarr(np.array([1000.0]),nu0))

sys.exit()
#loading spectrum
dat=pd.read_csv("data/luhman16a_spectra.csv",delimiter=",")
wavd=dat["wavelength_micron"]*1.e4
nusd=1.e8/wavd[::-1]
fobs=dat["normalized_flux"][::-1]
err=dat["err_normalized_flux"][::-1]

#masking
mask=wavd<23010.0
fobs=fobs[mask]
nusd=nusd[mask]
wavd=wavd[mask]
err=err[mask]
M=len(nusd)
#plt.plot(wavd[::-1],fobs)
#plt.show()

#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#grid for F0
N=1000
wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
nus=1.e8/wav[::-1]

#grid for F
#M=1000
#wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA        
#nusd=1.e8/wavd[::-1]

#dv matrix
c=299792.458
dvmat=jnp.array(c*np.log(nusd[:,jnp.newaxis]/nus[jnp.newaxis,:]))

#Atmospheric parameters
alpha_in=0.02
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)

########################
FCIA=2.5*e-16 #lambda F_lambda erg/s/cm2

iPfix=87 #fix index for CIA

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

mdbCO=moldb.MdbExomol('../exe/.database/CO/12C-16O/Li2015',nus) #loading molecular database 

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

#print(jnp.sum(dtaucH2He))
#print(jnp.sum(dtaucH2H2))

#sys.exit()
#------------------------------------------------------
#Running Radiative Transfer
dtau=dtaumCO+dtaucH2H2+dtaucH2He

#cf=plotcf(nus,dtau,Tarr,Parr,dParr,unit="AA")
#print(Parr[np.argmax(cf,axis=0)])
#plt.savefig("fig/cf.png")


sourcef=planck.piBarr(Tarr,nus)
F0=rtrun(dtau,sourcef)

#Applying a Response and Noise
Frot=response.rigidrot(nus,F0,vsini_in,0.0,0.0)
F=response.ipgauss(nus,nusd,Frot,beta,RV)

intfac=1.e6/1.11
#sigin=0.25
data=F*intfac#+np.random.normal(0,sigin,size=M)

#------------------------------------------------------
#some figures for checking
fig=plt.figure(figsize=(20,6.0))
plt.plot(wavd[::-1],data,lw=1,color="C2",label="F")
plt.plot(wavd[::-1],fobs)
plt.savefig("fig/spec.png")
