#!/usr/bin/env python
from exojax.spec import rtransfer as rt
from exojax.spec import planck, moldb, contdb, response, molinfo
from exojax.spec import make_numatrix0,xsvector
from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.plot.atmplot import plot_maxpoint
from exojax.spec.hitrancia import read_cia, logacia 
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
import numpy as np
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import vmap, jit

from exojax.spec.evalline import mask_weakline

#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#grid for F0
N=1000
wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
nus=1.e8/wav[::-1]

#Atmospheric parameters
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
mmw=2.33 #mean molecular weight
molmassCO=molinfo.molmass("CO") #molecular mass (CO)
MMR=0.05*np.ones_like(Parr) #mass mixing ratio
g=1.e5 # gravity cm/s2

#--------------------------------------------------
#ExoMol 
mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus) #loading molecular database
#for j,T0c in enumerate(np.linspace(500,1500,11)):
if True:
    T0c=1500.0
    Tarr = T0c*np.ones_like(Parr)    
    qt=vmap(mdbCO.qr_interp)(Tarr)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
               (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
          (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
             (mdbCO.nu_lines,Tarr,molmassCO)    
    #--------------------------------------------------
    #CIA
    mmrH2=0.74
    molmassH2=molinfo.molmass("H2")
    vmrH2=(mmrH2*mmw/molmassH2)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    
    mask,maxcf,maxcia=mask_weakline(mdbCO,Parr,dParr,Tarr,SijM,gammaLM,sigmaDM,MMR,molmassCO,mmw,g,vmrH2,cdbH2H2)

    plot_maxpoint(mask,Parr,maxcf,maxcia,mol="CO")
    plt.savefig("maxpoint.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()

mdbCO.masking(mask)
print(len(mdbCO.nu_lines))
#----------------------------------------------------------
#grid for F
M=1000
wavd=np.linspace(22900,23000,M,dtype=np.float64)#AA        
nusd=1.e8/wavd[::-1]


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
sourcef=planck.piBarr(Tarr,nus)
F0=rtrun(dtau,sourcef)

#Applying a Response and Noise

#F=response.response(dvmat,F0,vsini_in,beta,RV)
Frot=response.rigidrot(nus,F0,vsini_in,0.0,0.0)
F=response.ipgauss(nus,nusd,Frot,beta,RV)

intfac=1.e7
sigin=0.25
data=F*intfac+np.random.normal(0,sigin,size=M)

#some figures for checking
fig=plt.figure(figsize=(20,6.0))
plt.plot(wav[::-1],F0,lw=1,color="C1",alpha=0.5,label="F0")
plt.plot(wavd[::-1],F,lw=1,color="C2",label="F")
plt.show()
#plt.savefig("fig/highredCO.png")
#plt.clf()
