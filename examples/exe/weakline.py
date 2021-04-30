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

from exojax.spec.evalline import xsmatrix0, contfunc

#######################################################
#GENERATING A MOCK SPECTRUM PART
#######################################################

#grid for F0
N=1000
wav=np.linspace(22900,23000,N,dtype=np.float64)#AA
nus=1.e8/wav[::-1]


#dv matrix
c=299792.458

#Atmospheric parameters
alpha_in=0.02
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
Tarr = 1500.*(Parr/Parr[-1])**alpha_in
mmw=2.33 #mean molecular weight
molmassCO=molinfo.molmass("CO") #molecular mass (CO)
MMR=0.01*np.ones_like(Tarr) #mass mixing ratio
g=1.e5 # gravity cm/s2


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

#
ss=np.sum(SijM,axis=0)
print(len(ss[ss>0.0]))
xsm0=xsmatrix0(sigmaDM,gammaLM,SijM)
print(np.shape(xsm0))

dtaumCO=dtauM(dParr,xsm0,MMR,molmassCO,g)


cfCO=contfunc(dtaumCO,mdbCO.nu_lines,Parr,dParr,Tarr)

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
dtaucH2H2=dtauCIA(mdbCO.nu_lines,Tarr,Parr,dParr,vmrH2,vmrH2,\
              mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)

cfCIA=contfunc(dtaucH2H2,mdbCO.nu_lines,Parr,dParr,Tarr)

maxcf=np.argmax(cfCO,axis=0)
mask=(maxcf>0)*(maxcf<NP-1)
xarr=np.array(range(0,len(maxcf)))
print(len(xarr),"->",len(xarr[mask]))

fig=plt.figure(figsize=(14,6))
plt.plot(xarr,Parr[maxcf],".",label="CO",alpha=1.0,color="black")
plt.plot(xarr,Parr[np.argmax(cfCIA,axis=0)],"+",label="CIA (H2-H2)",alpha=0.4,color="C2")
plt.plot(xarr[mask],Parr[maxcf[mask]],"x",label="CO selected",alpha=0.4,color="C3")

plt.yscale("log")
plt.ylim(Parr[0],Parr[-1])
plt.gca().invert_yaxis()
plt.tick_params(labelsize=16)
plt.xlabel("#line",fontsize=16)
plt.ylabel("Pressure (bar)",fontsize=16)
plt.legend(fontsize=15)
plt.savefig("maxpoint.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()

#a=plt.imshow(cf)
#a=plt.plot(np.sum(cfCO,axis=0),".",label="CO")
#a=plt.plot(np.sum(cfCIA,axis=0),"+",label="CIA (H2-H2)")


#plt.colorbar(a)
#plt.yscale("log")
#plt.show()
