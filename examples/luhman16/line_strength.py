from exojax.spec.rtransfer import nugrid
from exojax.spec import moldb, molinfo
from exojax.spec.exomol import gamma_exomol
from exojax.spec import SijT, doppler_sigma, gamma_natural
from exojax.spec import planck
import jax.numpy as jnp
from jax import vmap, jit

N=1500
nus,wav,res=nugrid(23420,23450,N,unit="AA")
mdbM=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus)
#loading molecular database 
#molmass=molinfo.molmass("CO") #molecular mass (CO)
#mdbM=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-45) #loading molecular dat
#molmassM=molinfo.molmass("H2O") #molecular mass (H2O)


import matplotlib.pyplot as plt

q=mdbM.qr_interp(1500.0)
S=SijT(1500.0,mdbM.logsij0,mdbM.nu_lines,mdbM.elower,q)
mask=S>1.e-28
mdbM.masking(mask)


Tarr=jnp.logspace(jnp.log10(800),jnp.log10(1700),100)
qt=vmap(mdbM.qr_interp)(Tarr)
SijM=jit(vmap(SijT,(0,None,None,None,0)))\
    (Tarr,mdbM.logsij0,mdbM.nu_lines,mdbM.elower,qt)


imax=jnp.argmax(SijM,axis=0)
Tmax=Tarr[imax]
print(jnp.min(Tmax))

pl=planck.piBarr(jnp.array([1100.0,1000.0]),nus)
print(pl[1]/pl[0])

pl=planck.piBarr(jnp.array([1400.0,1200.0]),nus)
print(pl[1]/pl[0])

lsa=["solid","dashed","dotted"]
#for i in range(len(mdbM.A)):
for j,i in enumerate([7,5,3]):
#    w=str(int(1.e8/mdbM.nu_lines[i]*(1.0+28.0/300000.)))+"AA"+" i="+str(i)
    w=str(int(1.e8/mdbM.nu_lines[i]))+"$\\AA$"
    plt.plot(Tarr,SijM[:,i],color="C"+str(j),ls=lsa[j],alpha=1.0,label=w)
#    plt.text(Tmax[i],SijM[imax[i],i],w)
plt.axvspan(1000.0,1100.0,alpha=0.3)
plt.axvspan(1200.0,1400.0,alpha=0.3)

plt.legend()
#plt.xscale("log")
#plt.yscale("log")
plt.show()
