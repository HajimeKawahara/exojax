from exojax.spec.rtransfer import nugrid
from exojax.spec import moldb, molinfo
from exojax.spec.exomol import gamma_exomol
from exojax.spec import SijT, doppler_sigma, gamma_natural
from exojax.spec import planck
import jax.numpy as jnp
from jax import vmap, jit

N=1500
nus,wav,res=nugrid(22900,22960,N,unit="AA")
#mdbM=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus)
#loading molecular database 
#molmass=molinfo.molmass("CO") #molecular mass (CO)
mdbM=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',nus,crit=1.e-45) #loading molecular dat
molmassM=molinfo.molmass("H2O") #molecular mass (H2O)


import matplotlib.pyplot as plt

q=mdbM.qr_interp(1500.0)
S=SijT(1500.0,mdbM.logsij0,mdbM.nu_lines,mdbM.elower,q)
mask=S>1.e-25
mdbM.masking(mask)


Tarr=jnp.logspace(jnp.log10(800),jnp.log10(1600),100)
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

lsa=["solid","dashed","dotted","dashdot"]
lab=["A","B","C"]
fac=1.e22
fig=plt.figure(figsize=(12,6))
#for j,i in enumerate(range(len(mdbM.A))):
#for j,i in enumerate([56,72,141,147,173,236,259,211,290]):
for j,i in enumerate([259,236,56]):
    print(1.e8/mdbM.nu_lines[i])
    w=lab[j]+": "+str(int(1.e8/mdbM.nu_lines[i]))+"$\\AA$"
    plt.plot(Tarr,SijM[:,i]*fac,color="C"+str(j),ls=lsa[j],alpha=1.0,label=w)
    plt.text(Tmax[i],SijM[imax[i],i]*fac,lab[j],fontsize=20,color="C"+str(j))
    #w=str(int(1.e8/mdbM.nu_lines[i]*(1.0+28.0/300000.)))+"AA"+" i="+str(i)
    #plt.plot(Tarr,SijM[:,i],alpha=1.0,label=w)
    #plt.text(Tmax[i],SijM[imax[i],i],w)

plt.axvspan(1280.0,1310.0,alpha=0.3)
plt.tick_params(labelsize=18)
plt.xlabel("Temperature (K)",fontsize=18)
plt.ylabel("Line strength ($10^{-22}$ cm)",fontsize=18)
plt.ylim(0.0,2.1)
plt.legend(fontsize=18)
plt.title("H2O POKAZATEL",fontsize=18)
plt.savefig("lswater.pdf", bbox_inches="tight", pad_inches=0.0)
plt.savefig("lswater.png", bbox_inches="tight", pad_inches=0.0)
plt.show()
