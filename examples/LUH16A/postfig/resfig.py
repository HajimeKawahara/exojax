import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def model_c():
    return None

dat=pd.read_csv("../data/luhman16a_spectra.csv",delimiter=",")
wavd=(dat["wavelength_micron"].values)*1.e4 #AA 
fobs=(dat["normalized_flux"].values)[::-1]
err=(dat["err_normalized_flux"].values)[::-1]
#d=np.load("nofix/saveres.npz",allow_pickle=True)
#x=d["arr_0"]
#p=x[1]["y1"]
#print(p)

d=np.load("~/fig/npz/saveplotpred.npz",allow_pickle=True)["arr_0"]
wavd1,fobs1,err1,median_mu1,hpdi_mu1=d
p=np.load("~/fig/npz/savepos.npz",allow_pickle=True)["arr_0"][0]

fac=0.7
fig=plt.figure(figsize=(25*fac,7*fac))
ax=plt.subplot2grid((12, 1), (0, 0),rowspan=9)
red=(1.0+28.07/300000.0)
sig=0.0135

ax.plot(wavd1[::-1],median_mu1,color="C0",lw=1)
ax.fill_between(wavd1[::-1], hpdi_mu1[0], hpdi_mu1[1], alpha=0.3, interpolate=True,color="C0",
                label="90% area")

ax.plot(wavd1[::-1],fobs1,"+",color="C1",label="data")
#ax.errorbar(wavd1[::-1],fobs1,yerr=np.sqrt(err1**2+sig**2),ecolor="gray",color="gray",fmt='.',label="data",alpha=0.5)
#ax.plot(wavd1[::-1],fobs1,".",color="gray")
ax.plot(wavd1[::-1],median_mu1,color="C0",lw=1.5)


ax.plot([22913.3*red,22913.3*red],[0.6,0.75],color="C0",lw=1)
ax.plot([22918.07*red,22918.07*red],[0.6,0.77],color="C1",lw=1)
#ax.plot([22955.67*red,22955.67*red],[0.6,0.75],color="C2",lw=1)
ax.plot([22955.67*red,22955.67*red],[0.87,0.97],color="C2",lw=1)

plt.text(22913.3*red,0.55,"A",color="C0",fontsize=12,horizontalalignment="center")
plt.text(22918.07*red,0.55,"B",color="C1",fontsize=12,horizontalalignment="center")
#plt.text(22955.67*red,0.55,"C",color="C2",fontsize=12,horizontalalignment="center")
plt.text(22955.67*red,0.99,"C",color="C2",fontsize=12,horizontalalignment="center")

#                                                                               
plt.xlim(np.min(wavd1)-1.0,np.max(wavd1)+1.0)
plt.legend(fontsize=16)
plt.tick_params(labelsize=16)
plt.xticks(color="None")
plt.ylabel("normalized flux",fontsize=15)

#ax=fig.add_subplot(212)
ax=plt.subplot2grid((12, 1), (9, 0),rowspan=3)

#ax.plot(wavd1[::-1],fobs1-median_mu1,color="C0",label="")
ax.errorbar(wavd1[::-1],fobs1-median_mu1,yerr=np.sqrt(err1**2+sig**2),ecolor="gray",color="gray",fmt='.',label="data",alpha=0.5)

plt.axhline(0.0,color="black")
plt.xlim(np.min(wavd1)-1.0,np.max(wavd1)+1.0)
plt.tick_params(labelsize=16)
plt.xlabel("wavelength ($\AA$)",fontsize=16)
plt.ylabel("residual",fontsize=15)

plt.savefig("results.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
