import numpy as np
import matplotlib.pyplot as plt
import arviz
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


#10**5==2478.57730044555*Mp/Rp**2

def Rpg(Mp,logg):
    #R from logg and M in MJ, RJ
    return np.sqrt(2478.58*Mp/10**logg)

p=np.load("MassHIT/npz/savepos.npz",allow_pickle=True)["arr_0"][0]
px=np.load("MassEMb/npz/savepos.npz",allow_pickle=True)["arr_0"][0]

ax=arviz.plot_kde(p["Mp"],values2=p["Rp"])
ax2=arviz.plot_kde(px["Mp"],values2=px["Rp"])
plt.xlim(10,69.9)
plt.ylim(0.8,1.1)
plt.xscale("log")
plt.tick_params(which ="minor",labelsize=16)
plt.tick_params(which ="major",labelsize=16)
plt.axvspan(33.5-0.3,33.5+0.3,alpha=0.3)
plt.xlabel("mass ($M_J)$",fontsize=16)
plt.ylabel("radius ($R_J)$",fontsize=16)
plt.text(12.0,0.96,"ExoMol (CO,H2O)",fontsize=16)
plt.text(35.0,1.06,"HITEMP (CO)",fontsize=16)
plt.text(35.0,1.04,"+ ExoMol (H2O)",fontsize=16)
plt.text(33.5,0.85,"astrometry",fontsize=14,color="C0",rotation=90)
#logg
Marr=np.linspace(10,70,100)
plt.plot(Marr,Rpg(Marr,5.2),color="gray",alpha=0.5,ls="dashed")
plt.text(39,0.81,"logg=5.2",fontsize=12,color="gray",rotation=65)
plt.plot(Marr,Rpg(Marr,5.0),color="gray",alpha=0.5,ls="dashed")
plt.text(24.9,0.81,"5.0",fontsize=12,color="gray",rotation=65)
plt.plot(Marr,Rpg(Marr,4.8),color="gray",alpha=0.5,ls="dashed")
plt.text(15.8,0.81,"4.8",fontsize=12,color="gray",rotation=65)
plt.plot(Marr,Rpg(Marr,4.6),color="gray",alpha=0.5,ls="dashed")
plt.text(10,0.81,"4.6",fontsize=12,color="gray",rotation=65)


ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
ax2.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

plt.savefig("massinf.pdf", bbox_inches="tight", pad_inches=0.0)
plt.savefig("massinf.png", bbox_inches="tight", pad_inches=0.0)

#plt.show()
