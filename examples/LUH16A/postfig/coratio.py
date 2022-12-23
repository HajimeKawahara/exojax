import numpy as np
import matplotlib.pyplot as plt
import arviz
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from exojax.spec import molinfo

#10**5==2478.57730044555*Mp/Rp**2

def Rpg(Mp,logg):
    #R from logg and M in MJ, RJ
    return np.sqrt(2478.58*Mp/10**logg)

def est(val,N=3):
    per=np.percentile(val,[5,95])
    med=np.round(np.median(val),N)
    per0=np.round(per[0]-med,N)
    per1=np.round(per[1]-med,N)

    print(str(med)+'_{'+str(per0)+'}^{'+str(per1)+'}')


p=np.load("npz/savepos.npz",allow_pickle=True)["arr_0"][0]
mCO=molinfo.molmass_isotope("CO") #molecular mass (CO)
mH2O=molinfo.molmass_isotope("H2O") #molecular mass (CO)
T0=p["T0"]
mmrCO=p["MMR_CO"]
mmrH2O=p["MMR_H2O"]
corat=(1.0+(mCO*mmrH2O)/(mH2O*mmrCO))**-1
est(T0)
est(corat)
est(mmrCO,4)
est(mmrH2O,4)



#plt.hist(corat,bins=100)
#plt.savefig("coratio.pdf", bbox_inches="tight", pad_inches=0.0)
#plt.show()
