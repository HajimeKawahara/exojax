import numpy as np
import matplotlib.pyplot as plt
import arviz
import jax.numpy as jnp
from exojax.spec import rtransfer as rt

#ATMOSPHERE                                                                     
NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
T0=1295.0 #K
alpha=0.099
Tarrc = T0*(Parr)**alpha

p=np.load("npz/savepos.npz",allow_pickle=True)["arr_0"][0]
Tsample=p["Tarr"]
T0sample=p["T0"]

from numpyro.diagnostics import hpdi
mean_muy = jnp.mean(Tsample*T0, axis=0)
hpdi_muy = hpdi(Tsample+T0sample[:,None], 0.90,axis=0)


fig=plt.figure(figsize=(5,7))
for i in range(0,np.shape(Tsample)[0]):
    T0=T0sample[i]
    Tarr=Tsample[i,:]
    plt.plot(Tarr+T0,Parr,alpha=0.05,color="green",rasterized=True)

plt.plot(Tarrc,Parr,alpha=1.0,color="black",lw=1,label="best-fit power law")
#plt.fill_betweenx(Parr, hpdi_muy[0], hpdi_muy[1], alpha=0.3, interpolate=True,color="C0")

plt.yscale("log")
plt.xlabel("temperature (K)",fontsize=17)
plt.ylabel("pressure (bar)",fontsize=17)
plt.xlim(0,3200)
plt.tick_params(labelsize=17)
plt.ylim(Parr[0],Parr[-1])

plt.gca().invert_yaxis()
plt.savefig("Tarr.png")
plt.savefig("Tarr.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()

