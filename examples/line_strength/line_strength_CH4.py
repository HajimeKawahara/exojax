from exojax.spec.rtransfer import nugrid
from exojax.spec import moldb, molinfo
from exojax.spec.exomol import gamma_exomol
from exojax.spec import SijT, doppler_sigma, gamma_natural
from exojax.spec import planck
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import matplotlib.pyplot as plt

N=1500
#nus,wav,res=nugrid(23365,23385,N,unit="AA")
nus,wav,res=nugrid(23200,23300,N,unit="AA")
mdbM=moldb.MdbExomol('.database/CH4/12C-1H4/YT34to10',nus)

q=mdbM.qr_interp(1300.0)
S=SijT(1300.0,mdbM.logsij0,mdbM.nu_lines,mdbM.elower,q)
mask=S>1.e-28
mdbM.masking(mask)

hcperk=1.4387773538277202
Tfixx=1300.0
qtT=mdbM.qr_interp(Tfixx)
logSijTx=np.log10(SijT(Tfixx,mdbM.logsij0,mdbM.nu_lines,mdbM.elower,qtT))

fig=plt.figure(figsize=(10,5.5))
c=plt.scatter(1.e8/mdbM.nu_lines,hcperk*mdbM.elower,c=logSijTx)
plt.yscale("log")
plt.show()
