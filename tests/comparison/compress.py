import numpy as np
import tqdm
import jax.numpy as jnp
from jax import vmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
from exojax.spec import make_numatrix0
from exojax.spec.lpf import xsvector as lpf_xsvector
from exojax.spec.modit import xsvector as modit_xsvector
from exojax.spec import initspec
from exojax.spec import xsection as lpf_xsection
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
from exojax.spec import rtcheck, moldb
from exojax.spec.dit import set_ditgrid
from exojax.spec.hitran import normalized_doppler_sigma

nus=np.logspace(np.log10(6110),np.log10(6190.0),80000,dtype=np.float64)
mdbCH4=moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10',nus)

print(np.min(mdbCH4.elower),np.max(mdbCH4.elower))

c=plt.hist2d(mdbCH4.nu_lines,(mdbCH4.elower),bins=[len(nus),100],rasterized=True)
plt.colorbar(c[3])
plt.savefig("enux.png")


import sys
sys.exit()
Sij=SijT(Tfix,mdbCH4.logsij0,mdbCH4.nu_lines,mdbCH4.elower,qt)
gammaL = gamma_hitran(Pfix,Tfix,Pfix, mdbCH4.n_air, mdbCH4.gamma_air, mdbCH4.gamma_self)
#+ gamma_natural(A) #uncomment if you inclide a natural width
sigmaD=doppler_sigma(mdbCH4.nu_lines,Tfix,Mmol)

cnu,indexnu,R,pmarray=initspec.init_modit(mdbCH4.nu_lines,nus)
nsigmaD=normalized_doppler_sigma(Tfix,Mmol,R)
ngammaL=gammaL/(mdbCH4.nu_lines/R)
ngammaL_grid=set_ditgrid(ngammaL)

xs_modit_lp=modit_xsvector(cnu,indexnu,R,pmarray,nsigmaD,ngammaL,Sij,nus,ngammaL_grid)
wls_modit = 100000000/nus

#ref (direct)
d=10000
ll=mdbCH4.nu_lines
xsv_lpf_lp=lpf_xsection(nus,ll,sigmaD,gammaL,Sij,memory_size=30)

from jax.config import config
config.update("jax_enable_x64", True)


#PLOT
llow=2300.4
lhigh=2300.7
tip=20.0
fig=plt.figure(figsize=(12,3))
ax=plt.subplot2grid((12, 1), (0, 0),rowspan=8)
plt.plot(wls_modit,xsv_lpf_lp,label="Direct",color="C0",markersize=3,alpha=0.3)
plt.plot(wls_modit,xs_modit_lp,lw=1,color="C1",alpha=1,label="MODIT (F32)")
plt.plot(wls_modit,xs_modit_lp_f64,lw=1,color="C2",alpha=1,label="MODIT (F64)")


plt.ylim(1.1e-28,1.e-17)
#plt.ylim(1.e-27,3.e-20)
plt.yscale("log")

plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
plt.legend(loc="upper right")
plt.ylabel("   cross section $(\mathrm{cm}^2)$",fontsize=10)
#plt.text(22986,3.e-21,"$P=10^{-3}$ bar")
plt.xlabel('wavelength [$\AA$]')

ax=plt.subplot2grid((12, 1), (8, 0),rowspan=4)
plt.plot(wls_modit,np.abs(xs_modit_lp/xsv_lpf_lp-1.)*100,lw=1,alpha=0.5,color="C1",label="MODIT (F32)")
plt.plot(wls_modit,np.abs(xs_modit_lp_f64/xsv_lpf_lp-1.)*100,lw=1,alpha=1,color="C2",label="MODIT (F64)")
plt.yscale("log")
plt.ylabel("difference (%)",fontsize=10)
plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
plt.ylim(0.01,100.0)
plt.xlabel('wavelength [$\AA$]')
plt.legend(loc="upper left")

plt.savefig("comparison_modit.png", bbox_inches="tight", pad_inches=0.0)
plt.savefig("comparison_modit.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
