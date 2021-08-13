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

from exojax.spec.dit import npgetix
from exojax.spec.modit import xsvector_np as modit_xsvector_np

from exojax.spec import xsection as lpf_xsection
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
from exojax.spec import rtcheck, moldb
from exojax.spec.dit import make_dLarray
from exojax.spec.dit import set_ditgrid
from exojax.spec.hitran import normalized_doppler_sigma


nus_modit=np.logspace(np.log10(3000),np.log10(6000.0),3000000,dtype=np.float64)
mdbCO=moldb.MdbHit('/home/kawahara/exojax/data/CO/05_hit12.par',nus_modit)

Mmol=28.010446441149536
Tref=296.0
Tfix=1000.0
Pfix=1.e-3 #


#USE TIPS partition function
Q296=np.array([107.25937215917970,224.38496958496091,112.61710362499998,\
     660.22969049609367,236.14433662109374,1382.8672147421873])
Q1000=np.array([382.19096582031250,802.30952197265628,402.80326733398437,\
2357.1041210937501,847.84866308593757,4928.7215078125000])
qr=Q1000/Q296

qt=np.ones_like(mdbCO.isoid,dtype=np.float64)
for idx,iso in enumerate(mdbCO.uniqiso):
    mask=mdbCO.isoid==iso
    qt[mask]=qr[idx]

Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
gammaL = gamma_hitran(Pfix,Tfix,Pfix, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self)
#+ gamma_natural(A) #uncomment if you inclide a natural width
sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
R=(len(nus_modit)-1)/np.log(nus_modit[-1]/nus_modit[0]) #resolution
print("R=",R)
#sys.exit()
dv_lines=mdbCO.nu_lines/R
nsigmaD=normalized_doppler_sigma(Tfix,Mmol,R)
ngammaL=gammaL/dv_lines
ngammaL_grid=set_ditgrid(ngammaL)

Nfold=1
dLarray=make_dLarray(Nfold,1)

#subtract median
dfnus=nus_modit-2300.5 #remove median
dfnu_lines=mdbCO.nu_lines-2300.5 #remove median
dv=nus_modit/R #delta wavenumber grid
xs_modit_lp=modit_xsvector(dfnu_lines,nsigmaD,ngammaL,Sij,dfnus,ngammaL_grid,dLarray,dv_lines,dv)

#use numpy 64
cnu,indexnu=npgetix(mdbCO.nu_lines,nus_modit)
xs_modit_lp_np=modit_xsvector_np(cnu,indexnu,nsigmaD,ngammaL,Sij,nus_modit,ngammaL_grid,dLarray,dv_lines,dv)

wls_modit = 100000000/nus_modit

#PLOT
d=10000
ll=mdbCO.nu_lines
#ref (direct)
xsv_lpf_lp=lpf_xsection(nus_modit,ll,sigmaD,gammaL,Sij,memory_size=30)


llow=2300.4
lhigh=2300.7
tip=20.0
#tip=0.1
fig=plt.figure(figsize=(12,3))
ax=plt.subplot2grid((12, 1), (0, 0),rowspan=8)
plt.plot(wls_modit,xsv_lpf_lp,label="Direct",color="C0",markersize=3,alpha=0.99)
#plt.plot(wls_modit,xs_modit_lp,ls="dashed",color="C1",alpha=0.7,label="MODIT")
plt.plot(wls_modit,xs_modit_lp_np,ls="dashed",color="C1",alpha=0.7,label="MODIT")

plt.ylim(1.1e-28,1.e-17)
#plt.ylim(1.e-27,3.e-20)
plt.yscale("log")

plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
plt.legend(loc="upper right")
plt.ylabel("       cross section",fontsize=10)
#plt.text(22986,3.e-21,"$P=10^{-3}$ bar")
plt.xlabel('wavelength [$\AA$]')

ax=plt.subplot2grid((12, 1), (8, 0),rowspan=4)
#plt.plot(wls_modit,(xs_modit_lp/xsv_lpf_lp-1.),alpha=0.3,color="C1")
plt.plot(wls_modit,(xs_modit_lp_np/xsv_lpf_lp-1.)*100,alpha=0.6,color="C1")

plt.ylabel("difference (%)",fontsize=10)
#for iline in wline:
#    plt.axvline(iline,lw=0.3)
plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
plt.ylim(-30,30)
plt.xlabel('wavelength [$\AA$]')
#plt.legend()
plt.legend(loc="upper left")

plt.savefig("comparison_modit.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
