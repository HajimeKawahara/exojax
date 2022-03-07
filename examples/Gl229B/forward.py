#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap, jit

from exojax.spec import rtransfer as rt
from exojax.spec import dit, modit
from exojax.spec import lpf
from exojax.spec import initspec
from exojax.spec import moldb, contdb
from exojax.spec import molinfo
from exojax.spec import SijT
from exojax.spec.rtransfer import nugrid
from exojax.spec.exomol import gamma_exomol
from exojax.spec import gamma_natural
from exojax.spec import normalized_doppler_sigma
from exojax.spec.rtransfer import dtauCIA
from exojax.spec.rtransfer import dtauM
import jax.numpy as jnp
from exojax.plot.atmplot import plotcf
from exojax.spec import planck
from exojax.spec.rtransfer import rtrun
from exojax.spec import response
from exojax.utils.constants import c
import jax.numpy as jnp


import pandas as pd


dats = pd.read_csv('data/Gl229B/Gl229B_spectrum_CH4.dat',
                   names=('wav', 'flux'), delimiter='\s')
wavmic = dats['wav'].values*1.e4
ccgs = 29979245800.0
flux = dats['flux'].values*ccgs
# print(wavmic)
# plt.plot(wavmic,flux)
# plt.show()
#import sys
# sys.exit()


dat = pd.read_csv('data/profile.dat')

Parr = dat['P'].values
NP = len(Parr)
Parrx, dParr, k = rt.pressure_layer(
    NP=NP, logPtop=np.log10(Parr[0]), logPbtm=np.log10(Parr[-1]))
Tarr = dat['T'].values
MMR = dat['C1H4'].values


#Parr, dParr, k=rt.pressure_layer(NP=NP)
#Tarr = T0*(Parr)**0.1
nus, wav, R = nugrid(15900, 16300, 30000, unit='AA', xsmode='modit')
print('R=', R)
mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus, crit=1.e-36)
cdbH2H2 = contdb.CdbCIA('.database/H2-H2_2011.cia', nus)
print('N=', len(mdbCH4.A))
molmassCH4 = molinfo.molmass('CH4')
qt = vmap(mdbCH4.qr_interp)(Tarr)
gammaLMP = jit(vmap(gamma_exomol, (0, 0, None, None)))(
    Parr, Tarr, mdbCH4.n_Texp, mdbCH4.alpha_ref)
gammaLMN = gamma_natural(mdbCH4.A)
gammaLM = gammaLMP+gammaLMN[None, :]
nsigmaDl = normalized_doppler_sigma(Tarr, molmassCH4, R)[:, np.newaxis]
SijM = jit(vmap(SijT, (0, None, None, None, 0)))(
    Tarr, mdbCH4.logsij0, mdbCH4.nu_lines, mdbCH4.elower, qt)

dv_lines = mdbCH4.nu_lines/R
ngammaLM = gammaLM/dv_lines
dv = nus/R

dgm_ngammaL = dit.dgmatrix(ngammaLM, 0.2)


cnu, indexnu, R, pmarray = initspec.init_modit(mdbCH4.nu_lines, nus)
xsmmodit = modit.xsmatrix(cnu, indexnu, R, pmarray,
                          nsigmaDl, ngammaLM, SijM, nus, dgm_ngammaL)


# g=2478.57730044555*Mp/Rp**2
g = 1.e5  # gravity cm/s2
# MMR=0.0059 #mass mixing ratio

# 0-padding for negative values
xsmnp = np.array(xsmmodit)
print(len(xsmnp[xsmnp < 0.0]))
xsmnp[xsmnp < 0.0] = 0.0
xsmditc = jnp.array(xsmnp)
# -------------------------------

dtaum = dtauM(dParr, xsmditc, MMR*np.ones_like(Tarr), molmassCH4, g)

mmw = 2.33  # mean molecular weight
mmrH2 = 0.74
molmassH2 = molinfo.molmass('H2')
vmrH2 = (mmrH2*mmw/molmassH2)  # VMR
dtaucH2H2 = dtauCIA(nus, Tarr, Parr, dParr, vmrH2, vmrH2,
                    mmw, g, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)

dtau = dtaum+dtaucH2H2

plotcf(nus, dtau, Tarr, Parr, dParr)
plt.show()


# radiative transfering...

# In[26]:


sourcef = planck.piBarr(Tarr, nus)
F0 = rtrun(dtau, sourcef)

wavd = jnp.linspace(15910, 16290, 1500)  # observational wavelength grid
nusd = 1.e8/wavd[::-1]

RV = 10.0  # RV km/s
vsini = 20.0  # Vsini km/s
u1 = 0.0  # limb darkening u1
u2 = 0.0  # limb darkening u2

Rinst = 100000.
beta = c/(2.0*np.sqrt(2.0*np.log(2.0))*Rinst)  # IP sigma need check

Frot = response.rigidrot(nus, F0, vsini, u1, u2)
F = response.ipgauss_sampling(nusd, nus, Frot, beta, RV)


###

# In[42]:

fig = plt.figure(figsize=(20, 4))
plt.plot(wav[::-1], F0, alpha=0.5, color='C1', label='exojax (CH4 only)')
# plt.plot(wavd[::-1],F)
plt.plot(wavmic, flux, alpha=0.5, color='C2', label='petit?')
plt.legend()
plt.ylim(0, 15000)
plt.xlim(np.min(wav), np.max(wav))
plt.xlabel('wavelength ($\AA$)')
plt.savefig('moditCH4.png')
# plt.show()

np.savez('ch4.npz', [wav, F0, wavd, F])
# In[ ]:
