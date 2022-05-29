""" MODIT  HITRAN CO 

"""
from jax.config import config
from exojax.spec.hitran import normalized_doppler_sigma
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.spec import moldb
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
from exojax.spec import xsection as lpf_xsection
from exojax.spec import initspec
from exojax.spec.modit import xsvector as modit_xsvector
from exojax.spec.lpf import xsvector as lpf_xsvector
from exojax.spec import make_numatrix0
import numpy as np
import tqdm
import jax.numpy as jnp
from jax import vmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')

nus = np.logspace(np.log10(4000), np.log10(4500.0), 3000000, dtype=np.float64)
mdbCO = moldb.MdbHit('~/exojax/data/CO/05_hit12.par', nus)

Mmol = 28.010446441149536
Tref = 296.0
Tfix = 1000.0
Pfix = 1.e-3

# USE TIPS partition function
Q296 = np.array([107.25937215917970, 224.38496958496091, 112.61710362499998,
                 660.22969049609367, 236.14433662109374, 1382.8672147421873])
Q1000 = np.array([382.19096582031250, 802.30952197265628, 402.80326733398437,
                  2357.1041210937501, 847.84866308593757, 4928.7215078125000])
qr = Q1000/Q296

qt = np.ones_like(mdbCO.isoid, dtype=np.float64)
for idx, iso in enumerate(mdbCO.uniqiso):
    mask = mdbCO.isoid == iso
    qt[mask] = qr[idx]

Sij = SijT(Tfix, mdbCO.logsij0, mdbCO.nu_lines, mdbCO.elower, qt)
gammaL = gamma_hitran(Pfix, Tfix, Pfix, mdbCO.n_air,
                      mdbCO.gamma_air, mdbCO.gamma_self)
# + gamma_natural(A) #uncomment if you inclide a natural width
sigmaD = doppler_sigma(mdbCO.nu_lines, Tfix, Mmol)

cnu, indexnu, R, pmarray = initspec.init_modit(mdbCO.nu_lines, nus)
nsigmaD = normalized_doppler_sigma(Tfix, Mmol, R)
ngammaL = gammaL/(mdbCO.nu_lines/R)
ngammaL_grid = ditgrid_log_interval(ngammaL)

xs_modit_lp = modit_xsvector(
    cnu, indexnu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid)
wls_modit = 100000000/nus

#ref (direct)
d = 10000
ll = mdbCO.nu_lines
xsv_lpf_lp = lpf_xsection(nus, ll, sigmaD, gammaL, Sij, memory_size=30)

config.update('jax_enable_x64', True)

xs_modit_lp_f64 = modit_xsvector(
    cnu, indexnu, R, pmarray, nsigmaD, ngammaL, Sij, nus, ngammaL_grid)


# PLOT
llow = 2300.4
lhigh = 2300.7
tip = 20.0
fig = plt.figure(figsize=(12, 3))
ax = plt.subplot2grid((12, 1), (0, 0), rowspan=8)
plt.plot(wls_modit, xsv_lpf_lp, label='Direct',
         color='C0', markersize=3, alpha=0.3)
plt.plot(wls_modit, xs_modit_lp, lw=1,
         color='C1', alpha=1, label='MODIT (F32)')
plt.plot(wls_modit, xs_modit_lp_f64, lw=1,
         color='C2', alpha=1, label='MODIT (F64)')


plt.ylim(1.1e-28, 1.e-17)
# plt.ylim(1.e-27,3.e-20)
plt.yscale('log')

plt.xlim(llow*10.0-tip, lhigh*10.0+tip)
plt.legend(loc='upper right')
plt.ylabel('   cross section $(\mathrm{cm}^2)$', fontsize=10)
#plt.text(22986,3.e-21,"$P=10^{-3}$ bar")
plt.xlabel('wavelength [$\AA$]')

ax = plt.subplot2grid((12, 1), (8, 0), rowspan=4)
plt.plot(wls_modit, np.abs(xs_modit_lp/xsv_lpf_lp-1.)*100,
         lw=1, alpha=0.5, color='C1', label='MODIT (F32)')
plt.plot(wls_modit, np.abs(xs_modit_lp_f64/xsv_lpf_lp-1.) *
         100, lw=1, alpha=1, color='C2', label='MODIT (F64)')
plt.yscale('log')
plt.ylabel('difference (%)', fontsize=10)
plt.xlim(llow*10.0-tip, lhigh*10.0+tip)
plt.ylim(0.01, 100.0)
plt.xlabel('wavelength [$\AA$]')
plt.legend(loc='upper left')

plt.savefig('comparison_modit.png', bbox_inches='tight', pad_inches=0.0)
plt.savefig('comparison_modit.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()
