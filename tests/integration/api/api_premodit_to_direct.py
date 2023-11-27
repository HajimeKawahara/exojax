
"""
Uses OpaDIrect after calling PreModit #437 made by @ykawashima (see #437, #438, #439) 
"""

from exojax.utils.grids import wavenumber_grid
from exojax.spec import api
from exojax.spec import molinfo
from exojax.spec.lpf import auto_xsection
from exojax.spec.hitran import line_strength, doppler_sigma, gamma_hitran, gamma_natural, line_strength_numpy
from exojax.spec.exomol import gamma_exomol
from exojax.spec.opacalc import OpaPremodit, OpaDirect
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

nus, wav, res = wavenumber_grid(22980.0,
                                23030.0,
                                100000,
                                unit='AA',
                                xsmode="premodit")

mdb = api.MdbHitemp(".database/CO/05_HITEMP2019",nus,crit=1.e-30,Ttyp=1000.,gpu_transfer=True,isotope=1)

P = 1.e-3
T = 1000.
vmr = 1.
Ppart = P * vmr
Mmol = molinfo.molmass("CO")

logsij0 = np.log(mdb.line_strength_ref)
sigmaD = doppler_sigma(mdb.nu_lines,T,Mmol)
qt = mdb.qr_interp(mdb.isotope, T)
gammaL = gamma_hitran(P,T, Ppart, mdb.n_air, mdb.gamma_air, mdb.gamma_self) + gamma_natural(mdb.A)
Sij = line_strength(T,logsij0,mdb.nu_lines,mdb.elower,qt, mdb.Tref)
#Sij = line_strength(T,logsij0,mdb.nu_lines,mdb.elower,qt)                                                                                                          
xsv0 = auto_xsection(np.array(nus),mdb.nu_lines,sigmaD,gammaL,Sij,memory_size=30)

opa = OpaPremodit(mdb=mdb,
                  nu_grid=np.array(nus),
                  diffmode=2,
                  auto_trange=[500., 1500.],
                  dit_grid_resolution=1.0,
                  allow_32bit=True)

opad = OpaDirect(mdb=mdb,
                 nu_grid=np.array(nus))

logsij0 = np.log(mdb.line_strength_ref)
qt = mdb.qr_interp(mdb.isotope, T)
Sij = line_strength(T,logsij0,mdb.nu_lines,mdb.elower,qt, mdb.Tref)
#Sij = line_strength(T,logsij0,mdb.nu_lines,mdb.elower,qt)                                                                                                          
xsv = auto_xsection(np.array(nus),mdb.nu_lines,sigmaD,gammaL,Sij,memory_size=30)

fig, ax = plt.subplots()
ax.plot(1.0e8/np.array(nus), xsv0, c='C0')
ax.plot(1.0e8/np.array(nus), xsv, c='C1')
ax.plot(1.0e8/np.array(nus), opa.xsvector(T, P), c='C2',ls="dashed")
ax.plot(1.0e8/np.array(nus), opad.xsvector(T, P), c='C3',ls="dotted")
ax.set_xlim(22985, 23025)
ax.set_yscale('log')
plt.show()