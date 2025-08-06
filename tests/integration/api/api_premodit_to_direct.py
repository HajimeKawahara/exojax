
"""
Uses OpaDIrect after calling PreModit #437 made by @ykawashima (see #437, #438, #439) 
"""

from exojax.utils.grids import wavenumber_grid
from exojax.database.exomol.api import MdbExomol 
from exojax.database import molinfo 
from exojax.database.core.line_strength import line_strength, doppler_sigma, gamma_hitran, gamma_natural, line_strength_numpy
from exojax.opacity import OpaPremodit, OpaDirect
import numpy as np
import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)


from exojax.opacity.lpf.lpf import xsvector, make_numatrix0
import tqdm
def auto_xsection(nu, nu_lines, sigmaD, gammaL, Sij, memory_size=15.):
    """computes cross section .

    Warning:
        This is NOT auto-differentiable function.

    Args:
        nu: wavenumber array
        nu_lines: line center
        sigmaD: sigma parameter in Doppler profile 
        gammaL:  broadening coefficient in Lorentz profile 
        Sij: line strength
        memory_size: memory size for numatrix0 (MB)

    Returns:
        numpy.array: cross section (xsv)

    Example:
        >>> from exojax.opacity.lpf.lpf import auto_xsection
        >>> from exojax.database.hitran  import SijT, doppler_sigma, gamma_hitran, gamma_natural
        >>> from exojax.database import moldb 
        >>> import numpy as np
        >>> nus=np.linspace(1000.0,10000.0,900000,dtype=np.float64) #cm-1
        >>> mdbCO=moldb.MdbHit('~/exojax/data/CO','05_hit12',nus)
        >>> Mmol=28.010446441149536 # molecular weight
        >>> Tfix=1000.0 # we assume T=1000K
        >>> Pfix=1.e-3 # we compute P=1.e-3 bar
        >>> Ppart=Pfix #partial pressure of CO. here we assume a 100% CO atmosphere. 
        >>> qt=mdbCO.qr_interp_lines(Tfix)
        >>> Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
        >>> gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self) + gamma_natural(mdbCO.A) 
        >>> sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
        >>> nu_lines=mdbCO.nu_lines
        >>> xsv=auto_xsection(nus,nu_lines,sigmaD,gammaL,Sij,memory_size=30)
        100%|████████████████████████████████████████████████████| 456/456 [00:03<00:00, 80.59it/s]
    """
    NL = len(nu_lines)
    d = int(memory_size/(NL*4/1024./1024.))
    if d > 0:
        Ni = int(len(nu)/d)
        xsv = []
        for i in tqdm.tqdm(range(0, Ni+1)):
            s = int(i*d)
            e = int((i+1)*d)
            e = min(e, len(nu))
            numatrix = make_numatrix0(nu[s:e], nu_lines, warning=False)
            xsv = np.concatenate(
                [xsv, xsvector(numatrix, sigmaD, gammaL, Sij)])
    else:
        NP = int((NL*4/1024./1024.)/memory_size)+1
        d = int(memory_size/(int(NL/NP)*4/1024./1024.))
        Ni = int(len(nu)/d)
        dd = int(NL/NP)
        xsv = []
        for i in tqdm.tqdm(range(0, Ni+1)):
            s = int(i*d)
            e = int((i+1)*d)
            e = min(e, len(nu))
            xsvtmp = np.zeros_like(nu[s:e])
            for j in range(0, NP+1):
                ss = int(j*dd)
                ee = int((j+1)*dd)
                ee = min(ee, NL)
                numatrix = make_numatrix0(
                    nu[s:e], nu_lines[ss:ee], warning=False)
                xsvtmp = xsvtmp + \
                    xsvector(numatrix, sigmaD[ss:ee],
                             gammaL[ss:ee], Sij[ss:ee])
            xsv = np.concatenate([xsv, xsvtmp])

    return xsv



nus, wav, res = wavenumber_grid(22980.0,
                                23030.0,
                                100000,
                                unit='AA',
                                xsmode="premodit")

mdb = MdbHitemp(".database/CO/05_HITEMP2019",nus,crit=1.e-30,Ttyp=1000.,gpu_transfer=True,isotope=1)

P = 1.e-3
T = 1000.
vmr = 1.
Ppart = P * vmr
Mmol = molinfo.molmass("CO")

logsij0 = np.log(mdb.line_strength)
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

logsij0 = np.log(mdb.line_strength)
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



