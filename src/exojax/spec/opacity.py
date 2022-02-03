from exojax.spec.lpf import xsvector
from exojax.spec.make_numatrix import make_numatrix0
import numpy as np
import tqdm


def xsection(nu, nu_lines, sigmaD, gammaL, Sij, memory_size=15.):
    """compute cross section.

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
       >>> from exojax.spec import xsection
       >>> from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
       >>> from exojax.spec import moldb
       >>> import numpy as np
       >>> nus=np.linspace(1000.0,10000.0,900000,dtype=np.float64) #cm-1
       >>> mdbCO=moldb.MdbHit('/home/kawahara/exojax/data/CO','05_hit12',nus)
       >>> Mmol=28.010446441149536 # molecular weight
       >>> Tfix=1000.0 # we assume T=1000K
       >>> Pfix=1.e-3 # we compute P=1.e-3 bar
       >>> Ppart=Pfix #partial pressure of CO. here we assume a 100% CO atmosphere. 
       >>> qt=mdbCO.Qr_line_HAPI([Tfix])[0]
       >>> Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
       >>> gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self) + gamma_natural(mdbCO.A) 
       >>> sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
       >>> nu_lines=mdbCO.nu_lines
       >>> xsv=xsection(nus,nu_lines,sigmaD,gammaL,Sij,memory_size=30)
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

    if(nu.dtype != np.float64):
        print('Warning: nu is not np.float64 but ', nu.dtype)

    return xsv
