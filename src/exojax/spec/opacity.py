from exojax.spec.lpf import xsvector
from exojax.spec.make_numatrix import make_numatrix0
import numpy as np
import tqdm

def xsection(nu,nu0,sigmaD,gammaL,Sij,memory_size=15.):
    """compute cross section

    Note:
       This is not auto-differentiable routine.

    Args:
       nu: wavenumber array
       nu0: line center
       sigmaD: sigma parameter in Doppler profile 
       gammaL:  broadening coefficient in Lorentz profile 
       Sij: line strength
       memory_size: memory size for numatrix0 (MB)

    Returns:
       xsv: cross section

    Examples:

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
    >>> qt=mdbCO.Qr_line(Tfix)
    >>> Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    >>> gammaL = gamma_hitran(Pfix,Tfix, Ppart, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self) + gamma_natural(mdbCO.A) 
    >>> sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    >>> nu0=mdbCO.nu_lines
    >>> xsv=xsection(nus,nu0,sigmaD,gammaL,Sij,memory_size=30)
        100%|████████████████████████████████████████████████████| 456/456 [00:03<00:00, 80.59it/s]

    """
    d=int(memory_size/(len(nu0)*4/1024./1024.))
    Ni=int(len(nu)/d)
    xsv=[]
    for i in tqdm.tqdm(range(0,Ni+1)):
        s=int(i*d);e=int((i+1)*d);e=min(e,len(nu))
        numatrix=make_numatrix0(nu[s:e],nu0)
        xsv = np.concatenate([xsv,xsvector(numatrix,sigmaD,gammaL,Sij)])
    return xsv
