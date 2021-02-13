from exojax.spec.lpf import xsvector
from exojax.spec.make_numatrix import make_numatrix0
import numpy as np
import tqdm

def xsection(nu,nu0,sigmaD,gammaL,Sij,memory_size=15.):
    """compute cross section

    Args:
       nu: wavenumber array
       nu0: line center
       sigmaD: sigma parameter in Doppler profile 
       gammaL:  broadening coefficient in Lorentz profile 
       Sij: line strength
       memory_size: memory size for numatrix0 (MB)

    Returns:
       xsv: cross section

    """
    d=int(memory_size/(len(nu0)*4/1024./1024.))
    Ni=int(len(nu)/d)
    xsv=[]
    for i in tqdm.tqdm(range(0,Ni+1)):
        s=int(i*d);e=int((i+1)*d);e=min(e,len(nu))
        numatrix=make_numatrix0(nu[s:e],nu0)
        xsv = np.concatenate([xsv,xsvector(numatrix,sigmaD,gammaL,Sij)])
    return xsv
