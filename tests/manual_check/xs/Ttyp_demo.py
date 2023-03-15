from exojax.spec.lpf import auto_xsection
from exojax.spec import line_strength, doppler_sigma,  gamma_natural
from exojax.spec.exomol import gamma_exomol
from exojax.spec import moldb
import numpy as np

def demo(Tfix,Ttyp,crit=1.e-40):
    """reproduce userguide/moldb.html#masking-weak-lines

    Args:
       Tfix: gas temperature
       Ttyp: Ttyp for line strength criterion
       crit: line strength criterion

    Returns
       nus, xsv

    """
    
    nus=np.linspace(1000.0,10000.0,900000,dtype=np.float64) #cm-1
    mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus, Ttyp=Ttyp, crit=crit)
    Mmol=28.010446441149536 # molecular weight
    Pfix=1.e-3 # we compute P=1.e-3 bar
    qt=mdbCO.qr_interp(Tfix)
    Sij=line_strength(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    gammaL = gamma_exomol(Pfix,Tfix,mdbCO.n_Texp,mdbCO.alpha_ref)\
        + gamma_natural(mdbCO.A)
    # thermal doppler sigma
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    #line center
    nu0=mdbCO.nu_lines
    xsv=auto_xsection(nus,nu0,sigmaD,gammaL,Sij,memory_size=30)
    return nus, xsv

if __name__=="__main__":
    Tfix=2000.0
    Ttyp=1000.
    nus,xsv=demo(Tfix,Ttyp,crit=1.e-40)
    import matplotlib.pyplot as plt
    plt.plot(1.e4/nus,xsv)
    plt.yscale("log")
    plt.show()
