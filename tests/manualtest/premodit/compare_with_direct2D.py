"""This code compores premodit with direct computation in terms LSD of the line strength, ignoring the broadening parameters.

"""
import numpy as np
import jax.numpy as jnp

def compare_with_direct2D(mdb,nus,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0):
    """ compare the premodit LSD with the direct computation of LSD

    """
    from exojax.spec.lsd import npadd1D, npgetix
    from exojax.spec.hitran import SijT        
    from exojax.spec.premodit import make_elower_grid, make_lbd2D, unbiased_lsd

    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    lbd=make_lbd2D(mdb.Sij0, mdb.nu_lines, nus, mdb._elower, elower_grid, Ttyp)    
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdb.qr_interp)
    
    cont_inilsd_nu, index_inilsd_nu = npgetix(mdb.nu_lines, nus)
    qT = mdb.qr_interp(Ttest)
    logsij0 = jnp.array(np.log(mdb.Sij0))
    S=SijT(Ttest, logsij0, mdb.nu_lines, mdb._elower, qT)
    Slsd_direct = np.zeros_like(nus,dtype=np.float64)
    Slsd_direct = npadd1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
    print("Number of the E_lower grid=",len(elower_grid))
    maxdev=np.max(np.abs(Slsd/Slsd_direct-1.0))
    print("max deviation=",maxdev)
    assert np.abs(maxdev) < 0.05
    return Slsd, Slsd_direct

def test_2d():
    from exojax.spec import moldb
    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus, gpu_transfer=False)
    Slsd,Slsd_direct=compare_with_direct2D(mdbCH4,nus,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0)    
    return Slsd, Slsd_direct



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Slsd, Slsd_direct=test_2d()
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot((Slsd),alpha=0.3)
    plt.plot((Slsd_direct),alpha=0.3)
    plt.yscale("log")
    ax=fig.add_subplot(212)
    plt.plot((Slsd/Slsd_direct-1.0),".",alpha=0.3)
    plt.xlabel("wavenumber bin")
    plt.ylabel("error (premodit - direct)/direct")
    plt.show()