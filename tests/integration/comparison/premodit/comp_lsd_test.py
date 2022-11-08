"""Comparision test between Premodit and LPF

   * test for unbiased_lsd

"""
import numpy as np
from exojax.spec.premodit import make_broadpar_grid
from exojax.spec.premodit import generate_lbd
from exojax.utils.instfunc import resolution_eslog
import numpy as np
import jax.numpy as jnp

def npadd1D(a, w, cx, ix):
    """numpy version: Add into an array when contirbutions and indices are given (1D).

    Args:
        a: lineshape density (LSD) array (np.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        
    Returns:
        lineshape density a(nx)

    """
    np.add.at(a, (ix), w*(1-cx))
    np.add.at(a, (ix+1), w*cx)
    return a

def compare_line_shape_density(mdb,nu_grid,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0):
    """ compare the premodit LSD with the direct computation of LSD 3D version
    
    """
    from exojax.spec.lsd import npgetix
    from exojax.spec.hitran import SijT
    from exojax.spec.premodit import make_elower_grid
    from exojax.spec.premodit import unbiased_lsd
    
    dit_grid_resolution = 0.1    
    R = resolution_eslog(nu_grid)
    ngamma_ref = mdb.alpha_ref / mdb.nu_lines * R
    ngamma_ref_grid, n_Texp_grid = make_broadpar_grid(
        ngamma_ref, mdb.n_Texp, Ttyp, dit_grid_resolution=dit_grid_resolution)
    elower_grid = make_elower_grid(Ttyp,
                                   mdb.elower,
                                   interval_contrast=interval_contrast)
    lbd, multi_index_uniqgrid = generate_lbd(mdb.Sij0, mdb.nu_lines, nu_grid, ngamma_ref,
                       ngamma_ref_grid, mdb.n_Texp, n_Texp_grid, mdb.elower,
                       elower_grid, Ttyp)
    qT = mdb.qr_interp(Ttest)
    Slsd=unbiased_lsd(lbd,Ttest,nu_grid,elower_grid,qT)
    Slsd=np.sum(Slsd,axis=1)
    cont_inilsd_nu, index_inilsd_nu = npgetix(mdb.nu_lines, nu_grid)
    logsij0 = jnp.array(np.log(mdb.Sij0))
    S=SijT(Ttest, logsij0, mdb.nu_lines, mdb.elower, qT)
    Slsd_direct = np.zeros_like(nu_grid,dtype=np.float64)
    Slsd_direct = npadd1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
    return Slsd, Slsd_direct

def test_comp_lsd():
    from exojax.spec import moldb
    nu_grid=np.logspace(np.log10(6030.0), np.log10(6060.0), 20000, dtype=np.float64)
    mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nu_grid, gpu_transfer=False)
    Slsd,Slsd_direct=compare_line_shape_density(mdbCH4,nu_grid,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0)    
    maxdev=np.max(np.abs(Slsd/Slsd_direct-1.0))
    print("max deviation=",maxdev)
    assert np.abs(maxdev) < 0.05
    return Slsd, Slsd_direct


if __name__ == "__main__":
    Slsd, Slsd_direct=test_comp_lsd()
    import matplotlib.pyplot as plt
    plt.plot(Slsd,alpha=0.3)
    plt.plot(Slsd_direct,alpha=0.3)
    plt.yscale("log")
    plt.show()  
