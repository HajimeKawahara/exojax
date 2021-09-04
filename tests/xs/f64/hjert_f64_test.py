""" test for hjert
    
   - This test compares hjert with scipy wofz, see Appendix in Paper I.


"""

import pytest
from exojax.spec import hjert
from scipy.special import wofz as sc_wofz
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from jax.config import config                                                 #
config.update("jax_enable_x64", True)

def test_comparison_hjert_scipy():

    Na=300
    vl=-3
    vm=5
    xarrv=jnp.logspace(vl,vm,Na)
    xarr=xarrv[:,None]*jnp.ones((Na,Na))
    aarrv=jnp.logspace(vl,vm,Na)
    aarr=aarrv[None,:]*jnp.ones((Na,Na))
    
    #scipy
    def H(a,x):
        z=x+(1j)*a
        w = sc_wofz(z)
        return w.real

    # hjert
    def vhjert(a):
        return vmap(hjert,(0,None),0)(xarrv,a)
    
    vvhjert=jit(vmap(vhjert,0,0))
    diffarr=(vvhjert(aarrv).T-H(aarr,xarr))/H(aarr,xarr)
    print("MEDIAN=",np.median(diffarr),"MAX=",np.max(diffarr))

    #figure
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    fig=plt.figure()
    ax=fig.add_subplot(111)
    c=ax.imshow((vvhjert(aarrv).T-H(aarr,xarr))/H(aarr,xarr),vmin=-3.e-7,vmax=3.e-7,
                cmap="RdBu",extent=([vl,vm,vm,vl]),rasterized=True)
    plt.gca().invert_yaxis()
    plt.ylabel("$\log_{10}(x)$")
    plt.xlabel("$\log_{10}(a)$")
    cb=plt.colorbar(c)
    cb.formatter.set_powerlimits((0, 0))
    cb.set_label("(hjert - scipy)/scipy",size=14)
    plt.savefig("hjert_f64.png", bbox_inches="tight", pad_inches=0.0)
    plt.savefig("hjert_f64.pdf", bbox_inches="tight", pad_inches=0.0)

    assert np.max(diffarr)<1.e-6

if __name__ == "__main__":
    test_comparison_hjert_scipy()
