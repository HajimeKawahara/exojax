from jax import jit
import jax.numpy as jnp
import numpy as np


def make_numatrix0(nu,hatnu,warning=True):
    """Generate numatrix0

    Note: 
       This routine autmatically convert the input to float32 to use XLA. Please check nu/your precision is much smaller than 1e-7. Otherwise, use make_numatrix0.

    Args:
       nu: wavenumber matrix (Nnu,)
       hatnu: line center wavenumber vector (Nline,), where Nm is the number of lines
       warning: True=warning on for nu.dtype=float32

    Returns:
       f: numatrix (Nline,Nnu)

    """
    if(nu.dtype!=np.float64 and warning):
        print("Warning!: nu is not np.float64 but ",nu.dtype)
    if(hatnu.dtype!=np.float64 and warning):
        print("Warning!: hatnu is not np.float64 but ",nu.dtype)

    numatrix=nu[None,:]-hatnu[:,None]
    return jnp.array(numatrix)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exojax.spec.rtransfer import nugrid
    from exojax.spec import moldb
    import time
    import numpy as np
    nus,wav,res=nugrid(22920,23000,1000,unit="AA")
    mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus,crit=1.e-46)
    ts=time.time()
    numatrix=make_numatrix0(nus,mdbCO.nu_lines)
    print(np.median(numatrix))
    te=time.time()
    print(te-ts,"sec")

    ts=time.time()
    numatrixc=make_numatrix0_direct(nus,mdbCO.nu_lines)
    print(np.median(numatrix))
    te=time.time()
    print(te-ts,"sec")

    print(np.sum((numatrixc-numatrix)**2))
