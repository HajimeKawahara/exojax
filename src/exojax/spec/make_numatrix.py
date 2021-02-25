from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

def divwavnum(nu,Nz=1):
    """separate an integer part and a residual
    """

    fn=np.floor(nu*Nz)
    dfn=nu*Nz-fn
    return fn,dfn,Nz

@jit
def subtract_nu(dnu,dhatnu):
    """compute nu - hatnu using subtract an integer part w/JIT
    """
    jdnu=jnp.array(dnu)
    jdhatnu=jnp.array(dhatnu)
    dd=(jdnu[None,:]-jdhatnu[:,None])
    return dd

@jit
def add_nu(dd,fnu,fhatnu,Nz):
    """re-adding an interger part w/JIT
    """
    jfnu=jnp.array(fnu)
    jfhatnu=jnp.array(fhatnu)
    intarray=fnu[None,:]-fhatnu[:,None]
    return (dd+intarray)/Nz

def make_numatrix0(nu,hatnu,Nz=1,warning=True):
    """Generate numatrix0

    Note: 
       This function computes a wavenumber matrix using XLA. Because XLA does not support float64, a direct computation sometimes results in large uncertainity. For instace, let's assum nu=2000.0396123 cm-1 and hatnu=2000.0396122 cm-1. If applying float32, we get np.float32(2000.0396123)-np.float32(2000.0396122) = 0.0. But, after subtracting 2000 from both nu and hatnu, we get np.float32(0.0396123)-np.float32(0.0396122)=1.0058284e-07. make_numatrix0 does such computation. Nz=1 means we subtract a integer part (i.e. 2000), Nz=10 means we subtract 2000.0, and Nz=10 means we subtract 2000.00.

    Args:
       nu: wavenumber matrix (Nnu,)
       hatnu: line center wavenumber vector (Nline,), where Nm is the number of lines
       Nz: boost factor (default=1)
       warning: True=warning on for nu.dtype=float32

    Returns:
       numatrix0: wavenumber matrix w/ no shift

    """

    if(nu.dtype!=np.float64 and warning):
        print("Warning!: nu is not np.float64 but ",nu.dtype)
        # Float32 significantly decreases the accuracy of opacity.
        # Consider to use float64 for wavenumber array.
    
    fnu,dnu,Nz=divwavnum(nu,Nz)
    fhatnu,dhatnu,Nz=divwavnum(hatnu,Nz)
    dd=subtract_nu(dnu,dhatnu)
    numatrix0=add_nu(dd,fnu,fhatnu,Nz)
    return numatrix0


@jit
def make_numatrix_direct(nu,hatnu,nu0):
    """Generate numatrix (directly)

    Note: 
       This routine autmatically convert the input to float32 to use XLA. Please check nu/your precision is much smaller than 1e-7. Otherwise, use make_numatrix0.

    Args:
       nu: wavenumber matrix (Nnu,)
       hatnu: line center wavenumber vector (Nline,), where Nm is the number of lines
       nu0: nu0

    Returns:
       f: numatrix (Nline,Nnu)

    """
    numatrix=nu[None,:]-hatnu[:,None]-nu0
    return numatrix
