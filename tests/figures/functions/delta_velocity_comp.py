import jax.numpy as jnp
import numpy as np

def dv_comparison():
    from jax.config import config
    #config.update("jax_enable_x64", True)
    import matplotlib.pyplot as plt
        
    N = 60
    Rarray = np.logspace(2,7,N)
    dv_np = np.log(1.0/Rarray + 1.0)
    dv_jnp = jnp.log(1.0/Rarray + 1.0)
    dv_jnp_log1p = jnp.log1p(1.0/Rarray)
    plt.plot(Rarray,np.abs(dv_jnp_log1p/dv_np-1.0),".",label="jnp.log1p(1/R) (FP32)")
    plt.plot(Rarray,np.abs(dv_jnp/dv_np-1.0),".",label="jnp.log(1/R + 1.0) (FP32)")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("R")
    plt.ylabel("difference from np.log(1.0/R + 1.0)")
    plt.legend()
    plt.savefig("delta_velocity_comp.png")
    plt.show()
    
if __name__ == "__main__":
    dv_comparison()
