"""comparison of the errors for exp(-9/2 log^2 sigmag) for FP32 jax

Note:
    this code was used to determine the optimal function of exp(-9/2 * log^2 sigmag) used in layeropacity.layer_optical_depth_clouds_lognormal
    The conclusion is that we should use f(sig) in the following code.
"""

import jax.numpy as jnp
import numpy as np

def gnp(sig):
    logs = np.log(sig)
    return np.exp(-4.5*logs**2)

def f0(sig):
    return (sig**(-4.5*jnp.log(sig)))

def f(sig):
    return sig**(jnp.log(sig**-4.5))

def g(sig):
    logs = jnp.log(sig)
    return jnp.exp(-4.5*logs**2)


if __name__ == "__main__":
    arr = np.logspace(0,3,10001)
    
    print(len(arr[f0(arr)>0.0]))    
    print(len(arr[f(arr)>0.0]))
    print(len(arr[g(arr)>0.0]))
    print(np.min(arr[f(arr)>0.0]))
    print(np.max(arr[f(arr)>0.0]))
    farr = f(arr)
    print(np.min(farr[farr>0.0]))
    print(np.max(farr))
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
#    plt.plot(arr,np.log(arr))
    plt.plot(arr,gnp(arr))
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(1.0,color="gray",ls="dashed")
    ax = fig.add_subplot(212)
    plt.plot(arr,g(arr)/gnp(arr)-1.0,label="exp",alpha=0.3)
    plt.plot(arr,f0(arr)/gnp(arr)-1.0,label="sig**(-4.5*jnp.log(sig))",alpha=0.3)
    plt.plot(arr,f(arr)/gnp(arr)-1.0,label="sig**(jnp.log(sig**-4.5))",alpha=0.7)
    plt.xscale("log")
    plt.ylim(-2.e-5,2.e-5)
    plt.legend()
    plt.show()

    
