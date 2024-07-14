import jax.numpy as jnp
from jax import grad

    
def a_from_searchsorted(x):
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ind = jnp.searchsorted(a, x)
    return a[ind]



def searchsorted_is_null_derivative():
    xarr, farr, dfarr = check_derivative(a_from_searchsorted)
    import matplotlib.pyplot as plt
    plt.plot(xarr,farr,".",label="searchsorted(x)") 
    plt.plot(xarr,dfarr,".",label="grad(searchsorted(x))") 
    plt.legend()
    plt.xlabel("x")
    plt.savefig("searchsorted.png")
    plt.show()

def check_derivative(a_from_func):
    d_a_from_func = grad(a_from_func)
    xarr = jnp.linspace(1.1, 4.9, 100)
    farr = []
    dfarr = []
    for x in xarr:
        farr.append(a_from_func(x))
        dfarr.append(d_a_from_func(x))
    farr = jnp.array(farr)
    dfarr = jnp.array(dfarr)
    return xarr,farr,dfarr

# xp = log10(pressures)
# x = log10(pressure)

def get_smooth_index(xp, x):
    findex = jnp.arange(len(xp), dtype=float)
    smooth_index = jnp.interp(x, xp, findex)
    return smooth_index

#array = log10(pressures) or temperatures

def value_at_smooth_index(array, smooth_index):
    ind = int(smooth_index)
    res = smooth_index - float(ind)
    return (1.0 - res)*array[ind] + res*array[ind+1]

def a_from_smooth_index(x):
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smooth_index = get_smooth_index(a, x)
    return value_at_smooth_index(a, smooth_index)

def smooth_index_is_not_null_derivative():
    xarr, farr, dfarr = check_derivative(a_from_smooth_index)
    import matplotlib.pyplot as plt
    plt.plot(xarr,farr,".",label="smoothindex(x)") 
    plt.plot(xarr,dfarr,".",label="grad(smoothindex(x))") 
    plt.legend()
    plt.xlabel("x")
    plt.savefig("smoothindex.png")
    plt.show()


if __name__ == "__main__":
    searchsorted_is_null_derivative()
    smooth_index_is_not_null_derivative()