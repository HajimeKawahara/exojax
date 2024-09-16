import jax.numpy as jnp
from jax import grad
from exojax.atm.amclouds import get_smooth_index, get_value_at_smooth_index
    
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


def a_from_smooth_index(x):
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    smooth_index = get_smooth_index(a, x)
    return get_value_at_smooth_index(a, smooth_index)

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