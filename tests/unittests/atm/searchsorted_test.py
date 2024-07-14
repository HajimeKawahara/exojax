import jax.numpy as jnp


def a_from_searchsorted(x):
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ind = jnp.searchsorted(a, x)
    return a[ind]


# def test_searchsorted():
#    ax = a_from_searchsorted(2.3)

if __name__ == "__main__":
    from jax import grad

    d_a_from_searchsorted = grad(a_from_searchsorted)

    print(d_a_from_searchsorted(2.3))
    xarr = jnp.linspace(1.1, 4.9, 100)
    farr = []
    dfarr = []
    for x in xarr:
        farr.append(a_from_searchsorted(x))
        dfarr.append(d_a_from_searchsorted(x))
    farr = jnp.array(farr)
    dfarr = jnp.array(dfarr)

    import matplotlib.pyplot as plt
    plt.plot(xarr,farr,".",label="searchsorted(x)") 
    plt.plot(xarr,dfarr,".",label="grad(searchsorted(x))") 
    plt.legend()
    plt.xlabel("x")
    plt.savefig("searchsorted.png")
    plt.show()