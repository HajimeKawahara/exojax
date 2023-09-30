import pytest
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def test_simpson():
    """ test simpson integral

    f = 0.01, (0.3), 1.0, (1.3), 2.0, (2.7), 3.0
    h = 0.7, 0.8, 0.9
    """
    f_lower = jnp.array([[1.0,2.0,3.0]]).T 
    print(jnp.shape(f_lower))
    f_top = jnp.array([0.01])
    f = jnp.array([[0.3, 1.3, 2.7]]).T
    h = jnp.array([0.7,0.8,0.9])
    simpson0 = h[0]*(f_top + 4.0*f[0,:] + f_lower[0,:])/6.0
    simpson1 = h[1]*(f_lower[0,:] + 4.0*f[1,:] + f_lower[1,:])/6.0
    simpson2 = h[2]*(f_lower[1,:] + 4.0*f[2,:] + f_lower[2,:])/6.0
    ref_integral = simpson0 + simpson1 + simpson2

    integral = simpson(f, f_lower, f_top, h)

    assert integral[0] == pytest.approx(ref_integral[0])

def simpson(f, f_lower, f_top, h):
    N=len(f)
    hh = jnp.roll(h, -1) + h  # h_{n+1} + h_n
    fac = hh[:N-1,None]*f_lower[:N-1,:]
    return 2.0/3.0*jnp.sum(h[:,None]*f) + h[0]*f_top/6.0 + h[-1]*f_lower[-1,:]/6.0 + jnp.sum(fac)/6.0
    

if __name__ == "__main__":
    test_simpson()

