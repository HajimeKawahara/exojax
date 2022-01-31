import jax.numpy as jnp
from jax import jit
from jax.lax import scan


@jit
def erfcx(x):
    """erfcx (float) based on Shepherd and Laframboise (1981)

    Scaled complementary error function exp(-x*x) erfc(x)

    Args:
         x: should be larger than -9.3

    Returns:
         jnp.array: erfcx(x)

    Note:
       We acknowledge the post in stack overflow (https://stackoverflow.com/questions/39777360/accurate-computation-of-scaled-complementary-error-function-erfcx).
    """
    a = jnp.abs(x)
    q = (-a*(a-2.0)/(a+2.0)-2.0*((a-2.0)/(a+2.0)+1.0)+a) / \
        (a+2.0) + (a-2.0)/(a+2.0)
    p = (((((((((((5.92470169e-5*q+1.61224554e-4)*q-3.46481771e-4)*q-1.39681227e-3)*q+1.20588380e-3)*q+8.69014394e-3)
         * q-8.01387429e-3)*q-5.42122945e-2)*q+1.64048523e-1)*q-1.66031078e-1)*q-9.27637145e-2)*q+2.76978403e-1)

    q = (p+1.0)/(1.0+2.0*a)
    d = (p+1.0)-q*(1.0+2.0*a)
    f = 0.5*d/(a+0.5) + q
    f = jnp.where(x >= 0.0, f, 2.0*jnp.exp(x**2) - f)

    return f
