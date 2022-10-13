from jax import custom_jvp
import jax.numpy as jnp

@custom_jvp
def rotkernel(x, u1, u2):
    """rotation kernel w/ the quadratic Limb dwarkening law, numerator of (54) in Kawahara+2022

    Args:
        x: x variable
        u1: Limb-darkening coefficient 1
        u2: Limb-darkening coefficient 2

    Return:
        rotational kernel
    """    
    x2 = x*x
    kernel = jnp.where(x2 <= 1.0, jnp.pi/2.0*u1*(1.0 - x2) -
                       2.0/3.0*jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2+2.0*u2*x2), 0.0)
    return kernel


@rotkernel.defjvp
def rotkernel_jvp(primals, tangents):
    x, u1, u2 = primals
    ux, uu1, uu2 = tangents
    x2 = x*x
    dHdx = jnp.where(x2 <= 1.0, - jnp.pi*x*u1 +
                    2.0/3.0*x/jnp.sqrt(1.0-x2)*(-3.0+3.0*u1+u2+2.0*u2*x2)+8.0*x*u2*jnp.sqrt(1.0-x2), 0.0)
    dHdu1= jnp.where(x2 <= 1.0, -2.0*jnp.sqrt(1.0-x2)+jnp.pi/2.0*(1.0-x2), 0.0)
    dHdu2= jnp.where(x2 <= 1.0, -2.0*(1.0+2.0*x2)*(jnp.sqrt(1.0-x2))/3.0, 0.0)
    
    primal_out = rotkernel(x, u1, u2)
    tangent_out = dHdx * ux + dHdu1 * uu1 + dHdu2 * uu2
    return primal_out, tangent_out

