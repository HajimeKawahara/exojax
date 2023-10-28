import jax.numpy as jnp

def refractive_index_Lorentz_Lorenz(polarizability, number_density):
    """Refractive index using Lorentz-Lorenz relataion
    
    Notes:
        See D.25 in Liou 2002, for instance.

    Args:
        polarizability: polarizability (cm3)
        number_density: number density of molecule (cm-3)

    Returns:
        refractive index

    """
    fac = 4.0*jnp.pi*polarizability*number_density/3.0
    return jnp.sqrt((1.0 + 2.0*fac)/(1.0 - fac))