from exojax.atm.lorentz_lorenz import refractive_index_Lorentz_Lorenz

import jax.numpy as jnp

def test_refractive_index_Lorentz_Lorenz():
    polarizability = 1e-24  # cm^3
    number_density = 1e19   # cm^-3
    expected_refractive_index = jnp.sqrt((1.0 + 2.0 * 4.0 * jnp.pi * polarizability * number_density / 3.0) / 
                                         (1.0 - 4.0 * jnp.pi * polarizability * number_density / 3.0))
    
    calculated_refractive_index = refractive_index_Lorentz_Lorenz(polarizability, number_density)
    assert jnp.isclose(calculated_refractive_index, expected_refractive_index), \
        f"Expected {expected_refractive_index}, but got {calculated_refractive_index}"
