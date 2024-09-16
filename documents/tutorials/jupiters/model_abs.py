import jax.numpy as jnp


def unpack_params(params):
    multiple_factor = jnp.array([1.0, 1.0, 1.0, 10000.0, 0.01, 1.0])
    par = params * multiple_factor
    log_surface_pressure = par[0]
    vrv = par[1]
    vv = par[2]
    _broadening = par[3]
    const_mmr_ch4 = par[4]
    factor = par[5]
    surface_pressure = 10**log_surface_pressure
    return surface_pressure, vrv, vv, _broadening, const_mmr_ch4, factor
