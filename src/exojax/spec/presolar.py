"""PreSOLAR Precomputing Shape density and OverLap Add convolution Rxxxx

"""
import numpy as np
import jax.numpy as jnp
from exojax.spec.shapefilter import compute_filter_length


def generate_reshaped_lbd(line_strength_ref, nu_lines, nu_grid, ngamma_ref,
                 ngamma_ref_grid, n_Texp, n_Texp_grid, elower, elower_grid,
                 Ttyp):
    """generate log-biased line shape density (LBD)

    Args:
        line_strength_ref (_type_): _description_
        nu_lines (_type_): _description_
        nu_grid (_type_): _description_
        ngamma_ref (_type_): _description_
        ngamma_ref_grid (_type_): _description_
        n_Texp (_type_): _description_
        n_Texp_grid (_type_): _description_
        elower (_type_): _description_
        elower_grid (_type_): _description_
        Ttyp (_type_): _description_

    Returns:
        jnp array: reshaped log-biased line shape density (rsLBD)
        jnp.array: multi_index_uniqgrid (number of unique broadpar, 2)
        
    Examples:

        >>> lbd, multi_index_uniqgrid = generate_reshaped_lbd(mdb.Sij0, mdb.nu_lines, nu_grid, ngamma_ref,
        >>>               ngamma_ref_grid, mdb.n_Texp, n_Texp_grid, mdb.elower,
        >>>               elower_grid, Ttyp)
        >>> ngamma_ref = ngamma_ref_grid[multi_index_uniqgrid[:,0]] # ngamma ref for the unique broad par
        >>> n_Texp = n_Texp_grid[multi_index_uniqgrid[:,0]] # temperature exponent for the unique broad par
        
    """
    logmin = -np.inf
    cont_nu, index_nu = npgetix(nu_lines, nu_grid)
    cont_elower, index_elower = npgetix_exp(elower, elower_grid, Ttyp)
    multi_index_lines, multi_cont_lines, uidx_bp, neighbor_uidx, multi_index_uniqgrid, Ng_broadpar = broadpar_getix(
        ngamma_ref, ngamma_ref_grid, n_Texp, n_Texp_grid)

    Ng_nu = len(nu_grid)

    # We extend the LBD grid to +1 along elower direction. See #273
    Ng_elower_plus_one = len(elower_grid) + 1

    lbd = np.zeros((Ng_nu, Ng_broadpar, Ng_elower_plus_one), dtype=np.float64)
    lbd = npadd3D_multi_index(lbd, line_strength_ref, cont_nu, index_nu,
                              cont_elower, index_elower, uidx_bp,
                              multi_cont_lines, neighbor_uidx)
    lbd[lbd > 0.0] = np.log(lbd[lbd > 0.0])
    lbd[lbd == 0.0] = logmin

    # Removing the extended grid of elower. See #273
    lbd = lbd[:, :, 0:-1]

    return jnp.array(lbd), multi_index_uniqgrid

