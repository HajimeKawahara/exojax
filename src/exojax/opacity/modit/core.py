"""Core functionality for MODIT opacity calculations."""

from typing import Optional, Union, List

import jax.numpy as jnp
import numpy as np


def _setdgm(
    mdb,
    dit_grid_resolution: float,
    R: float,
    Tarr_list: Union[np.ndarray, List[np.ndarray]],
    Parr: np.ndarray,
    Pself_ref: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Set up DIT grid matrix (dgm) for gammaL.

    Args:
        mdb: Molecular database (mdbExomol, mdbHitemp, mdbHitran)
        dit_grid_resolution: DIT grid resolution
        R: Spectral resolution (float)
        Tarr_list: Temperature array(s) to be tested. Can be 1D or 2D array
                  such as [Tarr_1, Tarr_2, ..., Tarr_n]
        Parr: Pressure array in bar (1D)
        Pself_ref: Self pressure array in bar (1D), optional.
                  If None, defaults to zeros with same shape as Parr

    Returns:
        DIT grid matrix for gammaL as JAX array
        
    Raises:
        ValueError: If database type is not supported
    """
    from exojax.opacity.modit.modit import exomol, hitran
    from exojax.opacity._common.set_ditgrid import (
        minmax_ditgrid_matrix,
        precompute_modit_ditgrid_matrix,
    )

    # Ensure Tarr_list is 2D array for consistent processing
    if len(np.shape(Tarr_list)) == 1:
        Tarr_list = np.array([Tarr_list])
    
    # Set default self-pressure if not provided
    if Pself_ref is None:
        Pself_ref = np.zeros_like(Parr)

    # Compute DIT grid matrices for each temperature array
    set_dgm_minmax = []
    dbtype = mdb.dbtype
    
    for Tarr in Tarr_list:
        if dbtype == "exomol":
            _, ngammaLM, _ = exomol(mdb, Tarr, Parr, R, mdb.molmass)
        elif dbtype == "hitran":
            _, ngammaLM, _ = hitran(mdb, Tarr, Parr, Pself_ref, R, mdb.molmass)
        else:
            raise ValueError(
                f"Unsupported database type: '{dbtype}'. "
                "Supported types: exomol, hitran"
            )
            
        set_dgm_minmax.append(
            minmax_ditgrid_matrix(ngammaLM, dit_grid_resolution)
        )
    
    # Precompute the final DIT grid matrix
    dgm_ngammaL = precompute_modit_ditgrid_matrix(
        set_dgm_minmax, dit_grid_resolution=dit_grid_resolution
    )
    
    return jnp.array(dgm_ngammaL)