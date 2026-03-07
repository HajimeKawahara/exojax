from dataclasses import dataclass

import jax.numpy as jnp
@dataclass(frozen=True)
class CKDTableInfo:
    """Immutable container for CKD table information.

    Attributes:
        log_kggrid: Log k-values on g-grid, shape (nT, nP, Ng, nnu_bands)
        ggrid: Gauss-Legendre g-ordinates, shape (Ng,)
        weights: Gauss-Legendre quadrature weights, shape (Ng,)
        T_grid: Temperature grid, shape (nT,)
        P_grid: Pressure grid, shape (nP,)
        nu_bands: Wavenumber band centers, shape (nnu_bands,)
        band_edges: Wavenumber band edges, shape (nnu_bands, 2)
    """

    log_kggrid: jnp.ndarray
    ggrid: jnp.ndarray
    weights: jnp.ndarray
    T_grid: jnp.ndarray
    P_grid: jnp.ndarray
    nu_bands: jnp.ndarray
    band_edges: jnp.ndarray
