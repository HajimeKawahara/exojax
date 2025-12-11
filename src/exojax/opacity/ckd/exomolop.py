"""Loading a petitRADTRANS correlated-k opacity HDF5 file.

A part of this file is originally based on load_hdf5_ktables in petitRADTRANS under the MIT license.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np
import jax.numpy as jnp
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.provider.exomolop import load_ckd

def load_exomolop_ckd(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""

    xsgrid, samples, weights, temperatures, pressures, wavenumber, molecule, mol_mass = load_ckd(path)

    ckdinfo = CKDTableInfo(
        log_kggrid=jnp.log(jnp.array(xsgrid)),
        ggrid=jnp.array(samples),
        weights=jnp.array(weights),
        T_grid=jnp.array(temperatures),
        P_grid=jnp.array(pressures),
        nu_bands=jnp.array(wavenumber),
        band_edges=jnp.array([]),  # Not used in this context
    )


    return ckdinfo