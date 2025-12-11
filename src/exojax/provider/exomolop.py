from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np
import jax.numpy as jnp

def load_ckd(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""
    with h5py.File(path, "r") as fh5:
        molecule = fh5["mol_name"][()][0].decode("utf-8")
        mol_mass = float(fh5["mol_mass"][()][0])
        wavenumber = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]    # K
        pressures = fh5["p"][:]       # bar
        kcoeff = np.array(fh5["kcoeff"])

    
    # reshape to (T, P, g, wavenumber)
    xsgrid = np.swapaxes(kcoeff, 0, 1)
    xsgrid = np.swapaxes(xsgrid, 2, 3)
    
    # Clip negative values
    tiny = np.finfo(xsgrid.dtype).tiny 
    xsgrid = np.where(xsgrid == 0, tiny, xsgrid)
    
    return xsgrid, samples, weights, temperatures, pressures, wavenumber, molecule, mol_mass

    