"""Loading a petitRADTRANS correlated-k opacity HDF5 file.

A part of this file is originally based on load_hdf5_ktables in petitRADTRANS under the MIT license.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np


def load_ktable(path: Path):
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
    xsgrid[xsgrid < 0.0] = 0.0

    return {
        "molecule": molecule,
        "molecular_mass_amu": mol_mass,
        "temperatures_K": temperatures,
        "pressures_bar": pressures,
        "g_samples": samples,
        "g_weights": weights,
        "wavenumber_cm-1": wavenumber,
        "xs_grid": xsgrid,  # shape: (T, P, g, wavenumber)
    }



if __name__ == "__main__":
    from jax import config 
    config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description="Load a petitRADTRANS correlated-k opacity .h5 file.")
    parser.add_argument(
        "h5_file",
        type=Path,
        help="Path to a *.ktable.petitRADTRANS.h5 file (e.g., 12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5)",
    )
    args = parser.parse_args()
    info = load_ktable(args.h5_file)

    from exojax.utils.grids import wavenumber_grid
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit

    nu_grid, wav, resolution = wavenumber_grid(22920.0, 24000.0, 100000, unit="AA", xsmode="premodit")
    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)

    molmass = mdb.molmass # we use molmass later
    snap = mdb.to_snapshot() # extract snapshot from mdb
    del mdb # save the memory

    opa = OpaPremodit.from_snapshot(
        snap,
        nu_grid,
        auto_trange=(500.0, 1500.0),
        dit_grid_resolution=1.0,
    )


    import matplotlib.pyplot as plt
    iT = 10
    jP = 10
    temperature = info['temperatures_K'][iT]
    pressure = info['pressures_bar'][jP]
    print(temperature, "K", pressure,"bar")
    
    xsv = opa.xsvector(temperature, pressure)
    
    ktable = info['xs_grid']
    nu_ckd = info['wavenumber_cm-1']
    wav_ckd = 1e4/nu_ckd  # micron
    print(np.min(nu_ckd), np.max(nu_ckd))
    print(np.min(wav_ckd), np.max(wav_ckd))
    xs_ckd = np.sum(ktable[iT, jP, :, :] * info['g_weights'][:, np.newaxis], axis=0)
    
    fig = plt.figure(figsize=(12,4))
    plt.plot(wav_ckd*1.e4, xs_ckd, label="ckd", alpha=0.7)
    plt.plot(wav, xsv, label="premodit", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Wavelength [angstrom]")
    plt.ylabel("Cross section [cm$^2$]")
    plt.legend()
    plt.xlim(22000,24000)
    plt.tight_layout()
    plt.show()

