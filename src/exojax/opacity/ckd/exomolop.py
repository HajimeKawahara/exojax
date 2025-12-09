"""Loading a petitRADTRANS correlated-k opacity HDF5 file.

This code is based on load_hdf5_ktables in petitRADTRANS under the MIT license.
"""

from __future__ import annotations

import argparse
from cProfile import label
from pathlib import Path

import h5py
import numpy as np
from exojax.utils.constants import ccgs
from exojax.utils.constants import m_u_2018  # CODATA 2018 values were used in pRT


def _reshape_kcoeff(kcoeff: np.ndarray, n_pressures: int, n_temperatures: int) -> np.ndarray:
    """Apply the reshaping convention used in Radtrans.load_hdf5_ktables."""
    # Swap the pressure/temperature axes, flatten them into a single axis,
    # move the g-ordinate axis first, and flip the wavelength direction.
    k_table = np.swapaxes(kcoeff, 0, 1)
    k_table = k_table.reshape((n_pressures * n_temperatures, k_table.shape[2], k_table.shape[3]))
    k_table = np.swapaxes(k_table, 0, 2)
    k_table = k_table[:, ::-1, :]
    # Final shape: (g, freq, T*P)
    return k_table


def load_ktable(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""
    with h5py.File(path, "r") as fh5:
        molecule = fh5["mol_name"][()][0].decode("utf-8")
        mol_mass = float(fh5["mol_mass"][()][0])
        bin_centers = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]    # K
        pressures = fh5["p"][:]       # bar
        kcoeff = np.array(fh5["kcoeff"])

    wavenumber = bin_centers[::-1]  # cm-1 (reversed order)

    n_t = temperatures.size
    n_p = pressures.size
    g_size = samples.size
    n_freq = wavenumber.size

    # k_table shape: (g, freq, T*P)
    k_table = _reshape_kcoeff(
        kcoeff=kcoeff,
        n_pressures=n_p,
        n_temperatures=n_t,
    )

    # === Convert to (T, P, g, wavenumber) ===
    # Current shape: (g, wavenumber, T*P)
    # First reshape (T*P -> P, T), then reorder axes.
    xsgrid = k_table.reshape(g_size, n_freq, n_p, n_t)      # -> (g, wavenumber, P, T)
    xsgrid = np.transpose(xsgrid, (3, 2, 0, 1))       # -> (T, P, g, wavenumber)

    # Clip negative values
    xsgrid[xsgrid < 0.0] = 0.0
    #opacity_grid *= 1.0 / (mol_mass * m_u_2018) # convert to cm2 per gram

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
    plt.xlim(22000,24000)
    plt.tight_layout()
    plt.show()

