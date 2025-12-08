"""Minimal example to inspect a petitRADTRANS correlated-k opacity HDF5 file.

Run from the repository root:
    python load_h5_example.py 12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from petitRADTRANS import physical_constants as cst


def _reshape_kcoeff(kcoeff: np.ndarray, n_pressures: int, n_temperatures: int) -> np.ndarray:
    """Apply the same reshaping convention as Radtrans.load_hdf5_ktables."""
    # Swap pressure/temperature axes, collapse them, move g-ordinate axis first, and flip wavelength order.
    k_table = np.swapaxes(kcoeff, 0, 1)
    k_table = k_table.reshape((n_pressures * n_temperatures, k_table.shape[2], k_table.shape[3]))
    k_table = np.swapaxes(k_table, 0, 2)
    k_table = k_table[:, ::-1, :]
    return k_table


def load_ktable(path: Path):
    """Load a correlated-k opacity file and return basic information plus the opacity grid."""
    with h5py.File(path, "r") as fh5:
        molecule = fh5["mol_name"][()][0].decode("utf-8")
        mol_mass = float(fh5["mol_mass"][()][0])
        bin_centers = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]
        pressures = fh5["p"][:]
        kcoeff = np.array(fh5["kcoeff"])

    frequencies = cst.c * bin_centers[::-1]  # Hz, descending
    g_size = samples.size
    tp_grid_size = temperatures.size * pressures.size

    k_table = _reshape_kcoeff(
        kcoeff=kcoeff,
        n_pressures=pressures.size,
        n_temperatures=temperatures.size,
    )

    # Allocate the opacity grid and convert cross-sections to opacities (divide by mass).
    opacity_grid = np.zeros((g_size, frequencies.size, 1, tp_grid_size))
    opacity_grid[:, :, 0, :] = k_table
    opacity_grid[opacity_grid < 0.0] = 0.0
    opacity_grid *= 1.0 / (mol_mass * cst.amu)

    return {
        "molecule": molecule,
        "molecular_mass_amu": mol_mass,
        "temperatures_K": temperatures,
        "pressures_bar": pressures,
        "g_samples": samples,
        "g_weights": weights,
        "frequencies_Hz": frequencies,
        "opacity_grid": opacity_grid,
    }


def main():
    parser = argparse.ArgumentParser(description="Load a petitRADTRANS correlated-k opacity .h5 file.")
    parser.add_argument(
        "h5_file",
        type=Path,
        help="Path to a *.ktable.petitRADTRANS.h5 file (e.g., 12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5)",
    )
    args = parser.parse_args()

    info = load_ktable(args.h5_file)

    print(f"Loaded molecule          : {info['molecule']}")
    print(f"Molecular mass [amu]     : {info['molecular_mass_amu']}")
    print(f"T grid size              : {info['temperatures_K'].size}")
    print(f"P grid size              : {info['pressures_bar'].size}")
    print(f"g-ordinates (len {info['g_samples'].size}): {np.array2string(info['g_samples'], precision=3)}")
    print(f"Opacity grid shape       : {info['opacity_grid'].shape} (g, freq, 1, T*P)")

    # Show a single opacity value as a quick sanity check.
    sample_value = info["opacity_grid"][0, 0, 0, 0]
    print(f"First opacity value      : {sample_value:.3e} cm^2 g^-1")


if __name__ == "__main__":
    print(cst.c, cst.amu) #29979245800.0 1.6605390666e-24
    main()
