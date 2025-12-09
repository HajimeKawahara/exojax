"""Loading a petitRADTRANS correlated-k opacity HDF5 file.

This code is based on load_hdf5_ktables in petitRADTRANS under the MIT license.
"""

from __future__ import annotations

import argparse
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
    """Load a correlated-k opacity file and return metadata and the opacity grid."""
    with h5py.File(path, "r") as fh5:
        molecule = fh5["mol_name"][()][0].decode("utf-8")
        mol_mass = float(fh5["mol_mass"][()][0])
        bin_centers = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]    # K
        pressures = fh5["p"][:]       # bar
        kcoeff = np.array(fh5["kcoeff"])

    frequencies = ccgs * bin_centers[::-1]  # Hz (reversed order)

    n_t = temperatures.size
    n_p = pressures.size
    g_size = samples.size
    n_freq = frequencies.size

    # k_table shape: (g, freq, T*P)
    k_table = _reshape_kcoeff(
        kcoeff=kcoeff,
        n_pressures=n_p,
        n_temperatures=n_t,
    )

    # === Convert to (T, P, g, freq) ===
    # Current shape: (g, freq, T*P)
    # First reshape (T*P -> P, T), then reorder axes.
    opacity_grid = k_table.reshape(g_size, n_freq, n_p, n_t)      # -> (g, freq, P, T)
    opacity_grid = np.transpose(opacity_grid, (3, 2, 0, 1))       # -> (T, P, g, freq)

    # Clip negative values and convert cross-sections to mass opacities.
    opacity_grid[opacity_grid < 0.0] = 0.0
    opacity_grid *= 1.0 / (mol_mass * m_u_2018)

    return {
        "molecule": molecule,
        "molecular_mass_amu": mol_mass,
        "temperatures_K": temperatures,
        "pressures_bar": pressures,
        "g_samples": samples,
        "g_weights": weights,
        "frequencies_Hz": frequencies,
        "opacity_grid": opacity_grid,  # shape: (T, P, g, freq)
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
    print(f"Opacity grid shape       : {info['opacity_grid'].shape} (T, P, g, freq)")

    # Quick sanity check
    sample_value = info["opacity_grid"][0, 0, 0, 0]  # (T=0, P=0, g=0, freq=0)
    print(f"First opacity value      : {sample_value:.3e} cm^2 g^-1")


if __name__ == "__main__":
    main()
