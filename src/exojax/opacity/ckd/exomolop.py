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
    ckd_info = load_exomolop_ckd(args.h5_file)

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
    temperature = ckd_info.T_grid[iT]
    pressure = ckd_info.P_grid[jP]
    print(temperature, "K", pressure,"bar")
    
    xsv = opa.xsvector(temperature, pressure)
    
    ktable = jnp.exp(ckd_info.log_kggrid)
    nu_ckd = ckd_info.nu_bands
    wav_ckd = 1e4/nu_ckd  # micron
    print(np.min(nu_ckd), np.max(nu_ckd))
    print(np.min(wav_ckd), np.max(wav_ckd))
    xs_ckd = np.sum(ktable[iT, jP, :, :] * ckd_info.weights[:, np.newaxis], axis=0)
    
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

