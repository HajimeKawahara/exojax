# saveopa using zarr format test
from jax import config 
config.update("jax_enable_x64", True)

import os
from exojax.database import MdbExomol
from exojax.opacity import OpaPremodit
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import saveopa
Nx = 4500
nu_grid, wav, res = wavenumber_grid(22000.0, 23000.0, Nx, unit="AA", xsmode="premodit")
Tlow = 210.0
Thigh = 3500.0
mol_paths = {"CO": ".database/CO/12C-16O/Li2015"}

def calc_or_load_opa(molecule, nu_grid, Tlow, Thigh, elower_max, mol_paths, filename, diffmode=0):
    if os.path.exists(filename):
        print("load saved opa from ", filename)
        opa = OpaPremodit.from_saved_opa(filename)
        molmass_ = opa.aux["molmass"]
    else:
        print("calculate opa and save to ", filename)
        mdbCO = MdbExomol(mol_paths[molecule], nurange=nu_grid, gpu_transfer=False, elower_max=elower_max)
        molmass_ = mdbCO.molmass # we use molmass later
        snap = mdbCO.to_snapshot() # extract snapshot from mdb
        del mdbCO # save the memory
        opa = OpaPremodit.from_snapshot(snap, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[Tlow, Thigh], dit_grid_resolution=1.0)
        saveopa(opa, filename, format="zarr", aux={"molmass": molmass_})
    return opa, molmass_

opaCO, molmassCO = calc_or_load_opa("CO", nu_grid, Tlow, Thigh, 58242.689, mol_paths, "opaCO.zarr", diffmode=0)