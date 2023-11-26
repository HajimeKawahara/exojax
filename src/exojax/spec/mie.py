""" Mie scattering calculation using PyMieScatt


"""

import numpy as np


def compute_mie_coeff_lognormal_grid(
    refractive_indices, refractive_wavenm, sigmag_arr, rg_arr, npart=1.0e6
):
    from tqdm import tqdm
    from PyMieScatt import Mie_Lognormal as mief

    cm2nm = 1.0e7
    Nwav = len(refractive_indices)
    Nsigmag = len(sigmag_arr)
    Nrg = len(rg_arr)
    Nmiecoeff = 7
    miegrid = np.zeros((Nrg, Nsigmag, Nwav, Nmiecoeff), dtype=np.complex128)

    for ind_sigmag, sigmag in enumerate(tqdm(sigmag_arr)):
        for ind_rg, rg in enumerate(tqdm(np.array(rg_arr) * cm2nm)):
            for ind_m, m in enumerate(tqdm(refractive_indices)):
                coeff = mief(m, refractive_wavenm[ind_m], sigmag, rg, npart)
                miegrid[ind_rg, ind_sigmag, ind_m, :] = coeff

    return miegrid


if __name__ == "__main__":
    from exojax.spec.pardb import PdbCloud

    sigmag_arr = [1.0, 3.0, 10.0]
    rg_arr = [1.0e-6, 1.0e-5, 1.0e-4]
    pdb_nh3 = PdbCloud("NH3")
    miegrid = compute_mie_coeff_lognormal_grid(
        pdb_nh3.refraction_index,
        pdb_nh3.refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        npart=1.0,
    )
    np.savez("miegrid_sample.mgd", miegrid)
