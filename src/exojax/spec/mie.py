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


def read_miegrid(filename):
    miegrid = np.load(filename)["arr_0"]
    print(np.shape(miegrid))
    return miegrid


from exojax.utils.interp import interp2d_bilinear


def evaluate_miegrid(rg, sigmag):
    interp2d_bilinear()


if __name__ == "__main__":
    from exojax.spec.pardb import PdbCloud

    pdb = PdbCloud("NH3")
    filename = "miegrid_lognorm_" + pdb.condensate + ".mgd"

    # read_miegrid(filename+".npz")
    # exit()

    # pdb = PdbCloud("NH3")
    # filename = "miegrid_lognorm_"+pdb.condensate+".mgd"
    print(filename)

    Nsigmag = 10
    sigmag_arr = np.logspace(-1, 1, Nsigmag)
    Nrg = 40
    rg_arr = np.logspace(-7, -3, Nsigmag)  # cm

    miegrid = compute_mie_coeff_lognormal_grid(
        pdb.refraction_index,
        pdb.refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        npart=1.0,
    )
    np.savez(filename, miegrid, rg_arr, sigmag_arr)
