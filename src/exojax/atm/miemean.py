"""takes an average of mean assuming a particle size distribution
"""
from PyMieScatt import MieQ
from PyMieScatt import Mie_Lognormal 



def compute_each_mie_coeff_lognormal(wavenm, refractive_index_array, sigmag, rg_array):
    from tqdm import tqdm
    from PyMieScatt import Mie_Lognormal as mief
    #### needs to modify and check inverse?
    #wavenm = self.refraction_index_wavelength_nm[-30:-8]
    #marr = self.refraction_index[-30:-8]
    #####
    npart = 1.0e6
    cm2nm = 1.0e7
    mie = []
    for rgl in tqdm(rg_array * cm2nm):
        mie_each = []
        for i, m in enumerate(tqdm(refractive_index_array)):
            coeff = mief(m, wavenm[i], sigmag, rgl, npart)
            mie_each.append(coeff)
        mie.append(mie_each)
    print("Not complete yet")
    return mie


if __name__ == "__main__":
    import numpy as np
    import time
    from exojax.spec.pardb import PdbCloud
    pdb_nh3 = PdbCloud("NH3")
    wavenm = pdb_nh3.refraction_index_wavelength_nm
    marray = pdb_nh3.refraction_index

    sigmag = 2.0
    N=10
    rg_array = np.logspace(-7,-4,N)
    ts = time.time()
    compute_each_mie_coeff_lognormal(wavenm, marray, sigmag, rg_array)
    te = time.time()
    print(te-ts,"sec")