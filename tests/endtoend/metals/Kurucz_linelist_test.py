"""test for opacity calculation using the Kurucz linelist.

- This test calculates Fe I opacity from Kurucz linelist (http://kurucz.harvard.edu/linelists/)
  The calculation of gamma is based on the van der Waals gamma in the line list

Note: The input data "gf2600.all" will be downloaded from [Index of /linelists/gfall](http://kurucz.harvard.edu/linelists/gfall/).
"""

import os

filepath_Kurucz = ".database/gf2600.all"
if not os.path.isfile(filepath_Kurucz):
    import urllib.request

    try:
        url = "http://kurucz.harvard.edu/linelists/gfall/gf2600.all"
        urllib.request.urlretrieve(url, filepath_Kurucz)
    except:
        print("could not connect ", url)


def test_Kurucz_linelist():
    from exojax.utils.grids import wavenumber_grid
    from exojax.database import moldb 
    from exojax.database import atomll 
    from exojax.rt import ArtEmisPure
    from exojax.opacity import OpaDirect
    import numpy as np
    from exojax.opacity.lpf.lpf import xsmatrix

    wls, wll = 10350, 10450
    wavenumber_grid_res = 0.01
    nus, wav, res = wavenumber_grid(
        wls, wll, int((wll - wls) / wavenumber_grid_res), unit="AA", xsmode="lpf"
    )

    NP = 100
    T0 = 3000.0
    alpha = 0.1
    art = ArtEmisPure(nu_grid=nus, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=NP)
    Parr = art.pressure
    Tarr = art.powerlaw_temperature(T0, alpha)

    H_He_HH_VMR = [0.0, 0.16, 0.84]  # typical quasi-"solar-fraction"
    mmw = 2.33  # mean molecular weight

    adbK = moldb.AdbKurucz(filepath_Kurucz, nus, vmr_fraction=H_He_HH_VMR)

    Rp = 0.36 * 10  # R_sun*10
    Mp = 0.37 * 1e3  # M_sun*1e3
    g = 2478.57730044555 * Mp / Rp**2
    print("logg: " + str(np.log10(g)))
    VMR_Fe = atomll.get_VMR_uspecies(np.array([[26, 1]]))

#    See Issue #539
#    opa = OpaDirect(mdb=adbK, nu_grid=nus)
#    xsmatrix = opa.xsmatrix(Tarr, Parr)
#    mmr_arr = art.constant_mmr_profile(VMR_Fe)
#    dtaua_K = art.opacity_profile_xs(xsmatrix, mmr_arr, mmw, g)
#    assert np.isclose(np.sum(dtaua_K), 6644.0303)


if __name__ == "__main__":
    test_Kurucz_linelist()
