"""test for opacity calculation using the Kurucz linelist.

- This test calculates Fe I opacity from Kurucz linelist (http://kurucz.harvard.edu/linelists/)
  The calculation of gamma is based on the van der Waals gamma in the line list

Note: The input data "gf2600.all" will be downloaded from [Index of /linelists/gfall](http://kurucz.harvard.edu/linelists/gfall/).
"""

import os

filepath_Kurucz = '.database/gf2600.all'
if not os.path.isfile(filepath_Kurucz):
    import urllib.request
    try:
        url = "http://kurucz.harvard.edu/linelists/gfall/gf2600.all"
        urllib.request.urlretrieve(url, filepath_Kurucz)
    except:
        print('could not connect ', url_developer_data())


def test_Kurucz_linelist():
    from exojax.utils.grids import wavenumber_grid
    from exojax.spec.rtransfer import pressure_layer
    from exojax.spec import moldb
    from exojax.spec import atomll
    from exojax.spec import line_strength, doppler_sigma
    import jax.numpy as jnp
    from jax import vmap, jit
    import numpy as np
    from exojax.spec.initspec import init_lpf
    from exojax.spec.lpf import xsmatrix
    from exojax.spec.rtransfer import dtauM

    NP = 100
    T0 = 3000.
    Parr, dParr, k = pressure_layer(NP=NP)
    Tarr = T0 * (Parr)**0.1

    H_He_HH_VMR = [0.0, 0.16, 0.84]  #typical quasi-"solar-fraction"
    mmw = 2.33  #mean molecular weight

    PH = Parr * H_He_HH_VMR[0]
    PHe = Parr * H_He_HH_VMR[1]
    PHH = Parr * H_He_HH_VMR[2]

    wls, wll = 10350, 10450
    wavenumber_grid_res = 0.01
    nus, wav, reso = wavenumber_grid(wls,
                                     wll,
                                     int((wll - wls) / wavenumber_grid_res),
                                     unit="AA",
                                     xsmode="lpf")
    adbK = moldb.AdbKurucz(filepath_Kurucz, nus)
    qt_284 = vmap(adbK.QT_interp_284)(Tarr)

    qt_K = np.zeros([len(adbK.QTmask), len(Tarr)])
    for i, mask in enumerate(adbK.QTmask):
        qt_K[i] = qt_284[:, mask]
    qt_K = jnp.array(qt_K)

    gammaLM_K = jit(vmap(atomll.gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
        (Tarr, PH, PHH, PHe, adbK.ielem, adbK.iion, \
                adbK.dev_nu_lines, adbK.elower, adbK.eupper, adbK.atomicmass, adbK.ionE, \
                adbK.gamRad, adbK.gamSta, adbK.vdWdamp, 1.0)

    sigmaDM_K = jit(vmap(doppler_sigma,(None,0,None)))\
        (adbK.nu_lines, Tarr, adbK.atomicmass)

    SijM_K = jit(vmap(line_strength,(0,None,None,None,0)))\
        (Tarr, adbK.logsij0, adbK.nu_lines, adbK.elower, qt_K.T)

    numatrix_K = init_lpf(adbK.nu_lines, nus)

    Rp = 0.36 * 10  #R_sun*10
    Mp = 0.37 * 1e3  #M_sun*1e3
    g = 2478.57730044555 * Mp / Rp**2
    print('logg: ' + str(np.log10(g)))

    VMR_Fe = atomll.get_VMR_uspecies(np.array([[26, 1]]))
    xsm_K = xsmatrix(numatrix_K, sigmaDM_K, gammaLM_K, SijM_K)
    dtaua_K = dtauM(dParr, xsm_K, VMR_Fe * np.ones_like(Tarr), mmw, g)

    assert np.isclose(np.sum(dtaua_K), 12558.645)


if __name__ == '__main__':
    test_Kurucz_linelist()
