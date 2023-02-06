from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_XS_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_XS_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_XS_REF_AIR

import numpy as np
from exojax.spec.modit import xsvector
from exojax.spec.hitran import line_strength
from exojax.spec.molinfo import molmass_isotope
from exojax.spec import normalized_doppler_sigma, gamma_natural
from exojax.spec.hitran import line_strength
from exojax.spec.hitran import gamma_hitran
from exojax.spec.exomol import gamma_exomol
from exojax.utils.grids import wavenumber_grid
from exojax.spec.initspec import init_modit
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp


def gendata_xs_modit_exomol():
    from jax.config import config
    config.update("jax_enable_x64", True)

    mdbCO = mock_mdbExomol()
    Tfix = 1200.0
    Pfix = 1.0
    Mmol = molmass_isotope("CO")
    Nx = 5000
    nus, wav, res = wavenumber_grid(22800.0,
                                    23100.0,
                                    Nx,
                                    unit='AA',
                                    xsmode="modit")
    cont_nu, index_nu, R, pmarray = init_modit(mdbCO.nu_lines, nus)
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp,
                          mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    dv_lines = mdbCO.nu_lines / R
    ngammaL = gammaL / dv_lines
    nsigmaD = normalized_doppler_sigma(Tfix, Mmol, R)
    Sij = line_strength(Tfix, mdbCO.logsij0, mdbCO.nu_lines, mdbCO.elower, qt)

    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    xsv = xsvector(cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij, nus,
                   ngammaL_grid)
    np.savetxt(TESTDATA_CO_EXOMOL_MODIT_XS_REF,
               np.array([nus, xsv]).T,
               delimiter=",")


def gendata_xs_modit_hitemp(airmode=False):
    """generate cross section sample for HITEMP

    Args:
        airmode (bool, optional): If True, Pself=0.0 applied. Defaults to False.
    """
    from jax.config import config
    config.update("jax_enable_x64", True)
    from exojax.spec import api

    Tfix = 1200.0
    Pfix = 1.0
    if airmode:
        Pself = 0.0
        filename = TESTDATA_CO_HITEMP_MODIT_XS_REF_AIR
    else:
        Pself = Pfix
        filename = TESTDATA_CO_HITEMP_MODIT_XS_REF
    Nx = 5000
    nus, wav, res = wavenumber_grid(22800.0,
                                    23100.0,
                                    Nx,
                                    unit='AA',
                                    xsmode="modit")


    #### HERE IS Temporary
    mdbCO = api.MdbHitemp('CO', nus, gpu_transfer=True, isotope=1)
    #mdbCO = mock_mdbHitemp(multi_isotope=False)


    Mmol = mdbCO.molmass
    
    cont_nu, index_nu, R, pmarray = init_modit(mdbCO.nu_lines, nus)
    qt = mdbCO.qr_interp(mdbCO.isotope, Tfix)
    gammaL = gamma_hitran(Pfix, Tfix, Pself, mdbCO.n_air, mdbCO.gamma_air,
                          mdbCO.gamma_self) + gamma_natural(mdbCO.A)
    dv_lines = mdbCO.nu_lines / R
    ngammaL = gammaL / dv_lines
    nsigmaD = normalized_doppler_sigma(Tfix, Mmol, R)
    Sij = line_strength(Tfix, mdbCO.logsij0, mdbCO.nu_lines, mdbCO.elower, qt)
    cont_nu, index_nu, R, pmarray = init_modit(mdbCO.nu_lines, nus)
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.1)
    xsv = xsvector(cont_nu, index_nu, R, pmarray, nsigmaD, ngammaL, Sij,
                         nus, ngammaL_grid)

    np.savetxt(filename, np.array([nus, xsv]).T, delimiter=",")


if __name__ == "__main__":
    gendata_xs_modit_exomol()
    gendata_xs_modit_hitemp(airmode=False)
    gendata_xs_modit_hitemp(airmode=True)

    print(
        "to include the generated files in the package, move .txt to exojax/src/exojax/data/testdata/"
    )
