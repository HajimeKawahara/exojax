import pytest
import pkg_resources
import pandas as pd
import numpy as np
from exojax.spec.modit import xsvector
from exojax.spec.hitran import line_strength
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_SPECTRUM_REF
from exojax.spec.lpf import exomol
from exojax.spec.molinfo import molmass
from exojax.spec import normalized_doppler_sigma,  gamma_natural
from exojax.spec.hitran import line_strength
from exojax.spec.exomol import gamma_exomol
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.initspec import init_modit
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.test.emulate_mdb import mock_mdbExoMol


def test_exomol():
    mdbCO = mock_mdbExoMol()
    Tfix = 1200.0
    Pfix = 1.0
    Mmol = molmass("CO")
    Nx = 5000
    nus, wav, res = gen_wavenumber_grid(
        22800.0, 23100.0, Nx, unit='AA', xsmode="modit")
    cont_nu, index_nu, R, pmarray = init_modit(mdbCO.nu_lines, nus)
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp,
                          mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    dv_lines = mdbCO.nu_lines/R
    ngammaL = gammaL/dv_lines
    nsigmaD = normalized_doppler_sigma(Tfix, Mmol, R)
    Sij = line_strength(Tfix, mdbCO.logsij0, mdbCO.nu_lines, mdbCO.elower, qt)

    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution = 0.1)
    xsv = xsvector(cont_nu, index_nu, R, pmarray, nsigmaD,
                   ngammaL, Sij, nus, ngammaL_grid)
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/'+TESTDATA_CO_EXOMOL_MODIT_SPECTRUM_REF)
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "xsv"))

    assert np.all(xsv == pytest.approx(dat["xsv"].values))


if __name__ == "__main__":
    test_exomol()
