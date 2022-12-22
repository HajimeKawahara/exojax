import pytest
from exojax.spec.hitran import line_strength
from exojax.spec.lpf import xsvector

import pkg_resources
import pandas as pd
import numpy as np
from exojax.test.data import TESTDATA_CO_EXOMOL_LPF_XS_REF
#from exojax.spec.lpf import exomol
from exojax.spec.molinfo import molmass_major_isotope
from exojax.spec import doppler_sigma,  gamma_natural
from exojax.spec.hitran import line_strength
from exojax.spec.exomol import gamma_exomol
from exojax.utils.grids import wavenumber_grid
from exojax.spec.initspec import init_lpf
from exojax.test.emulate_mdb import mock_mdbExomol

def test_exomol():
    mdbCO = mock_mdbExomol()    
    Tfix=1200.0
    Pfix=1.0
    Mmol = molmass_major_isotope("CO")
    
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp, mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    Sij = line_strength(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    
    Nx=1000
    nus, wav, res = wavenumber_grid(22940.0,22960.0, Nx, unit='AA')
    numatrix=init_lpf(mdbCO.nu_lines, nus)
    xsv=xsvector(numatrix, sigmaD, gammaL, Sij)

    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_CO_EXOMOL_LPF_XS_REF)
    dat=pd.read_csv(filename,delimiter=",",names=("nus","xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))
    
if __name__ == "__main__":
    test_exomol()
