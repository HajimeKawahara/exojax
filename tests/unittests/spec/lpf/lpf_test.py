import pytest
from exojax.spec.lpf import xsvector
import pandas as pd
import numpy as np
import pickle
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL, TESTDATA_CO_EXOMOL_LPF_SPECTRUM_REF
from exojax.spec.lpf import exomol
from exojax.spec.molinfo import molmass
from exojax.spec import SijT, doppler_sigma,  gamma_natural
from exojax.spec.exomol import gamma_exomol
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.initspec import init_lpf

import pkg_resources

def test_exomol():
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_CO_EXOMOL)
    with open(filename, 'rb') as f:
        mdbCO = pickle.load(f)
        
    Tfix=1200.0
    Pfix=1.0
    Mmol = molmass("CO")
    
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp, mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    
    Nx=1000
    nus, wav, res = gen_wavenumber_grid(22940.0,22960.0, Nx, unit='AA')
    numatrix=init_lpf(mdbCO.nu_lines, nus)
    xsv=xsvector(numatrix, sigmaD, gammaL, Sij)

    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_CO_EXOMOL_LPF_SPECTRUM_REF)
    dat=pd.read_csv(filename,delimiter=",",names=("nus","xsv"))
    assert np.all(xsv == pytest.approx(dat["xsv"].values))
    
#    exomol()
if __name__ == "__main__":
    test_exomol()