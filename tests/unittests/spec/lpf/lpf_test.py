import pytest
from exojax.spec.lpf import xsection

import pickle
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL
from exojax.spec.lpf import exomol
from exojax.utils.molinfo import molmass
from exojax.spec import SijT, doppler_sigma,  gamma_natural
from exojax.spec.exomol import gamma_exomol
import pkg_resources

def test_exomol():
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_CO_EXOMOL)
    with open(filename, 'rb') as f:
        mdbCO = pickle.load(f)
        
    Tfix=1200.0
    Mmol = molmass("CO")
    qt = mdbCO.qr_interp(Tfix)
    gammaL = gamma_exomol(Pfix, Tfix, mdbCO.n_Texp, mdbCO.alpha_ref) + gamma_natural(mdbCO.A)
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    
#    exomol()
if __name__ == "__main__":
    test_exomol()
