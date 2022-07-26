import pytest
import numpy as np
from exojax.spec.api import MdbExomol as MdbExomol_api
from exojax.spec.moldb import MdbExomol as MdbExomol_orig

def comparison_moldb_exomol():
    crit=1.e-30
    mapi = MdbExomol_api(".database/CO/12C-16O/Li2015",nurange=[4200.0,4300.0],crit=crit,margin=0.0)
    morig = MdbExomol_orig(".database/CO/12C-16O/Li2015",nurange=[4200.0,4300.0],crit=crit,margin=0.0)
    assert np.all(mapi.A - morig._A) == 0.0
    
    T=1000.0
    qr_api = mapi.qr_interp(T)
    qr_orig = morig.qr_interp(T)
    assert qr_api - qr_orig == pytest.approx(0.0)
    
if __name__ == "__main__":
    comparison_moldb_exomol()