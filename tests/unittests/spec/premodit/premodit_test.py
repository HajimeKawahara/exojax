import pytest
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL
import numpy as np
from exojax.spec.premodit import compute_dElower, make_elower_grid, make_broadpar_grid
from exojax.test.emulate_broadpar import mock_broadpar_exomol

def test_compute_dElower():
    assert compute_dElower(1000.0,interval_contrast=0.1)==pytest.approx(160.03762408883165)

def test_make_elower_grid():
    maxe=12001.0
    mine=99.01
    eg=make_elower_grid(1000, [mine,maxe], 1.0)
    assert eg[-1]>=maxe and eg[0]<=mine

def test_make_broadpar_grid():
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    Ttyp=3000.0
    Tref=296.0
    bp=make_broadpar_grid(ngamma_ref, n_Texp, Ttyp, dit_grid_resolution=0.2, adopt=True)
    ref=np.array([[0.1       ,0.4       ],
                  [0.1       ,0.45      ],
                  [0.1       ,0.5       ],
                  [0.11447142,0.4       ],
                  [0.11447142,0.45      ],
                  [0.11447142,0.5       ],
                  [0.13103707,0.4       ],
                  [0.13103707,0.45      ],
                  [0.13103707,0.5       ],
                  [0.15      ,0.4       ],
                  [0.15      ,0.45      ],
                  [0.15      ,0.5       ]])
    assert np.all(bp==pytest.approx(ref))

def test_unbiased_lsd():
    from exojax.spec.lsd import npadd1D, npgetix
    from exojax.spec.hitran import line_strength
    from exojax.spec.premodit import make_elower_grid, unbiased_lsd
    interval_contrast=0.1
    Ttyp=2000.0
    Tref=296.0
    ngamma_ref, n_Texp = mock_broadpar_exomol()
    broadpar_grid=make_broadpar_grid(ngamma_ref, n_Texp, Ttyp, dit_grid_resolution=0.2, adopt=True)    

    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    
#    cont_nu, index_nu = npgetix(mdb.nu_lines, nus)
#    #lbd=make_lbd3D_uniqidx(mdb.Sij0, cont_nu, index_nu, len(nus), mdb._elower, elower_grid, uidx_broadpar, Ttyp)
#    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdb.qr_interp)


if __name__=="__main__":
    test_make_broadpar_grid()
