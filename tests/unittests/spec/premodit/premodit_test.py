import pytest
import numpy as np
from exojax.spec.premodit import compute_dElower, make_elower_grid, make_broadpar_grid

def test_compute_dElower():
    assert compute_dElower(1000.0,interval_contrast=0.1)==pytest.approx(160.03762408883165)

def test_make_elower_grid():
    maxe=12001.0
    mine=99.01
    eg=make_elower_grid(1000, [mine,maxe], 1.0)
    assert eg[-1]>=maxe and eg[0]<=mine

def test_make_broadpar_grid():
    ngamma_ref=np.array([0.1,0.11,0.15])
    n_Texp=np.array([0.5,0.4,0.47])
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
    
if __name__=="__main__":
    test_make_broadpar_grid()
