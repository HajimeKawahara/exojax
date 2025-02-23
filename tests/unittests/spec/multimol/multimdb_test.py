from exojax.spec.multimol import MultiMol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit

def test_multimdb_no_nustitch():
    mul = MultiMol(molmulti=[["CO","H2O"]], dbmulti=[["SAMPLE","SAMPLE"]])
    nu_grid,wav,res = mock_wavenumber_grid()
    nu_grid_list = [nu_grid]
    
    multimdb = mul.multimdb(nu_grid_list)
    
    assert type(multimdb[0][0]) == MdbExomol
    assert type(multimdb[0][1]) == MdbExomol

def test_multiopa_no_nustitch():
    mul = MultiMol(molmulti=[["CO","H2O"]], dbmulti=[["SAMPLE","SAMPLE"]])
    nu_grid,wav,res = mock_wavenumber_grid()
    nu_grid_list = [nu_grid]
    multimdb = mul.multimdb(nu_grid_list)
    
    multiopa = mul.multiopa_premodit(multimdb, nu_grid_list, auto_trange=[500.,1500.], dit_grid_resolution=0.2, allow_32bit=True)
    
    assert type(multiopa[0][0]) == OpaPremodit
    assert type(multiopa[0][1]) == OpaPremodit
    

if __name__ == "__main__":
    test_multimdb_no_nustitch()
    test_multiopa_no_nustitch()