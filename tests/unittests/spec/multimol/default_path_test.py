from exojax.database.multimol  import database_path_hitran12

def test_database_path_hitran12():
    _hitran_dbpath = {
        "H2O": "H2O/01_hit12.par",
        "CH4": "CH4/06_hit12.par",
        "CO": "CO/05_hit12.par",
        "NH3": "NH3/11_hit12.par",
        "H2S": "H2S/31_hit12.par"
    }
    for mol in _hitran_dbpath:
        assert database_path_hitran12(mol) == _hitran_dbpath[mol]

if __name__ == "__main__":
    test_database_path_hitran12()