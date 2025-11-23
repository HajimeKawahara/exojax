import pytest
from exojax.database.multimol  import MultiMol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.database.exomol.api import MdbExomol


def test_multimdb_single_nu_grid():
    mul = MultiMol(molmulti=[["CO", "H2O"]], dbmulti=[["SAMPLE", "SAMPLE"]])
    nu_grid, wav, res = mock_wavenumber_grid()
    nu_grid_list = [nu_grid]

    multimdb = mul.multimdb(nu_grid_list)

    assert type(multimdb[0][0]) == MdbExomol
    assert type(multimdb[0][1]) == MdbExomol


def test_multimol_different_structure_raise_error():
    with pytest.raises(ValueError):
        MultiMol(molmulti=[["CO", "H2O"], ["H2O"]], dbmulti=[["SAMPLE", "SAMPLE"]])


def _check_structure(a, b):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_check_structure(sub_a, sub_b) for sub_a, sub_b in zip(a, b))
    return not isinstance(a, list) and not isinstance(b, list)


if __name__ == "__main__":
    # test_multimdb_single_nu_grid()
    # test_multiopa_single_nu_grid()
    # test_multiopa_multi_nu_grid()
    test_multimol_different_structure_raise_error()
