import pytest
from exojax.spec.multimol import MultiMol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit


def test_multimdb_single_nu_grid():
    mul = MultiMol(molmulti=[["CO", "H2O"]], dbmulti=[["SAMPLE", "SAMPLE"]])
    nu_grid, wav, res = mock_wavenumber_grid()
    nu_grid_list = [nu_grid]

    multimdb = mul.multimdb(nu_grid_list)

    assert type(multimdb[0][0]) == MdbExomol
    assert type(multimdb[0][1]) == MdbExomol


def test_multiopa_single_nu_grid():
    mul = MultiMol(molmulti=[["CO", "H2O"]], dbmulti=[["SAMPLE", "SAMPLE"]])
    nu_grid, wav, res = mock_wavenumber_grid()
    nu_grid_list = [nu_grid]
    multimdb = mul.multimdb(nu_grid_list)

    multiopa = mul.multiopa_premodit(
        multimdb,
        nu_grid_list,
        auto_trange=[500.0, 1500.0],
        dit_grid_resolution=0.2,
        allow_32bit=True,
    )

    assert type(multiopa[0][0]) == OpaPremodit
    assert type(multiopa[0][1]) == OpaPremodit


def test_multiopa_multi_nu_grid():
    molmulti = [["CO", "H2O"], ["H2O"]]

    mul = MultiMol(molmulti=molmulti, dbmulti=[["SAMPLE", "SAMPLE"], ["SAMPLE"]])
    nu_grid, wav, res = mock_wavenumber_grid()
    N = int(len(nu_grid) / 2)
    nu_grid_list = [nu_grid[:N], nu_grid[N:]]
    multimdb = mul.multimdb(nu_grid_list)

    multiopa = mul.multiopa_premodit(
        multimdb,
        nu_grid_list,
        auto_trange=[500.0, 1500.0],
        dit_grid_resolution=0.2,
        allow_32bit=True,
    )

    assert _check_structure(multiopa, [["CO", "H2O"], ["H2O"]])


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
