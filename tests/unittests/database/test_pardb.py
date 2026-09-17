"""unit tests for pardb
"""

from exojax.test.emulate_pdb import mock_PdbPlouds
import numpy as np
import pytest


def test_pdb_clouds_nurange_initialize():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()

    assert pdb.refraction_index_wavenumber[0] == pytest.approx(11990.407673860911)
    assert np.all(
        pdb.refraction_index_wavenumber
        == pytest.approx(1.0e7 / pdb.refraction_index_wavelength_nm)
    )


def test_pdb_clouds_nurange_redefine():
    pdb = mock_PdbPlouds()
    pdb.load_miegrid()
    pdb.nurange = [12000.0, 15000.0]

    pdb.reset_miegrid_for_nurange()

    assert pdb.refraction_index_wavenumber[0] == pytest.approx(11990.407673860911)
    assert np.all(
        pdb.refraction_index_wavenumber
        == pytest.approx(1.0e7 / pdb.refraction_index_wavelength_nm)
    )


def test_pdb_clouds_interp():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    sigmag = 2.0
    rg = 1.0e-5
    dtau_g, w_g, g_g = pdb.mieparams_cgs_at_refraction_index_wavenumber_from_miegrid(
        rg, sigmag
    )

    expected_shape = pdb.refraction_index_wavenumber.shape
    for values in (dtau_g, w_g, g_g):
        assert values.shape == expected_shape
        assert np.all(np.isfinite(values))
