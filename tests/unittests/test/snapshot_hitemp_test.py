import numpy as np

from exojax.test.emulate_mdb import mock_mdbHitemp


def test_hitemp_to_snapshot_smoke():
    mdb = mock_mdbHitemp()

    snap = mdb.to_snapshot()

    # Meta
    assert snap.meta.dbtype == "hitran"
    assert isinstance(snap.meta.T_gQT, np.ndarray)
    assert isinstance(snap.meta.gQT, np.ndarray)
    assert snap.meta.T_gQT.size > 0 and snap.meta.gQT.size > 0

    # Lines payload mirrors current selection
    assert snap.lines.nu_lines.shape == mdb.nu_lines.shape
    assert snap.lines.elower.shape == mdb.elower.shape
    assert (
        snap.lines.line_strength_ref_original.shape
        == mdb.line_strength_ref_original.shape
    )

    # HITRAN/HITEMP-specific fields exist and match shapes
    assert snap.isotope is not None and snap.isotope.shape == mdb.isoid.shape
    assert snap.uniqiso is not None and snap.uniqiso.shape == mdb.uniqiso.shape
    assert snap.n_air is not None and snap.n_air.shape == mdb.n_air.shape
    assert snap.gamma_air is not None and snap.gamma_air.shape == mdb.gamma_air.shape

