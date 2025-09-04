import numpy as np

from exojax.test.emulate_mdb import mock_mdbExomol


def test_exomol_to_snapshot_smoke():
    # Use bundled sample ExoMol data via helper
    mdb = mock_mdbExomol("CO")

    snap = mdb.to_snapshot()

    # Meta basics
    assert snap.meta.dbtype == "exomol"
    assert isinstance(snap.meta.T_gQT, np.ndarray)
    assert isinstance(snap.meta.gQT, np.ndarray)
    assert snap.meta.T_gQT.ndim == 1 and snap.meta.T_gQT.size > 0

    # Lines payload mirrors current selection
    assert isinstance(snap.lines.nu_lines, np.ndarray)
    assert isinstance(snap.lines.elower, np.ndarray)
    assert isinstance(snap.lines.line_strength_ref_original, np.ndarray)
    assert snap.lines.nu_lines.shape == mdb.nu_lines.shape
    assert snap.lines.elower.shape == mdb.elower.shape
    assert snap.lines.line_strength_ref_original.shape == mdb.line_strength_ref_original.shape

    # ExoMol-specific broadening payload should be present after activation
    assert snap.n_Texp is None or snap.n_Texp.shape == mdb.n_Texp.shape
    assert snap.alpha_ref is None or snap.alpha_ref.shape == mdb.alpha_ref.shape

