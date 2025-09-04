import numpy as np

from exojax.opacity.premodit.api import OpaPremodit
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot


def make_fake_snapshot_exomol():
    meta = MDBMeta(
        dbtype="exomol",
        molmass=18.0,
        T_gQT=np.array([300.0, 1000.0, 2000.0]),
        gQT=np.array([1.0, 2.0, 4.0]),
    )
    lines = Lines(
        nu_lines=np.array([1000.0, 1005.0, 1010.0]),
        elower=np.array([10.0, 20.0, 30.0]),
        line_strength_ref_original=np.array([1e-30, 2e-30, 3e-30]),
    )
    return MDBSnapshot(
        meta=meta,
        lines=lines,
        n_Texp=np.array([0.5, 0.5, 0.5]),
        alpha_ref=np.array([0.1, 0.1, 0.1]),
    )


def test_from_snapshot_minimal_construction():
    snap = make_fake_snapshot_exomol()
    nu_grid = np.linspace(990.0, 1020.0, 32)

    opa = OpaPremodit.from_snapshot(snap, nu_grid, allow_32bit=True)

    assert opa.method == "premodit"
    assert opa.dbtype == "exomol"
    assert opa.molmass == snap.meta.molmass
    np.testing.assert_allclose(opa.nu_grid, nu_grid)
    # ensure values flowed through
    np.testing.assert_allclose(opa.T_gQT, snap.meta.T_gQT)
    np.testing.assert_allclose(opa.gQT, snap.meta.gQT)
    np.testing.assert_allclose(opa.nu_lines, snap.lines.nu_lines)
    np.testing.assert_allclose(opa.elower, snap.lines.elower)


def test_from_mdb_calls_to_snapshot():
    class FakeMdb:
        def __init__(self, snap):
            self._snap = snap

        def to_snapshot(self):
            return self._snap

    snap = make_fake_snapshot_exomol()
    mdb = FakeMdb(snap)
    nu_grid = np.linspace(990.0, 1020.0, 16)

    opa = OpaPremodit.from_mdb(mdb, nu_grid, allow_32bit=True)
    assert opa.dbtype == "exomol"
