import numpy as np

from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.policies import MemoryPolicy
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot


def _fake_snapshot_exomol():
    meta = MDBMeta(
        dbtype="exomol",
        molmass=18.0,
        T_gQT=np.array([300.0, 1000.0]),
        gQT=np.array([1.0, 2.0]),
    )
    lines = Lines(
        nu_lines=np.array([1000.0, 1001.0, 1002.0]),
        elower=np.array([10.0, 20.0, 30.0]),
        line_strength_ref_original=np.array([1e-30, 2e-30, 3e-30]),
    )
    return MDBSnapshot(
        meta=meta,
        lines=lines,
        n_Texp=np.array([0.5, 0.5, 0.5]),
        alpha_ref=np.array([0.1, 0.1, 0.1]),
    )


class _FakeMdbLegacy:
    """Mimics the attribute surface that OpaPremodit.__init__ reads for exomol."""

    def __init__(self, snap: MDBSnapshot):
        self.dbtype = snap.meta.dbtype
        self.molmass = snap.meta.molmass
        self.T_gQT = snap.meta.T_gQT
        self.gQT = snap.meta.gQT
        self.nu_lines = snap.lines.nu_lines
        self.elower = snap.lines.elower
        self.line_strength_ref_original = snap.lines.line_strength_ref_original
        # exomol-specific
        self.n_Texp = snap.n_Texp
        self.alpha_ref = snap.alpha_ref


class _FakeMdbWithSnapshot(_FakeMdbLegacy):
    def __init__(self, snap: MDBSnapshot):
        super().__init__(snap)
        self._snap = snap

    def to_snapshot(self):
        return self._snap


def test_constructors_are_pairwise_equal(monkeypatch):
    # Stub heavy init to be deterministic and fast
    import exojax.opacity.initspec as initspec

    def _stub_init_premodit(*args, **kwargs):
        # lbd_coeff shape = (diffmode+1, L, N_broadening, N_elower)
        lbd_coeff = np.zeros((1, 4, 2, 3))
        mi = np.array([[0, 0], [1, 1]])
        elg = np.array([0.0, 1.0, 2.0])
        ngr = np.array([0.01, 0.02])
        ntg = np.array([0.5, 0.5])
        R = 1.0
        pm = np.array([42.0])
        return lbd_coeff, mi, elg, ngr, ntg, R, pm

    monkeypatch.setattr(initspec, "init_premodit", _stub_init_premodit)

    snap = _fake_snapshot_exomol()
    mdb_legacy = _FakeMdbLegacy(snap)
    mdb_with_snap = _FakeMdbWithSnapshot(snap)
    nu_grid = np.linspace(990.0, 1010.0, 8)
    policy = MemoryPolicy(allow_32bit=True)

    # Build three ways (use policy to avoid jax 64-bit dependency)
    opa_legacy = OpaPremodit(mdb_legacy, nu_grid, memory_policy=policy)
    opa_from_snap = OpaPremodit.from_snapshot(snap, nu_grid, memory_policy=policy)
    opa_from_mdb = OpaPremodit.from_mdb(mdb_with_snap, nu_grid, memory_policy=policy)

    # Apply identical params to all
    for opa in (opa_legacy, opa_from_snap, opa_from_mdb):
        opa.manual_setting(dE=5.0, Tref=1000.0, Twt=1100.0)

    # Pairwise equality using the class' __eq__
    assert opa_legacy == opa_from_snap
    assert opa_legacy == opa_from_mdb
    assert opa_from_snap == opa_from_mdb

    # Sanity: VO and legacy tuple both present and consistent
    for opa in (opa_legacy, opa_from_snap, opa_from_mdb):
        assert hasattr(opa, "pre_modit_info")
        mi, elg, ngr, ntg, R, pm = opa._get_info_tuple()
        info = opa.pre_modit_info
        np.testing.assert_allclose(info.multi_index_uniqgrid, mi)
        np.testing.assert_allclose(info.elower_grid, elg)
        np.testing.assert_allclose(info.ngamma_ref_grid, ngr)
        np.testing.assert_allclose(info.n_Texp_grid, ntg)
        assert info.R == R
        np.testing.assert_allclose(info.pmarray, pm)

