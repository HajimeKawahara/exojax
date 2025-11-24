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
        nu_lines=np.array([1000.0, 1001.0]),
        elower=np.array([10.0, 20.0]),
        line_strength_ref_original=np.array([1e-30, 2e-30]),
    )
    return MDBSnapshot(
        meta=meta,
        lines=lines,
        n_Texp=np.array([0.5, 0.5]),
        alpha_ref=np.array([0.1, 0.1]),
    )


def test_get_info_tuple_prefers_vo(monkeypatch):
    # Stub heavy init to keep the test hermetic
    import exojax.opacity.initspec as initspec

    def _stub_init_premodit(*args, **kwargs):
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
    nu_grid = np.linspace(990.0, 1010.0, 8)
    opa = OpaPremodit.from_snapshot(snap, nu_grid, memory_policy=MemoryPolicy(allow_32bit=True))
    opa.manual_setting(dE=5.0, Tref=1000.0, Twt=1100.0)

    # Both paths present: VO and legacy tuple must match
    t1 = opa._get_info_tuple()
    t2 = opa.pre_modit_info.as_tuple()
    for a, b in zip(t1, t2):
        np.testing.assert_allclose(a, b)

    # Remove legacy tuple to ensure accessor still works via VO
    opa.opainfo = None
    t3 = opa._get_info_tuple()
    for a, b in zip(t3, t2):
        np.testing.assert_allclose(a, b)
