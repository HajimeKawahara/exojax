import numpy as np

from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.policies import MemoryPolicy
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot
from exojax.utils.grids import wavenumber_grid


def _fake_snap():
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


def _stub_initspec(monkeypatch):
    import exojax.opacity.initspec as initspec

    def _stub_init_premodit(
        nu_lines,
        nu_grid,
        elower,
        gamma_ref,
        n_Texp,
        line_strength_Tref,
        Twt,
        *,
        Tref,
        Tref_broadening,
        Tmax,
        Tmin,
        dE,
        dit_grid_resolution,
        diffmode,
        single_broadening,
        single_broadening_parameters,
        warning,
    ):
        L = len(nu_grid)
        lbd = np.zeros((diffmode + 1, L, 2, 1))
        mi = np.array([[0, 0], [1, 1]])
        elg = np.array([0.0])
        ngr = np.array([0.01, 0.02])
        ntg = np.array([0.5, 0.5])
        R = 1.0
        pm = np.array([1.0])
        return lbd, mi, elg, ngr, ntg, R, pm

    monkeypatch.setattr(initspec, "init_premodit", _stub_init_premodit)


def test_delete_mdb_after_init_true_logs(monkeypatch, caplog):
    _stub_initspec(monkeypatch)
    caplog.set_level("INFO", logger="exojax.opacity.premodit.api")

    snap = _fake_snap()
    nu_grid, _, _ = wavenumber_grid(990.0, 1010.0, 8, xsmode="premodit")
    OpaPremodit.from_snapshot(
        snap,
        nu_grid,
        memory_policy=MemoryPolicy(allow_32bit=True),
        delete_mdb_after_init=True,
    )

    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "delete mdb" in msgs


def test_delete_mdb_after_init_false_no_log(monkeypatch, caplog):
    _stub_initspec(monkeypatch)
    caplog.set_level("INFO", logger="exojax.opacity.premodit.api")

    snap = _fake_snap()
    nu_grid, _, _ = wavenumber_grid(990.0, 1010.0, 8, xsmode="premodit")
    OpaPremodit.from_snapshot(
        snap,
        nu_grid,
        memory_policy=MemoryPolicy(allow_32bit=True),
        delete_mdb_after_init=False,
    )

    msgs = "\n".join(r.getMessage() for r in caplog.records)
    assert "delete mdb" not in msgs

