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


def test_memory_policy_overrides_ctor(monkeypatch):
    # keep heavy init hermetic
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

    snap = _fake_snap()
    # PreMODIT expects an ESLOG grid; use helper to generate it
    nu_grid, _, _ = wavenumber_grid(990.0, 1010.0, 8, xsmode="premodit")

    # ctor sets allow_32bit=False, nstitch=1, cutwing=1.0
    # policy overrides them to True, 2, 0.5
    policy = MemoryPolicy(allow_32bit=True, nstitch=2, cutwing=0.5)
    opa = OpaPremodit.from_snapshot(snap, nu_grid, memory_policy=policy)

    # Resolved values must come from policy
    assert opa.nstitch == 2
    assert opa.cutwing == 0.5
    assert opa.memory_policy == policy

    # Sanity: apply_params still runs
    opa.manual_setting(dE=5.0, Tref=1000.0, Twt=1100.0)
    assert hasattr(opa, "pre_modit_info")
