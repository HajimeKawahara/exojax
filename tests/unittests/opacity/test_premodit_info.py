import numpy as np

from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.policies import MemoryPolicy
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot


def fake_snapshot_exomol():
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
    # include minimal broadening payload
    return MDBSnapshot(
        meta=meta,
        lines=lines,
        n_Texp=np.array([0.5, 0.5]),
        alpha_ref=np.array([0.1, 0.1]),
    )


def test_apply_params_sets_pre_modit_info(monkeypatch):
    # stub initspec.init_premodit to avoid heavy computation / I/O
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
        # lbd_coeff shape: (diffmode+1, L, N_broadening, N_elower)
        lbd_coeff = np.zeros((diffmode + 1, 4, 2, 3))
        mi = np.array([[0, 0], [1, 1]])  # multi_index_uniqgrid
        elg = np.array([0.0, 1.0, 2.0])  # elower_grid
        ngr = np.array([0.01, 0.02])  # ngamma_ref_grid
        ntg = np.array([0.5, 0.5])  # n_Texp_grid
        R = 1.0
        pm = np.array([42.0])  # pmarray dummy
        return lbd_coeff, mi, elg, ngr, ntg, R, pm

    monkeypatch.setattr(initspec, "init_premodit", _stub_init_premodit)

    snap = fake_snapshot_exomol()
    nu_grid = np.linspace(990.0, 1010.0, 8)

    # Build via DI path (providers auto-selected)
    opa = OpaPremodit.from_snapshot(snap, nu_grid, memory_policy=MemoryPolicy(allow_32bit=True))
    opa.manual_setting(dE=5.0, Tref=1000.0, Twt=1100.0)

    # VO must exist and mirror opainfo
    assert hasattr(opa, "pre_modit_info")
    mi, elg, ngr, ntg, R, pm = opa.opainfo
    info = opa.pre_modit_info
    np.testing.assert_allclose(info.multi_index_uniqgrid, mi)
    np.testing.assert_allclose(info.elower_grid, elg)
    np.testing.assert_allclose(info.ngamma_ref_grid, ngr)
    np.testing.assert_allclose(info.n_Texp_grid, ntg)
    assert info.R == R
    np.testing.assert_allclose(info.pmarray, pm)

    # tuple view matches legacy opainfo layout
    tup = info.as_tuple()
    assert len(tup) == 6
    np.testing.assert_allclose(tup[0], mi)
    np.testing.assert_allclose(tup[1], elg)
    np.testing.assert_allclose(tup[2], ngr)
    np.testing.assert_allclose(tup[3], ntg)
    assert tup[4] == R
    np.testing.assert_allclose(tup[5], pm)

    # derived counters unchanged
    assert opa.ngrid_broadpar == len(mi)
    assert opa.ngrid_elower == len(elg)
