import numpy as np

from exojax.opacity.premodit.api import OpaPremodit
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot
from exojax.opacity.providers import (
    ExomolPartitionProvider,
    ExomolBroadening,
)


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


def test_default_provider_selection_exomol():
    snap = make_fake_snapshot_exomol()
    nu_grid = np.linspace(990.0, 1020.0, 16)
    opa = OpaPremodit.from_snapshot(snap, nu_grid, allow_32bit=True)
    assert isinstance(opa.pf_provider, ExomolPartitionProvider)
    assert isinstance(opa.broadening_strategy, ExomolBroadening)


def test_custom_provider_injection():
    class FakePF:
        def __init__(self, const_scalar=2.5):
            self.const_scalar = const_scalar

        def qr_single(self, T: float, Tref: float):
            return np.asarray(self.const_scalar)

        def qr_vector(self, Tarr: np.ndarray, Tref: float):
            Tarr = np.asarray(Tarr)
            return np.asarray([self.const_scalar for _ in Tarr])

    class FakeBroad:
        def __init__(self, n_lines: int):
            self._n = n_lines

        def compute(self, Tref_broadening: float):
            # deterministic placeholder outputs
            n_Texp = np.full(self._n, 0.123)
            gamma_ref = np.full(self._n, 4.567)
            return n_Texp, gamma_ref

    snap = make_fake_snapshot_exomol()
    nu_grid = np.linspace(990.0, 1020.0, 8)

    pf = FakePF(const_scalar=3.0)
    broad = FakeBroad(n_lines=snap.lines.nu_lines.size)

    opa = OpaPremodit.from_snapshot(
        snap,
        nu_grid,
        pf_provider=pf,
        broadening_strategy=broad,
        allow_32bit=True,
    )

    # Trigger apply_params via manual_setting to compute gamma_ref using the fake provider
    opa.manual_setting(dE=5.0, Tref=1000.0, Twt=1200.0)

    # Validate the broadening strategy output was used
    assert np.allclose(opa.gamma_ref, 4.567)
