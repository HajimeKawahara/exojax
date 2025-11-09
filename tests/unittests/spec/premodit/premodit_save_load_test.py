import numpy as np

from exojax.opacity.base import OpaCalc
from exojax.opacity.io.ioopa import saveopa_premodit
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.premodit.info import PreMODITInfo
from exojax.opacity.providers import ExomolPartitionProvider, ExomolBroadening
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog


def _build_minimal_ready_opa() -> OpaPremodit:
    """Fabricate a tiny but fully initialized OpaPremodit instance for I/O tests."""
    nu_grid = np.linspace(1000.0, 1002.0, 4)
    opa = object.__new__(OpaPremodit)
    OpaCalc.__init__(opa, nu_grid)

    opa.method = "premodit"
    opa.dbtype = "exomol"
    opa.molmass = 28.0
    opa.diffmode = 0
    opa.warning = True
    opa.wavelength_order = "descending"
    opa.wav = nu2wav(opa.nu_grid, wavelength_order=opa.wavelength_order, unit="AA")
    opa.resolution = resolution_eslog(opa.nu_grid)
    opa.version_auto_trange = 2
    opa.single_broadening = False
    opa.single_broadening_parameters = None
    opa.dE = 1.0
    opa.Tref = 1000.0
    opa.Twt = 1.0
    opa.Tmax = 1200.0
    opa.Tmin = 800.0
    opa.Tref_broadening = 296.0
    opa.dit_grid_resolution = 0.5
    opa.cutwing = 1.0
    opa.nstitch = 1
    opa.alias = "close"

    multi_index_uniqgrid = np.array([[0, 0]])
    elower_grid = np.array([0.1, 0.2])
    ngamma_ref_grid = np.array([0.05])
    n_Texp_grid = np.array([0.01])
    R = np.array([1.0])
    pmarray = np.ones(len(nu_grid) + 1, dtype=np.float64)
    pmarray[1::2] *= -1.0
    opa.ngrid_broadpar = len(multi_index_uniqgrid)
    opa.ngrid_elower = len(elower_grid)

    opa.pre_modit_info = PreMODITInfo(
        multi_index_uniqgrid=multi_index_uniqgrid,
        elower_grid=elower_grid,
        ngamma_ref_grid=ngamma_ref_grid,
        n_Texp_grid=n_Texp_grid,
        R=R,
        pmarray=pmarray,
    )
    opa.opainfo = opa.pre_modit_info.as_tuple()

    opa.gamma_ref = np.array([0.2])
    opa.n_Texp = np.array([0.7])
    # Structure mirrors real initspec output: (difforder, nu_bins, broadening_pts, elower_bins)
    opa.lbd_coeff = np.ones(
        (1, len(opa.nu_grid), opa.ngrid_broadpar, opa.ngrid_elower),
        dtype=np.float64,
    )

    T_gQT = np.array([100.0, 200.0])
    gQT = np.array([1.0, 2.0])
    opa.T_gQT = T_gQT
    opa.gQT = gQT
    opa.pf_provider = ExomolPartitionProvider(T_gQT, gQT)
    opa.broadening_strategy = ExomolBroadening(
        n_Texp=np.array([0.7]),
        alpha_ref=np.array([0.3]),
    )

    opa.set_aliasing()
    opa.memory_policy = None
    opa.ready = True
    return opa


def test_save_and_load_roundtrip(tmp_path):
    opa = _build_minimal_ready_opa()
    artifact = tmp_path / "opa_roundtrip"
    saveopa_premodit(opa, str(artifact), format="npz")

    loaded = OpaPremodit.from_saved_opa(str(artifact) + ".npz")

    assert loaded == opa
    assert np.allclose(loaded.gamma_ref, opa.gamma_ref)
    assert np.allclose(loaded.n_Texp, opa.n_Texp)
    assert np.array_equal(loaded.opainfo[0], opa.opainfo[0])

def test_save_and_load_roundtrip_zarr(tmp_path):
    opa = _build_minimal_ready_opa()
    artifact = tmp_path / "opa_roundtrip"
    saveopa_premodit(opa, str(artifact), format="zarr")

    loaded = OpaPremodit.from_saved_opa(str(artifact) + ".zarr")

    assert loaded == opa
    assert np.allclose(loaded.gamma_ref, opa.gamma_ref)
    assert np.allclose(loaded.n_Texp, opa.n_Texp)
    assert np.array_equal(loaded.opainfo[0], opa.opainfo[0])


def test_save_and_load_roundtrip_zarr_xsvector(tmp_path):
    opa = _build_minimal_ready_opa()
    artifact = tmp_path / "opa_roundtrip"
    saveopa_premodit(opa, str(artifact), format="zarr")

    loaded = OpaPremodit.from_saved_opa(str(artifact) + ".zarr")

    P = 1.0  # bar
    T_1 = 500.0  # K
    xsv_1 = loaded.xsvector(T_1, P)  # cm2
