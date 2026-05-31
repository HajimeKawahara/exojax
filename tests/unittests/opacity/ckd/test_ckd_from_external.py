import numpy as np
import jax.numpy as jnp
import pytest


def test_from_external_exomolop_basic(monkeypatch, tmp_path):
    # Synthetic CKD grid dimensions
    nT, nP, Ng, nBands = 2, 2, 3, 2

    # Build synthetic data returned by provider.exomolop.load_ckd
    temperatures = np.array([500.0, 1000.0], dtype=np.float32)
    pressures = np.array([0.1, 1.0], dtype=np.float32)
    samples = np.array([0.2, 0.5, 0.8], dtype=np.float32)  # g-grid
    weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    nu_centers = np.array([1000.0, 1005.0], dtype=np.float32)

    xsgrid = np.zeros((nT, nP, Ng, nBands), dtype=np.float32)
    for it in range(nT):
        for ip in range(nP):
            for ig in range(Ng):
                for ib in range(nBands):
                    # Unique, positive values per index
                    xsgrid[it, ip, ig, ib] = (
                        (it + 1) * 10.0 + (ip + 1) + ig * 0.1 + ib * 0.01
                    )

    def fake_load_ckd(_):
        # Return values follow provider.exomolop.load_ckd contract
        molecule, mol_mass = "H2O", 18.0
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            molecule,
            mol_mass,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = lambda path: path
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg
    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    table_path = tmp_path / "dummy.h5"
    table_path.touch()
    ckd = OpaCKD.from_external("exomolop", table_path)

    # Basic shape checks
    assert ckd.ready is True
    assert ckd.Ng == Ng
    assert ckd.ckd_info.log_kggrid.shape == (nT, nP, Ng, nBands)
    assert ckd.ckd_info.ggrid.shape == (Ng,)
    assert ckd.ckd_info.nu_bands.shape == (nBands,)
    assert ckd.ckd_info.band_edges.shape == (nBands, 2)
    np.testing.assert_allclose(
        ckd.ckd_info.band_edges,
        [[997.5, 1002.5], [1002.5, 1007.5]],
    )
    assert ckd.band_width == 5.0
    assert ckd.band_spacing == "external"

    # Interpolation at exact grid points should equal the original slice
    out = ckd.xsarray_ckd(temperatures[0], pressures[0])  # shape (Ng, nBands)
    np.testing.assert_allclose(np.asarray(out), xsgrid[0, 0, :, :], rtol=1e-6, atol=0.0)

    # Tensor interpolation across layers
    T_arr = jnp.array(temperatures)
    P_arr = jnp.array(pressures)
    xst = ckd.xstensor_ckd(T_arr, P_arr)  # (2, Ng, nBands)
    np.testing.assert_allclose(np.asarray(xst[0]), xsgrid[0, 0], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(np.asarray(xst[1]), xsgrid[1, 1], rtol=1e-6, atol=0.0)


def test_from_external_exomolop_nurange(monkeypatch, tmp_path):
    # Synthetic CKD grid dimensions
    nT, nP, Ng, nBands = 1, 1, 2, 4

    temperatures = np.array([500.0], dtype=np.float32)
    pressures = np.array([0.1], dtype=np.float32)
    samples = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    nu_centers = np.array([900.0, 1000.0, 1100.0, 1200.0], dtype=np.float32)

    xsgrid = np.zeros((nT, nP, Ng, nBands), dtype=np.float32)
    xsgrid[0, 0, :, 0] = [1.0, 2.0]
    xsgrid[0, 0, :, 1] = [10.0, 20.0]
    xsgrid[0, 0, :, 2] = [100.0, 200.0]
    xsgrid[0, 0, :, 3] = [1000.0, 2000.0]

    def fake_load_ckd(_):
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = lambda path: path
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg
    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    table_path = tmp_path / "dummy.h5"
    table_path.touch()
    ckd = OpaCKD.from_external("exomolop", table_path, nurange=(950.0, 1150.0))

    # The requested range is covered by overlapping band edges, not just centers.
    assert ckd.ckd_info.nu_bands.shape == (4,)
    np.testing.assert_allclose(ckd.ckd_info.nu_bands, [900.0, 1000.0, 1100.0, 1200.0])
    np.testing.assert_allclose(
        ckd.ckd_info.band_edges,
        [[850.0, 950.0], [950.0, 1050.0], [1050.0, 1150.0], [1150.0, 1250.0]],
    )
    assert ckd.ckd_info.log_kggrid.shape == (nT, nP, Ng, 4)
    np.testing.assert_allclose(
        np.exp(np.asarray(ckd.ckd_info.log_kggrid)),
        xsgrid,
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "nurange, message",
    [
        ((1150.0, 950.0), "nu_min <= nu_max"),
        ((950.0, np.nan), "finite"),
        (1000.0, "2 or more"),
    ],
)
def test_from_external_exomolop_invalid_nurange(
    monkeypatch, tmp_path, nurange, message
):
    temperatures = np.array([500.0], dtype=np.float32)
    pressures = np.array([0.1], dtype=np.float32)
    samples = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    nu_centers = np.array([900.0, 1000.0, 1100.0, 1200.0], dtype=np.float32)
    xsgrid = np.ones((1, 1, 2, 4), dtype=np.float32)

    def fake_load_ckd(_):
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = lambda path: path
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg

    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    table_path = tmp_path / "dummy.h5"
    table_path.touch()
    with pytest.raises(ValueError, match=message):
        OpaCKD.from_external("exomolop", table_path, nurange=nurange)


def test_from_external_exomolop_sorts_descending_wavenumber_bands(
    monkeypatch, tmp_path
):
    nT, nP, Ng, nBands = 1, 1, 2, 3

    temperatures = np.array([500.0], dtype=np.float32)
    pressures = np.array([0.1], dtype=np.float32)
    samples = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    nu_centers = np.array([1200.0, 1100.0, 1000.0], dtype=np.float32)

    xsgrid = np.zeros((nT, nP, Ng, nBands), dtype=np.float32)
    xsgrid[0, 0, :, 0] = [120.0, 121.0]
    xsgrid[0, 0, :, 1] = [110.0, 111.0]
    xsgrid[0, 0, :, 2] = [100.0, 101.0]

    def fake_load_ckd(_):
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = lambda path: path
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg
    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    table_path = tmp_path / "dummy.h5"
    table_path.touch()
    ckd = OpaCKD.from_external("exomolop", table_path)

    np.testing.assert_allclose(ckd.ckd_info.nu_bands, [1000.0, 1100.0, 1200.0])
    np.testing.assert_allclose(
        ckd.ckd_info.band_edges,
        [[950.0, 1050.0], [1050.0, 1150.0], [1150.0, 1250.0]],
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(ckd.ckd_info.log_kggrid)),
        xsgrid[..., [2, 1, 0]],
        rtol=1e-6,
        atol=0.0,
    )


def test_from_external_exomolop_missing_directory_download(monkeypatch):
    temperatures = np.array([500.0], dtype=np.float32)
    pressures = np.array([0.1], dtype=np.float32)
    samples = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    nu_centers = np.array([1000.0, 1100.0], dtype=np.float32)
    xsgrid = np.ones((1, 1, 2, 2), dtype=np.float32)
    downloaded_path = "downloaded_table.h5"
    calls = []

    def fake_download(path):
        calls.append(str(path))
        return downloaded_path

    def fake_load_ckd(path):
        assert path == downloaded_path
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = fake_download
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg
    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    ckd = OpaCKD.from_external("exomolop", "missing/without_suffix")

    assert calls == ["missing/without_suffix"]
    assert ckd.ready is True


def test_from_external_exomolop_missing_explicit_h5_raises(tmp_path):
    from exojax.opacity.ckd.api import OpaCKD

    missing_table = tmp_path / "missing_table.h5"

    with pytest.raises(FileNotFoundError, match="CKD table file does not exist"):
        OpaCKD.from_external("exomolop", missing_table)


def test_from_external_exomolop_existing_directory_h5(monkeypatch, tmp_path):
    temperatures = np.array([500.0], dtype=np.float32)
    pressures = np.array([0.1], dtype=np.float32)
    samples = np.array([0.25, 0.75], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    nu_centers = np.array([1000.0, 1100.0], dtype=np.float32)
    xsgrid = np.ones((1, 1, 2, 2), dtype=np.float32)
    table_path = tmp_path / "local_table.h5"
    table_path.touch()

    def fake_download(_path):
        raise AssertionError("Existing local h5 tables should not be downloaded")

    def fake_load_ckd(path):
        assert path == table_path
        return (
            xsgrid.copy(),
            samples.copy(),
            weights.copy(),
            temperatures.copy(),
            pressures.copy(),
            nu_centers.copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = fake_download
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg
    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    ckd = OpaCKD.from_external("exomolop", tmp_path)

    assert ckd.ready is True


def test_from_external_exomolop_existing_directory_multiple_h5(tmp_path):
    (tmp_path / "a.h5").touch()
    (tmp_path / "b.h5").touch()

    from exojax.opacity.ckd.api import OpaCKD

    try:
        OpaCKD.from_external("exomolop", tmp_path)
    except ValueError as exc:
        assert "Multiple CKD h5 files found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for multiple h5 files")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda payload: payload.__setitem__("xsgrid", payload["xsgrid"][0]), "four-dimensional"),
        (
            lambda payload: payload.__setitem__("xsgrid", payload["xsgrid"][..., :-1]),
            "shape does not match",
        ),
        (lambda payload: payload.__setitem__("weights", np.array([1.0])), "weights must match"),
        (
            lambda payload: payload.__setitem__("weights", np.array([0.5, 0.0])),
            "weights must be positive",
        ),
        (
            lambda payload: payload.__setitem__("weights", np.array([0.25, 0.25])),
            "weights must sum to one",
        ),
        (
            lambda payload: payload.__setitem__("temperatures", np.array([0.0])),
            "temperatures must be positive",
        ),
        (lambda payload: payload["xsgrid"].__setitem__((0, 0, 0, 0), 0.0), "xsgrid must be positive"),
    ],
)
def test_from_external_exomolop_invalid_table_arrays(
    monkeypatch, tmp_path, mutate, message
):
    payload = {
        "temperatures": np.array([500.0], dtype=np.float32),
        "pressures": np.array([0.1], dtype=np.float32),
        "samples": np.array([0.25, 0.75], dtype=np.float32),
        "weights": np.array([0.5, 0.5], dtype=np.float32),
        "nu_centers": np.array([1000.0, 1100.0], dtype=np.float32),
        "xsgrid": np.ones((1, 1, 2, 2), dtype=np.float32),
    }
    mutate(payload)

    def fake_load_ckd(_):
        return (
            payload["xsgrid"].copy(),
            payload["samples"].copy(),
            payload["weights"].copy(),
            payload["temperatures"].copy(),
            payload["pressures"].copy(),
            payload["nu_centers"].copy(),
            "H2O",
            18.0,
        )

    from exojax.opacity.ckd.api import OpaCKD
    import sys
    from types import ModuleType

    dummy_exomolop = ModuleType("exojax.provider.exomolop")
    dummy_exomolop.load_ckd = fake_load_ckd
    dummy_exomolop.download_exomolop_h5 = lambda path: path
    monkeypatch.setitem(sys.modules, "exojax.provider.exomolop", dummy_exomolop)
    import exojax.provider as provider_pkg

    monkeypatch.setattr(provider_pkg, "exomolop", dummy_exomolop, raising=False)

    table_path = tmp_path / "dummy.h5"
    table_path.touch()
    with pytest.raises(ValueError, match=message):
        OpaCKD.from_external("exomolop", table_path)
