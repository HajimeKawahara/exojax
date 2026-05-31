import numpy as np
import jax.numpy as jnp
import pytest

from exojax.opacity.ckd.api import OpaCKD, CKDTableInfo
from exojax.opacity.ckd.io import _ckd_save_as_npz


class _DummyBaseOpa:
    def __init__(self, nu_grid):
        self.nu_grid = nu_grid

    def meta(self):
        return {"tag": "dummy"}


def _write_ckd_npz(path, **overrides):
    arrays = {
        "meta": np.frombuffer(
            (
                '{"Ng": 2, "band_width": 40.0, "band_spacing": "linear", '
                '"base_fingerprint_hash": "dummy"}'
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
        "log_kggrid": np.log(np.asarray([[[[1.0, 2.0], [3.0, 4.0]]]])),
        "ggrid": np.asarray([0.25, 0.75]),
        "weights": np.asarray([0.5, 0.5]),
        "T_grid": np.asarray([800.0]),
        "P_grid": np.asarray([0.1]),
        "nu_bands": np.asarray([120.0, 160.0]),
        "band_edges": np.asarray([[100.0, 140.0], [140.0, 180.0]]),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


@pytest.mark.parametrize("band_spacing", ["linear", "log"])
def test_ckd_table_roundtrip(tmp_path, band_spacing):
    base = _DummyBaseOpa(jnp.linspace(100.0, 200.0, 5))
    opa = OpaCKD(base, Ng=2, band_width=25.0, band_spacing=band_spacing)

    nu_bands = jnp.asarray([120.0, 160.0])
    band_edges = jnp.asarray([[100.0, 140.0], [140.0, 180.0]])
    log_kggrid = jnp.log(jnp.asarray([[[[1.0, 2.0], [3.0, 4.0]]]]))
    ggrid = jnp.asarray([0.25, 0.75])
    weights = jnp.asarray([0.5, 0.5])
    T_grid = jnp.asarray([800.0])
    P_grid = jnp.asarray([0.1])

    opa.nu_bands = nu_bands
    opa.band_edges = band_edges
    opa.ckd_info = CKDTableInfo(
        log_kggrid=log_kggrid,
        ggrid=ggrid,
        weights=weights,
        T_grid=T_grid,
        P_grid=P_grid,
        nu_bands=nu_bands,
        band_edges=band_edges,
    )
    opa.ready = True

    out_path = tmp_path / "ckd_table.npz"
    _ckd_save_as_npz(opa, out_path)

    loaded = OpaCKD.from_saved_tables(base, out_path)

    assert loaded.ready is True
    assert loaded.Ng == opa.Ng
    assert loaded.band_width == pytest.approx(opa.band_width)
    assert loaded.band_spacing == opa.band_spacing

    inplace = OpaCKD(base, Ng=4, band_width=1.0, band_spacing="linear")
    inplace.load_tables(out_path)
    assert inplace.ready is True
    assert inplace.base_opa is base
    assert inplace.Ng == opa.Ng
    assert inplace.band_spacing == loaded.band_spacing
    np.testing.assert_allclose(np.asarray(inplace.ckd_info.log_kggrid), np.asarray(log_kggrid))
    np.testing.assert_allclose(np.asarray(loaded.ckd_info.ggrid), np.asarray(ggrid))
    np.testing.assert_allclose(
        np.asarray(loaded.ckd_info.log_kggrid), np.asarray(log_kggrid)
    )
    np.testing.assert_array_equal(np.asarray(loaded.nu_bands), np.asarray(nu_bands))
    np.testing.assert_array_equal(
        np.asarray(loaded.ckd_info.band_edges), np.asarray(band_edges)
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"weights": np.asarray([1.0])}, "weights shape"),
        ({"weights": np.asarray([0.5, 0.0])}, "weights must be positive"),
        ({"weights": np.asarray([0.25, 0.25])}, "weights must sum to one"),
        ({"log_kggrid": np.asarray([[[[0.0, np.nan], [0.0, 0.0]]]])}, "log_kggrid"),
        ({"T_grid": np.asarray([[800.0]])}, "one-dimensional"),
        ({"P_grid": np.asarray([0.1, 1.0])}, "shape does not match"),
        ({"T_grid": np.asarray([0.0])}, "must be positive"),
    ],
)
def test_ckd_saved_table_rejects_invalid_arrays(tmp_path, overrides, message):
    path = tmp_path / "bad_ckd_table.npz"
    _write_ckd_npz(path, **overrides)

    with pytest.raises(ValueError, match=message):
        OpaCKD.from_saved_tables(path)


def test_ckd_load_tables_validates_existing_base_opa(tmp_path):
    base = _DummyBaseOpa(jnp.linspace(100.0, 200.0, 5))
    other_base = _DummyBaseOpa(jnp.linspace(110.0, 210.0, 5))
    opa = OpaCKD(base, Ng=2, band_width=25.0, band_spacing="linear")

    nu_bands = jnp.asarray([120.0, 160.0])
    band_edges = jnp.asarray([[100.0, 140.0], [140.0, 180.0]])
    opa.ckd_info = CKDTableInfo(
        log_kggrid=jnp.log(jnp.asarray([[[[1.0, 2.0], [3.0, 4.0]]]])),
        ggrid=jnp.asarray([0.25, 0.75]),
        weights=jnp.asarray([0.5, 0.5]),
        T_grid=jnp.asarray([800.0]),
        P_grid=jnp.asarray([0.1]),
        nu_bands=nu_bands,
        band_edges=band_edges,
    )
    opa.nu_bands = nu_bands
    opa.band_edges = band_edges
    opa.ready = True
    path = tmp_path / "ckd_table.npz"
    _ckd_save_as_npz(opa, path)

    compatible = OpaCKD(base, Ng=4, band_width=1.0, band_spacing="linear")
    compatible.load_tables(path)
    assert compatible.base_opa is base

    incompatible = OpaCKD(other_base, Ng=4, band_width=1.0, band_spacing="linear")
    with pytest.raises(ValueError, match="base_opa fingerprint"):
        incompatible.load_tables(path)
