"""Regression tests for atomic line arrays across selection and transfer."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_LINE_FIELDS = (
    "A", "elower", "eupper", "gupper", "jlower", "jupper", "QTmask",
    "ielem", "iion", "gamRad", "gamSta", "vdWdamp",
)
_METADATA = ("solarA", "atomicmass", "ionE")
_HOST_FIELDS = ("nu_lines", "Sij0") + tuple("_" + name for name in _LINE_FIELDS)


@pytest.fixture(params=["vald", "kurucz"])
def make_adb(request, monkeypatch, tmp_path):
    """Supply normalized transitions while retaining real atomic metadata."""
    api = importlib.import_module(f"exojax.database.{request.param}.api")
    transitions = (
        np.array([1.0e6, 2.0e6, 3.0e6, 4.0e6]),
        np.array([1000.0, 1001.0, 1002.0, 1003.0]),
        np.array([0.0, 10.0, 20.0, 30.0]),
        np.array([1000.0, 1011.0, 1022.0, 1033.0]),
        np.array([3.0, 4.0, 6.0, 3.0]),
        np.array([0.0, 0.5, 1.5, 0.0]),
        np.array([1.0, 1.5, 2.5, 1.0]),
        np.array([26, 11, 26, 12]),
        np.array([1, 1, 2, 1]),
        np.array([8.0, 8.1, 8.2, 8.3]),
        np.array([-6.0, -6.1, -6.2, -6.3]),
        np.array([-7.0, -7.1, -7.2, -7.3]),
    )

    def read_transitions(_):
        return tuple(array.copy() for array in transitions)

    if request.param == "vald":
        monkeypatch.setattr(api, "_load_vald_dataframe", lambda *args: None)
        monkeypatch.setattr(api, "pickup_param", read_transitions)
        adb_class = api.AdbVald
    else:
        monkeypatch.setattr(api, "read_kurucz", read_transitions)
        adb_class = api.AdbKurucz

    def create(**kwargs):
        return adb_class(tmp_path / "synthetic.lines", **kwargs)

    return create


def _assert_arrays(adb, expected, transferred):
    for name in _HOST_FIELDS:
        actual = getattr(adb, name)
        assert isinstance(actual, np.ndarray), name
        np.testing.assert_array_equal(actual, expected[name], err_msg=name)
    for name in _METADATA:
        actual = getattr(adb, name)
        assert isinstance(actual, jax.Array if transferred else np.ndarray), name
        np.testing.assert_array_equal(actual, expected[name], err_msg=name)

    assert hasattr(adb, "dev_nu_lines") == transferred
    if transferred:
        for name in _LINE_FIELDS:
            actual = getattr(adb, name)
            assert isinstance(actual, jax.Array), name
            np.testing.assert_array_equal(actual, expected["_" + name], err_msg=name)
        np.testing.assert_array_equal(adb.dev_nu_lines, expected["nu_lines"])
        np.testing.assert_allclose(adb.logsij0, np.log(expected["Sij0"]), atol=0.0)
        for name in ("QTmask", "ielem", "iion"):
            assert np.issubdtype(getattr(adb, name).dtype, np.integer)
        for name in ("jlower", "jupper"):
            assert np.issubdtype(getattr(adb, name).dtype, np.floating)


def _snapshot(adb):
    return {
        name: np.array(getattr(adb, name), copy=True)
        for name in _HOST_FIELDS + _METADATA
    }


def test_atomic_host_construction_and_delayed_transfer(make_adb):
    adb = make_adb(nurange=[999.5, 1002.5], gpu_transfer=False)
    np.testing.assert_array_equal(adb.nu_lines, [1000.0, 1001.0, 1002.0])
    np.testing.assert_array_equal(adb._ielem, [26, 11, 26])
    np.testing.assert_array_equal(adb._iion, [1, 1, 2])
    np.testing.assert_allclose(adb.solarA, [-4.5, -5.76, -4.5])
    np.testing.assert_allclose(adb.atomicmass, [55.847, 22.98981, 55.847])
    np.testing.assert_allclose(adb.ionE, [7.9024681, 5.13907696, 16.19921])
    expected = _snapshot(adb)
    _assert_arrays(adb, expected, transferred=False)

    adb.generate_jnp_arrays()
    _assert_arrays(adb, expected, transferred=True)
    np.testing.assert_array_equal(adb.jlower, [0.0, 0.5, 1.5])
    np.testing.assert_array_equal(adb.jupper, [1.0, 1.5, 2.5])


@pytest.mark.parametrize("transfer", ["host", "eager", "delayed"])
def test_atomic_repeated_masking_preserves_line_alignment(make_adb, transfer):
    adb = make_adb(nurange=[999.5, 1002.5], gpu_transfer=transfer == "eager")
    if transfer == "delayed":
        adb.generate_jnp_arrays()
    expected = _snapshot(adb)
    for mask in (np.array([False, True, True]), np.array([True, False])):
        expected = {name: values[mask] for name, values in expected.items()}
        adb.masking(mask)
        _assert_arrays(adb, expected, transferred=transfer != "host")

    adb.generate_jnp_arrays()
    _assert_arrays(adb, expected, transferred=True)


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_atomic_empty_selection_remains_consistent(make_adb, gpu_transfer):
    adb = make_adb(gpu_transfer=gpu_transfer)
    expected = {name: values[:0] for name, values in _snapshot(adb).items()}
    with pytest.warns(UserWarning, match="no lines are selected"):
        adb.masking(np.zeros(len(adb.nu_lines), dtype=bool))
    _assert_arrays(adb, expected, transferred=gpu_transfer)

    with pytest.warns(UserWarning, match="no lines are selected"):
        empty = make_adb(nurange=[2000.0, 2001.0], gpu_transfer=gpu_transfer)
    _assert_arrays(empty, expected, transferred=gpu_transfer)


def test_atomic_masked_opacity_matches_constructor_selection(make_adb):
    from exojax.opacity import OpaDirect

    adb = make_adb()
    adb.masking(np.array([False, True, True, False]))
    selected = make_adb(nurange=[1000.5, 1002.5])
    nu_grid = np.linspace(1000.5, 1002.5, 33)
    temperatures = jnp.array([1500.0, 2000.0, 2500.0])
    pressures = jnp.array([0.01, 0.1, 1.0])
    actual = OpaDirect(adb, nu_grid).xsmatrix(temperatures, pressures)
    expected = OpaDirect(selected, nu_grid).xsmatrix(temperatures, pressures)

    assert actual.shape == (3, len(nu_grid))
    assert np.all(np.isfinite(actual))
    assert np.all(actual > 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
