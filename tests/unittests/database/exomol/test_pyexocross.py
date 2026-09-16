"""Acceptance checks for the optional PyExoCross ExoMol reader."""

import builtins
import bz2
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from exojax.database.exomol.api import MdbExomol
from exojax.test.data import get_testdata_filename


NURANGE = (4330.0, 4360.0)
LINE_FIELDS = (
    "nu_lines", "A", "elower", "gpp", "jlower", "jupper",
    "line_strength_ref_original", "logsij0", "alpha_ref", "n_Texp",
    "gamma_natural",
)


def _block_pyexocross(monkeypatch):
    real_import = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyexocross" or name.startswith("pyexocross."):
            raise ModuleNotFoundError("No module named 'pyexocross'", name="pyexocross")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setitem(sys.modules, "pyexocross", None)


@pytest.fixture
def pyexocross():
    module = pytest.importorskip("pyexocross")
    if module.__version__ != "1.1.16":
        pytest.skip("The optional ExoMol reader supports PyExoCross 1.1.16.")
    return module


@pytest.fixture
def raw_dataset(tmp_path, monkeypatch):
    """Copy text sources only, with all reference downloads disabled."""
    import requests
    from exojax.database.exomol import _pyexocross_download

    def unexpected_download(*args, **kwargs):
        pytest.fail("A local ExoMol fixture attempted a download.")

    monkeypatch.setattr(MdbExomol, "download", unexpected_download)
    monkeypatch.setattr(MdbExomol, "is_registered", lambda self: False)
    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_download)
    monkeypatch.setattr(_pyexocross_download, "urlopen", unexpected_download)

    def copy(molecule="CO", tag="data"):
        isotope = {"CO": "12C-16O", "H2O": "1H2-16O"}[molecule]
        source = Path(get_testdata_filename(molecule)) / isotope / "SAMPLE"
        target = tmp_path / tag / molecule / isotope / "SAMPLE"
        target.mkdir(parents=True)
        for file in source.iterdir():
            if file.name.endswith((".bz2", ".def", ".pf", ".broad")):
                shutil.copy2(file, target / file.name)
        return target

    return copy


def _load(path, *, backend="pyexocross", **options):
    kwargs = dict(
        nurange=NURANGE, gpu_transfer=False, inherit_dataframe=True,
        broadf_download=False, local_databases=str(path.parents[2]),
    )
    kwargs.update(options)
    if backend == "radis":
        kwargs["engine"] = "pytables"
    return MdbExomol(path, backend=backend, **kwargs)


def _assert_same_lines(actual, reference):
    actual_order = np.argsort(actual.nu_lines, kind="stable")
    reference_order = np.argsort(reference.nu_lines, kind="stable")
    for name in LINE_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name))[actual_order],
            np.asarray(getattr(reference, name))[reference_order],
            rtol=2e-12, atol=0, err_msg=name,
        )
    assert actual.molmass == reference.molmass
    np.testing.assert_array_equal(actual.T_gQT, reference.T_gQT)
    np.testing.assert_array_equal(actual.gQT, reference.gQT)
    for temperature in (296.0, 500.5, 1000.25, 1499.75):
        np.testing.assert_allclose(
            actual.QT_interp(temperature), reference.QT_interp(temperature),
            rtol=2e-12, atol=0,
        )
        np.testing.assert_allclose(
            actual.line_strength(temperature)[actual_order],
            reference.line_strength(temperature)[reference_order],
            rtol=2e-12, atol=0,
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"backend": "unknown"}, "backend"),
        ({"backend": "pyexocross", "engine": "pytables"}, "engine"),
        ({"backend": "pyexocross", "engine": "vaex"}, "engine"),
    ],
)
def test_invalid_options_fail_before_optional_import_or_io(monkeypatch, tmp_path, options, message):
    _block_pyexocross(monkeypatch)
    with pytest.raises(ValueError, match=message):
        MdbExomol(tmp_path / "missing/CO/12C-16O/SAMPLE", NURANGE, **options)


def test_missing_pyexocross_has_installation_hint_before_io(monkeypatch, tmp_path):
    _block_pyexocross(monkeypatch)
    with pytest.raises(ImportError, match="(?i)pyexocross") as error:
        MdbExomol(
            tmp_path / "missing/CO/12C-16O/SAMPLE", NURANGE,
            backend="pyexocross", broadf_download=False,
        )
    assert "pip install" in str(error.value)


def test_default_radis_backend_does_not_require_pyexocross(raw_dataset, monkeypatch):
    path = raw_dataset()
    _block_pyexocross(monkeypatch)
    mdb = MdbExomol(
        path, NURANGE, engine="pytables", broadf_download=False,
        gpu_transfer=False, local_databases=str(path.parents[2]),
    )
    assert mdb.backend == "radis"
    assert len(mdb.nu_lines) == 220
    assert mdb.molmass == 28.0101


@pytest.mark.parametrize(
    "molecule, options, expected_count",
    [
        ("CO", {}, 220),
        ("CO", {"bkgdatm": "He"}, 220),
        ("CO", {"broadf": False}, 220),
        ("CO", {"crit": 1e-25, "elower_max": 2000.0}, 9),
        ("H2O", {}, 175),
        ("H2O", {"bkgdatm": "He"}, 175),
        ("H2O", {"crit": 1e-25, "elower_max": 2000.0}, 19),
    ],
)
def test_raw_loader_matches_radis(pyexocross, raw_dataset, molecule, options, expected_count):
    path = raw_dataset(molecule)
    actual = _load(path, **options)
    reference = _load(path, backend="radis", **options)
    assert actual.backend == "pyexocross"
    assert actual.dbtype == "exomol"
    assert len(actual.nu_lines) == expected_count
    _assert_same_lines(actual, reference)

    snapshot = actual.to_snapshot()
    np.testing.assert_array_equal(snapshot.lines.nu_lines, actual.nu_lines)
    np.testing.assert_array_equal(snapshot.lines.elower, actual.elower)
    np.testing.assert_array_equal(
        snapshot.lines.line_strength_ref_original, actual.line_strength_ref_original,
    )
    np.testing.assert_array_equal(snapshot.alpha_ref, actual.alpha_ref)
    np.testing.assert_array_equal(snapshot.n_Texp, actual.n_Texp)


def test_optional_quantum_states_support_deferred_activation(pyexocross, raw_dataset):
    path = raw_dataset()
    options = dict(optional_quantum_states=True, activation=False, elower_max=50000.0)
    actual = _load(path, **options)
    reference = _load(path, backend="radis", **options)
    assert {"v_l", "v_u", "kp_l", "kp_u"}.issubset(actual.df.columns)
    assert not actual.activation
    for mdb in (actual, reference):
        quantum_mask = (mdb.df["v_u"] - mdb.df["v_l"]) == 3
        selected = mdb.df[quantum_mask & mdb.df_load_mask]
        assert 0 < len(selected) < len(mdb.df)
        expected_centers = selected.nu_lines.to_numpy()
        mdb.activate(mdb.df, quantum_mask)
        np.testing.assert_array_equal(mdb.nu_lines, expected_centers)
    _assert_same_lines(actual, reference)


def test_json_definition_preserves_source_broadening_defaults(pyexocross, raw_dataset):
    path = raw_dataset()
    definition = path / "12C-16O__SAMPLE.def"
    definition.unlink()
    definition.with_suffix(".def.json").write_text(json.dumps({
        "isotopologue": {"mass_in_Da": 28.0101},
        "broad": {
            "default_Lorentzian_half-width": 0.123,
            "default_temperature_exponent": 0.456,
        },
        "dataset": {
            "states": {"states_file_fields": [
                {"name": name} for name in ("ID", "E", "gtot", "J", "v", "kp")
            ]},
            "transitions": {"number_of_transition_files": 1, "max_wavenumber": 22000.0},
        },
    }))
    mdb = _load(path, broadf=False, optional_quantum_states=True)
    assert len(mdb.nu_lines) == 220
    assert mdb.molmass == 28.0101
    assert {"v_l", "v_u", "kp_l", "kp_u"}.issubset(mdb.df.columns)
    np.testing.assert_array_equal(mdb.alpha_ref, np.full(220, 0.123))
    np.testing.assert_array_equal(mdb.n_Texp, np.full(220, 0.456))


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_device_arrays_and_snapshot_follow_public_mask(pyexocross, raw_dataset, gpu_transfer):
    mdb = _load(raw_dataset(), gpu_transfer=gpu_transfer)
    fields = LINE_FIELDS + (("dev_nu_lines",) if gpu_transfer else ())
    assert hasattr(mdb, "dev_nu_lines") == gpu_transfer
    for cutoff in (5000.0, 2000.0):
        mask = mdb.elower < cutoff
        assert 0 < np.count_nonzero(mask) < len(mask)
        expected = {name: np.asarray(getattr(mdb, name))[mask] for name in fields}
        mdb.apply_mask_mdb(mask)
        for name, values in expected.items():
            np.testing.assert_array_equal(getattr(mdb, name), values, err_msg=name)
        snapshot = mdb.to_snapshot()
        np.testing.assert_array_equal(snapshot.lines.nu_lines, expected["nu_lines"])
        np.testing.assert_array_equal(snapshot.alpha_ref, expected["alpha_ref"])
        np.testing.assert_array_equal(snapshot.n_Texp, expected["n_Texp"])


def test_none_range_keeps_unsegmented_data_inactive(pyexocross, raw_dataset):
    with pytest.warns(UserWarning, match="Nonactive"):
        mdb = _load(raw_dataset(), nurange=None)
    assert not mdb.activation
    assert len(mdb.df) > 0
    assert not np.any(mdb.df_load_mask)
    with pytest.raises(ValueError, match="No line found"):
        mdb.activate(mdb.df)


@pytest.mark.parametrize("mode", ["three_columns", "late_missing_nu"])
def test_missing_transition_wavenumber_matches_radis(pyexocross, raw_dataset, mode):
    path = raw_dataset()
    transition_file = next(path.glob("*.trans.bz2"))
    with bz2.open(transition_file, "rt") as stream:
        rows = [line.split() for line in stream]
    if mode == "three_columns":
        rows = [row[:3] for row in rows]
    else:
        assert float(rows[-1][3]) > NURANGE[1]
        rows[-1][3] = "nan"
    with bz2.open(transition_file, "wt") as stream:
        stream.writelines(" ".join(row) + "\n" for row in rows)

    actual = _load(path)
    reference = _load(path, backend="radis")
    assert len(actual.nu_lines) == 220
    _assert_same_lines(actual, reference)
    if mode == "late_missing_nu":
        from exojax.database.exomol._pyexocross import load_exomol_data

        # The missing value occurs outside the selection, after several chunks.
        # Its source-wide fallback must also affect already-read transitions.
        chunked, _ = load_exomol_data(path, NURANGE, chunk_size=37)
        chunked_order = np.argsort(chunked.nu_lines.to_numpy(), kind="stable")
        actual_order = np.argsort(actual.nu_lines, kind="stable")
        np.testing.assert_array_equal(
            chunked.nu_lines.to_numpy()[chunked_order], actual.nu_lines[actual_order],
        )
        np.testing.assert_allclose(
            chunked.Sij0.to_numpy()[chunked_order],
            actual.line_strength_ref_original[actual_order], rtol=2e-12, atol=0,
        )


@pytest.mark.parametrize("unsupported_broadening", [False, True])
def test_loading_does_not_change_pyexocross_global_configuration(
    pyexocross, raw_dataset, monkeypatch, unsupported_broadening,
):
    from pyexocross.base.config_manager import ConfigManager

    path = raw_dataset()
    if unsupported_broadening:
        (path / "12C-16O__H2.broad").write_text("m0 0.1 0.5 1\n")
    sentinel = object()
    monkeypatch.setattr(ConfigManager, "_last_config", sentinel)
    namespace = vars(sys.modules["__main__"])
    before = namespace.copy()
    if unsupported_broadening:
        with pytest.raises((ValueError, NotImplementedError), match="(?i)broadening|a0|a1"):
            _load(path)
    else:
        assert len(_load(path).nu_lines) == 220
    assert ConfigManager._last_config is sentinel
    assert namespace.keys() == before.keys()
    for name, value in before.items():
        assert namespace[name] is value, name
