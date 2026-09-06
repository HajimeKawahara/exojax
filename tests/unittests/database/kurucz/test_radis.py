"""Exercise the RADIS reader, cache, and ExoJAX activation without networking."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pandas as pd
import pytest

from exojax.database.core_atom.io import read_kurucz
from exojax.database.kurucz import api
from exojax.database.kurucz._radis import transitions_from_dataframe
from exojax.opacity import OpaDirect


def _line(wavelength, lower, upper, jl, ju, species="26.00"):
    row = [" "] * 160
    fields = (
        (0, 11, f"{wavelength:11.4f}"), (11, 18, f"{-1.0:7.3f}"),
        (18, 24, f"{species:>6}"), (24, 36, f"{lower:12.3f}"),
        (36, 41, f"{jl:5.1f}"), (42, 52, "lower     "),
        (52, 64, f"{upper:12.3f}"), (64, 69, f"{ju:5.1f}"),
        (70, 80, "upper     "), (80, 86, "  8.00"),
        (86, 92, " -6.00"), (92, 98, " -7.00"), (106, 109, " 56"),
    )
    for start, stop, value in fields:
        row[start:stop] = value
    return "".join(row) + "\n"


@pytest.fixture
def offline_kurucz(monkeypatch, tmp_path, request):
    """Keep the real RADIS parser/cache while replacing transport and registry."""
    from radis.api import dbmanager
    from radis.api.kuruczapi import KuruczDatabaseManager

    fetch_module = importlib.import_module("radis.io.kurucz")
    species = getattr(request, "param", "Fe_I")
    code = {"Fe_I": "26.00", "Fe_II": "26.01"}[species]
    source = tmp_path / "gf2600.all"
    source.write_text(
        _line(500.0, 100.0, 20100.0, 0.5, 1.5, code)
        + _line(600.0, 22000.0, 5000.0, 2.5, 1.5, code)
        + _line(550.0, 0.0, 18181.0, 0.5, 1.5, code)
    )
    state = SimpleNamespace(source=source, downloads=0, registry={}, species=species,
                            local_databases=str(tmp_path / "cache"))

    def download(manager, urls, targets, total=None):
        state.downloads += 1
        opener = SimpleNamespace(abspath=lambda url: str(source))
        return manager.parse_to_local_file(opener, urls[0], targets[0], pbar_active=False)

    def register(manager, get_main_files):
        state.registry[manager.name] = {
            "path": [manager.actual_file], "download_url": [manager.actual_url],
        }

    monkeypatch.setattr(dbmanager.DatabaseManager, "is_registered",
                        lambda manager: manager.name in state.registry)
    monkeypatch.setattr(dbmanager, "getDatabankEntries", state.registry.__getitem__)
    monkeypatch.setattr(fetch_module, "getDatabankEntries", state.registry.__getitem__)
    monkeypatch.setattr(KuruczDatabaseManager, "download_and_parse", download)
    monkeypatch.setattr(KuruczDatabaseManager, "register", register)
    state.create = lambda **kwargs: api.AdbKurucz.from_radis(
        species, [17000.0, 19000.0], local_databases=state.local_databases, **kwargs
    )
    return state


@pytest.mark.parametrize("offline_kurucz,irwin", [
    ("Fe_I", False), ("Fe_I", True), ("Fe_II", False),
], indirect=["offline_kurucz"])
def test_radis_parser_cache_selection_and_opacity(offline_kurucz, irwin):
    from radis.api.kuruczapi import read_kurucz as radis_read_kurucz

    original = radis_read_kurucz(offline_kurucz.source).sort_values("wav")
    adb = offline_kurucz.create(margin=2000.0, Irwin=irwin, gpu_transfer=False)
    assert isinstance(adb, api.AdbKurucz)
    assert isinstance(adb.nu_lines, np.ndarray)
    assert not hasattr(adb, "dev_nu_lines")
    np.testing.assert_array_equal(adb.nu_lines, original.wav)
    np.testing.assert_array_equal(adb._jlower, original.jl)
    np.testing.assert_array_equal(adb._jupper, original.ju)
    np.testing.assert_array_equal(adb._A, original.A)
    np.testing.assert_array_equal(adb._ielem, [26, 26, 26])
    iion = 1 if offline_kurucz.species == "Fe_I" else 2
    np.testing.assert_array_equal(adb._iion, [iion, iion, iion])
    assert adb.provenance["backend"] == "radis"
    assert adb.provenance["species"] == offline_kurucz.species
    assert adb.provenance["radis_version"]
    assert all(Path(path).is_file() for path in adb.provenance["local_paths"])

    offline_kurucz.source.unlink()
    cached = offline_kurucz.create(Irwin=irwin)
    assert offline_kurucz.downloads == 1
    assert len(cached.nu_lines) == 1
    assert cached.provenance == adb.provenance
    mask = (adb.nu_lines > 17000.0) & (adb.nu_lines < 19000.0)
    adb.masking(mask)
    assert isinstance(adb.atomicmass, np.ndarray)
    adb.generate_jnp_arrays()
    assert isinstance(adb.jlower, jax.Array)
    np.testing.assert_array_equal(adb.jlower, [0.5])
    np.testing.assert_array_equal(adb.Sij0, cached.Sij0)
    np.testing.assert_array_equal(adb.qr_interp_lines(1800.0, adb.Tref),
                                  cached.qr_interp_lines(1800.0, cached.Tref))
    grid = np.linspace(18170.0, 18190.0, 33)
    actual = OpaDirect(adb, grid).xsvector(1800.0, 0.1)
    expected = OpaDirect(cached, grid).xsvector(1800.0, 0.1)
    assert np.all(np.isfinite(actual)) and np.all(actual > 0)
    np.testing.assert_array_equal(actual, expected)

    with pytest.warns(UserWarning, match="no lines are selected"):
        empty = api.AdbKurucz.from_radis(
            offline_kurucz.species, [1000.0, 2000.0],
            local_databases=offline_kurucz.local_databases, gpu_transfer=False,
        )
    assert len(empty.nu_lines) == len(empty.atomicmass) == len(empty.ionE) == 0
    assert offline_kurucz.downloads == 1


def test_local_constructor_preserves_reader_values(offline_kurucz):
    expected = read_kurucz(offline_kurucz.source)
    adb = api.AdbKurucz(offline_kurucz.source, gpu_transfer=False)
    fields = ("_A", "nu_lines", "_elower", "_eupper", "_gupper", "_jlower",
              "_jupper", "_ielem", "_iion", "_gamRad", "_gamSta", "_vdWdamp")
    for field, values in zip(fields, expected):
        np.testing.assert_array_equal(getattr(adb, field), values)
    assert offline_kurucz.downloads == 0


@pytest.mark.parametrize("species,kwargs,message", [
    ("Fe", {}, "such as"), ("Fe_IV", {}, "Unsupported"),
    ("Cs_I", {}, "atomic metadata"), ("Fe_I", {"nurange": [0.0, 1.0]}, "positive"),
    ("Fe_I", {"nurange": [1.0, np.inf]}, "finite"),
    ("Fe_I", {"margin": -1.0}, "margin"),
    ("Fe_I", {"crit": np.nan}, "crit"),
    ("Fe_I", {"vmr_fraction": [0.0, np.nan, 1.0]}, "vmr_fraction"),
])
def test_invalid_requests_fail_before_fetch(monkeypatch, species, kwargs, message):
    def unexpected_fetch(*args, **kwargs):
        pytest.fail("Invalid requests must be rejected before RADIS fetch.")

    monkeypatch.setattr(api, "fetch_kurucz_dataframe", unexpected_fetch)
    options = {"nurange": [1000.0, 2000.0], **kwargs}
    with pytest.raises(ValueError, match=message):
        api.AdbKurucz.from_radis(species, **options)


@pytest.mark.parametrize("column,value,message", [
    ("A", 0.0, "positive"), ("wav", np.inf, "nonfinite"),
    ("gamvdW", np.nan, "missing or nonfinite"),
    ("species", "26.01", "requested species"),
])
def test_converter_rejects_invalid_source_data(column, value, message):
    frame = pd.DataFrame({
        "A": [1.0e6], "wav": [20000.0], "El": [0.0], "Eu": [20000.0],
        "gu": [4.0], "jl": [0.5], "ju": [1.5], "gamRad": [8.0],
        "gamSta": [-6.0], "gamvdW": [-7.0], "species": ["26.00"],
    })
    frame[column] = value
    with pytest.raises(ValueError, match=message):
        transitions_from_dataframe(frame, 26, 1)
