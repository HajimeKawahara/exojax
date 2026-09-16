"""ExoAtom regression tests using small, independently sourced atomic records."""

import builtins
import bz2
import json
from pathlib import Path
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import atomic_mass, c, k
from scipy.special import voigt_profile

from exojax.database import AdbExoAtom
from exojax.opacity import OpaDirect
from exojax.utils.constants import ccgs, hcperk


DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def isolated(isolated_test_environment):
    """Enable double precision and keep writes outside the source fixtures."""


@pytest.fixture
def pyexocross():
    module = pytest.importorskip("pyexocross")
    if module.__version__ != "1.1.16":
        pytest.skip("The optional atomic reader supports PyExoCross 1.1.16.")
    return module


@pytest.fixture
def dataset(tmp_path):
    def copy(relative="Li/NIST"):
        target = tmp_path / relative
        shutil.copytree(DATA / relative, target)
        return target
    return copy


def _load(path, **kwargs):
    return AdbExoAtom(path, download=False, **kwargs)


def _raw(path):
    stem = f"{path.parent.name}__{path.name}"
    states = np.loadtxt(path / f"{stem}.states", usecols=(0, 1, 2, 3), ndmin=2)
    transitions = np.loadtxt(path / f"{stem}.trans", ndmin=2)
    return stem, {int(row[0]): row for row in states}, transitions


def _strength(T, A, gu, nu, elower, q):
    """Direct LTE Einstein formula, independent of the reference-strength path."""
    return (
        A * gu * np.exp(-hcperk * elower / T)
        * -np.expm1(-hcperk * nu / T)
        / (8 * np.pi * ccgs * nu**2 * q)
    )


@pytest.mark.parametrize("relative,species,mass,count", [
    ("Li/NIST", "Li_I", 6.941, 4),
    ("Li/Kurucz", "Li_I", 6.94, 4),
    ("Li_p/NIST", "Li_II", 6.941, 3),
    ("H/1H/NIST", "H_I", 1.00784, 3),
])
def test_real_sources_preserve_line_records_and_species(pyexocross, dataset, relative, species, mass, count):
    path = dataset(relative)
    adb = _load(path)
    _, states, lines = _raw(path)
    lines = lines[np.argsort(lines[:, 3], kind="stable")]
    assert adb.dbtype == "exoatom"
    assert adb.species == species
    assert len(adb.nu_lines) == count
    np.testing.assert_array_equal(adb.nu_lines, lines[:, 3])
    np.testing.assert_array_equal(adb.A, lines[:, 2])
    np.testing.assert_array_equal(adb.elower, [states[int(row[1])][1] for row in lines])
    np.testing.assert_array_equal(adb.eupper, [states[int(row[0])][1] for row in lines])
    np.testing.assert_array_equal(adb.glower, [states[int(row[1])][2] for row in lines])
    np.testing.assert_array_equal(adb.gupper, [states[int(row[0])][2] for row in lines])
    np.testing.assert_array_equal(adb.jlower, [states[int(row[1])][3] for row in lines])
    np.testing.assert_array_equal(adb.jupper, [states[int(row[0])][3] for row in lines])
    np.testing.assert_array_equal(adb.line_masses, np.full(count, mass))
    np.testing.assert_array_equal(adb.iion, np.full(count, 2 if species == "Li_II" else 1))
    assert np.isfinite(adb.logsij0).all()


def test_nist_optional_metadata_does_not_shift_mandatory_columns(pyexocross, dataset):
    path = dataset()
    metadata = json.loads((path / "Li__NIST.adef.json").read_text())
    assert metadata["dataset"]["states"]["lande_g_available"]
    # The real source declares a g-factor field but omits it from actual rows.
    assert len((path / "Li__NIST.states").read_text().splitlines()[0].split()) == 8
    adb = _load(path, nurange=[14900, 14910])
    np.testing.assert_array_equal(adb.jupper, [0.5, 1.5])
    np.testing.assert_array_equal(adb.nu_lines, [14903.66, 14903.99])
    assert adb.nu_lines[1] != float(adb.eupper[1] - adb.elower[1])
    assert np.isnan(adb.gamma_natural).all()


@pytest.mark.parametrize("relative", ["Li/NIST", "Li/Kurucz", "Li_p/NIST"])
def test_partition_and_runtime_strength_match_direct_lte(pyexocross, dataset, relative):
    path = dataset(relative)
    adb = _load(path)
    stem = f"{path.parent.name}__{path.name}"
    partition = np.loadtxt(path / f"{stem}.pf")
    np.testing.assert_array_equal(adb.T_gQT, partition[:, 0])
    np.testing.assert_array_equal(adb.gQT, partition[:, 1])
    for temperature in (296.0, 3000.0, 5000.0):
        q = np.interp(temperature, partition[:, 0], partition[:, 1])
        np.testing.assert_allclose(adb.QT_interp(temperature), q, rtol=1e-13)
        expected = _strength(
            temperature, np.asarray(adb.A), np.asarray(adb.gupper), adb.nu_lines,
            np.asarray(adb.elower), q,
        )
        # The shared temperature-scaling formula loses a few digits for the
        # Kurucz 0.01 cm-1 line through its existing 1 - exp(-x) expression.
        np.testing.assert_allclose(adb.line_strength(temperature), expected, rtol=3e-11, atol=0)
        np.testing.assert_allclose(
            adb.qr_interp_lines(temperature, adb.Tref),
            q / np.interp(adb.Tref, partition[:, 0], partition[:, 1]), rtol=1e-13,
        )


def test_excited_ion_lines_survive_reference_underflow(pyexocross, dataset):
    adb = _load(dataset("Li_p/NIST"))
    underflow = np.asarray(adb.Sij0) == 0
    assert underflow.any()
    assert np.isfinite(np.asarray(adb.logsij0)[underflow]).all()
    assert np.all(np.asarray(adb.line_strength(5000.0))[underflow] > 0)
    adb.apply_mask_mdb(underflow)
    assert len(adb.nu_lines) == np.count_nonzero(underflow)
    assert np.isfinite(adb.logsij0).all()
    assert np.all(np.asarray(adb.line_strength(5000.0)) > 0)


def test_natural_width_uses_both_complete_level_lifetimes(pyexocross, dataset):
    path = dataset("Li/Kurucz")
    adb = _load(path)
    states = np.loadtxt(path / "Li__Kurucz.states", usecols=(0, 5))
    rates = {int(identifier): 1 / lifetime for identifier, lifetime in states}
    lines = np.loadtxt(path / "Li__Kurucz.trans")
    lines = lines[np.argsort(lines[:, 3], kind="stable")]
    expected = [(rates[int(upper)] + rates[int(lower)]) / (4 * np.pi * ccgs)
                for upper, lower, _, _ in lines]
    np.testing.assert_allclose(adb.gamma_natural, expected, rtol=1e-13)
    assert rates[1] == 0.0
    assert np.all(np.isfinite(adb.gamma_natural))
    assert not np.allclose(adb.gamma_natural, np.asarray(adb.A) / (4 * np.pi * ccgs), rtol=1e-4, atol=0)


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_masking_keeps_host_and_device_records_aligned(pyexocross, dataset, gpu_transfer):
    adb = _load(dataset("Li/Kurucz"), gpu_transfer=gpu_transfer)
    fields = ("A", "elower", "eupper", "glower", "gupper", "jlower", "jupper",
              "ielem", "iion", "logsij0", "gamma_natural")
    host = {name: np.asarray(getattr(adb, "_" + name)).copy() for name in fields}
    mask = np.array([False, True, False, True])
    adb.masking(mask)
    adb.generate_jnp_arrays()
    for name in fields:
        np.testing.assert_array_equal(getattr(adb, "_" + name), host[name][mask])
        np.testing.assert_array_equal(getattr(adb, name), host[name][mask])
    np.testing.assert_array_equal(adb.dev_nu_lines, adb.nu_lines)
    np.testing.assert_array_equal(adb.line_masses, np.full(2, 6.94))
    adb.masking(np.array([True, False]))
    assert np.shape(adb.A) == (1,)
    with pytest.raises(ValueError):
        adb.masking(np.array([True, False]))


def test_range_margin_energy_and_strength_selection(pyexocross, dataset):
    path = dataset()
    full = _load(path)
    adb = _load(path, nurange=[14903.7, 14903.9], margin=0.1, elower_max=1)
    np.testing.assert_array_equal(adb.nu_lines, [14903.66, 14903.99])
    cutoff = np.sqrt(np.sort(full.Sij0)[1] * np.sort(full.Sij0)[2])
    cut = _load(path, crit=cutoff)
    np.testing.assert_array_equal(cut.nu_lines, full.nu_lines[full.Sij0 > cutoff])


def test_natural_width_lpf_matches_scipy_voigt(pyexocross, dataset):
    adb = _load(dataset("Li/Kurucz"), nurange=[14900, 14910])
    temperature = 3000.0
    grid = np.linspace(14903.0, 14904.8, 181)
    opa = OpaDirect(adb, grid)
    strengths = _strength(
        temperature, np.asarray(adb.A), np.asarray(adb.gupper), adb.nu_lines,
        np.asarray(adb.elower), float(adb.QT_interp(temperature)),
    )
    sigma = adb.nu_lines * np.sqrt(k * temperature / (np.asarray(adb.line_masses) * atomic_mass)) / c
    expected = np.sum(
        strengths[:, None] * voigt_profile(
            grid[None, :] - adb.nu_lines[:, None], sigma[:, None],
            np.asarray(adb.gamma_natural)[:, None],
        ), axis=0,
    )
    actual = jax.jit(opa.xsvector)(temperature, 0.01)
    # ExoJAX's Doppler coefficient is rounded relative to current SI constants.
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0)
    np.testing.assert_allclose(actual, opa.xsvector(temperature, 10.0), rtol=1e-13)
    derivative = jax.grad(lambda T: jnp.sum(opa.xsvector(T, 0.01)))(temperature)
    assert np.isfinite(derivative) and derivative != 0


def test_compressed_files_and_relative_cache_root(pyexocross, dataset, tmp_path):
    path = dataset()
    expected = _load(path)
    for extension in ("states", "trans"):
        source = path / f"Li__NIST.{extension}"
        source.with_suffix(source.suffix + ".bz2").write_bytes(bz2.compress(source.read_bytes()))
        source.unlink()
    actual = AdbExoAtom("Li/NIST", local_databases=tmp_path, download=False)
    np.testing.assert_array_equal(actual.nu_lines, expected.nu_lines)
    np.testing.assert_array_equal(actual.logsij0, expected.logsij0)


def test_unresolved_hydrogen_j_and_isotope_mass_remain_usable(pyexocross, dataset):
    path = dataset("H/1H/NIST")
    definition_path = path / "1H__NIST.adef.json"
    metadata = json.loads(definition_path.read_text())
    metadata["species"]["mass_in_Da"] = 1.00794
    definition_path.write_text(json.dumps(metadata))
    adb = _load(path, nurange=[82258, 82260])
    assert np.isnan(adb.jupper).any()
    np.testing.assert_array_equal(adb.gupper, [2, 8])
    np.testing.assert_array_equal(adb.line_masses, [1.00784, 1.00784])
    grid = np.linspace(82258, 82260, 21)
    opa = OpaDirect(adb, grid, atomic_broadening=lambda T, P: jnp.full(2, 0.01))
    assert np.all(np.isfinite(opa.xsvector(5000.0, 0.1)))
    assert np.all(np.asarray(opa.xsvector(5000.0, 0.1)) > 0)


@pytest.mark.parametrize("relative", ["Li/NIST", "Li_p/NIST", "H/1H/NIST"])
def test_missing_files_download_exact_dataset_and_are_reused(pyexocross, tmp_path, monkeypatch, relative):
    from exojax.database.exoatom import _files

    requested = []

    def fetch(url, target):
        requested.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((DATA / relative / target.name).read_bytes())

    monkeypatch.setattr(_files, "_download_file", fetch)
    adb = AdbExoAtom(relative, local_databases=tmp_path)
    filenames = [Path(path).name for path in adb.local_paths]
    assert set(requested) == {
        f"https://www.exomol.com/exoatom/db/{relative}/{name}" for name in filenames
    }
    assert len(requested) == 4
    AdbExoAtom(relative, local_databases=tmp_path)
    assert len(requested) == 4


def test_download_false_reports_missing_raw_file(pyexocross, tmp_path, monkeypatch):
    from exojax.database.exoatom import _files

    def unexpected(*args, **kwargs):
        pytest.fail("download=False attempted a network request")

    monkeypatch.setattr(_files, "_download_file", unexpected)
    with pytest.raises(FileNotFoundError, match="adef.json"):
        AdbExoAtom("Li/NIST", local_databases=tmp_path, download=False)
    assert not list(tmp_path.iterdir())


def test_partition_domain_and_reference_temperature_are_explicit(pyexocross, dataset):
    path = dataset("Li/Kurucz")
    adb = _load(path)
    assert np.isnan(adb.QT_interp(99.0))
    assert np.isnan(adb.QT_interp(208931.0))
    with pytest.raises(ValueError, match="Tref"):
        _load(path, Tref=50)


@pytest.mark.parametrize("kind", ["mass", "charge", "partition"])
def test_invalid_physical_metadata_is_rejected(pyexocross, dataset, kind):
    path = dataset()
    definition_path = path / "Li__NIST.adef.json"
    metadata = json.loads(definition_path.read_text())
    if kind == "mass":
        metadata["species"]["mass_in_Da"] = 0
    elif kind == "charge":
        metadata["species"]["charge"] = -1
    else:
        (path / "Li__NIST.pf").write_text("296 2\n100 0\n")
    definition_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        _load(path)


def test_missing_state_reference_fails_clearly(pyexocross, dataset):
    path = dataset()
    transition_path = path / "Li__NIST.trans"
    transition_path.write_text("99999 1 1.0 14903.66\n")
    with pytest.raises(ValueError, match="(?i)state"):
        _load(path)


def test_missing_optional_dependency_has_installation_hint(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyexocross" or name.startswith("pyexocross."):
            raise ModuleNotFoundError("No module named 'pyexocross'", name="pyexocross")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setitem(sys.modules, "pyexocross", None)
    with pytest.raises(ImportError, match="(?i)pyexocross") as error:
        AdbExoAtom(tmp_path / "Li/NIST", download=False)
    assert "pip install" in str(error.value)
    assert not list(tmp_path.iterdir())
