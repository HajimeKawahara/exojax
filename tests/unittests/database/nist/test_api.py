"""NIST adapters and explicit atomic broadening, without network access."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from exojax.database import AdbNist
from exojax.database.nist import api
from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector


@pytest.fixture
def make_adb(monkeypatch):
    frame = pd.DataFrame({
        "wav": [1002.0, 1000.0, 1001.0], "A": [3e6, 1e6, 2e6],
        "El": [20.0, 0.0, 10.0], "Eu": [1022.0, 1000.0, 1011.0],
        "gl": [6.0, 2.0, 4.0], "gu": [8.0, 4.0, 6.0],
        "jl": [2.5, 0.5, 1.5], "ju": [3.5, 1.5, 2.5],
        "species": ["Fe_II"] * 3,
    })
    calls = []

    def fetch(species, **kwargs):
        calls.append((species, kwargs))
        return frame.copy(), ["offline-nist.h5"]

    monkeypatch.setattr(api, "fetch_nist_lines", fetch)
    return SimpleNamespace(
        create=lambda **kwargs: AdbNist("Fe II", **kwargs), frame=frame, calls=calls
    )


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_nist_sorted_arrays_fractional_j_and_mask_alignment(make_adb, gpu_transfer):
    adb = make_adb.create(gpu_transfer=gpu_transfer)
    assert adb.species == "Fe_II"
    assert adb.local_paths == ["offline-nist.h5"]
    np.testing.assert_array_equal(adb.nu_lines, [1000.0, 1001.0, 1002.0])
    np.testing.assert_array_equal(adb._A, [1e6, 2e6, 3e6])
    np.testing.assert_array_equal(adb._jlower, [0.5, 1.5, 2.5])
    assert np.all(np.isfinite(adb.Sij0)) and np.all(adb.Sij0 > 0)
    assert not any(hasattr(adb, name) for name in ("gamRad", "gamSta", "vdWdamp", "ionE"))
    assert hasattr(adb, "logsij0") == gpu_transfer
    old_ratios = np.asarray(adb.qr_interp_lines(1800.0, adb.Tref))
    np.testing.assert_array_equal(adb.qr_interp_lines(adb.Tref, adb.Tref), np.ones(3))
    adb.masking(np.array([False, True, True]))
    adb.generate_jnp_arrays()
    np.testing.assert_array_equal(adb.A, [2e6, 3e6])
    np.testing.assert_array_equal(adb.jlower, [1.5, 2.5])
    np.testing.assert_array_equal(adb.jupper, [2.5, 3.5])
    np.testing.assert_array_equal(adb.iion, [2, 2])
    np.testing.assert_allclose(adb.qr_interp_lines(1800.0, adb.Tref), old_ratios[1:])
    np.testing.assert_array_equal(adb.line_masses, np.full(2, 55.847))
    adb.masking(np.array([False, True]))
    np.testing.assert_array_equal(adb.dev_nu_lines, adb.nu_lines)
    np.testing.assert_array_equal(adb.elower, adb._elower)
    np.testing.assert_allclose(adb.logsij0, np.log(adb.Sij0))


def test_nist_range_margin_and_strength_cutoff(make_adb, tmp_path):
    full = make_adb.create()
    cutoff = np.min(full.Sij0)
    adb = make_adb.create(
        nurange=[1000.0, 1001.0], margin=0.5, crit=cutoff,
        local_databases=tmp_path, engine="pytables", cache="force",
    )
    expected = (full.nu_lines > 999.5) & (full.nu_lines < 1001.5) & (full.Sij0 > cutoff)
    np.testing.assert_array_equal(adb.nu_lines, full.nu_lines[expected])
    assert make_adb.calls[-1] == ("Fe_II", dict(
        nurange=[999.5, 1001.5], local_databases=tmp_path, engine="pytables", cache="force",
        databank_name="ExoJAX-NIST-{molecule}",
    ))


@pytest.mark.parametrize("column,value", [
    ("A", np.nan), ("A", 0.0), ("wav", -1.0), ("gu", 0.0),
    ("El", np.inf), ("El", -1.0), ("jl", -0.5), ("Eu", 0.0),
])
def test_nist_discards_invalid_line_parameters(make_adb, column, value):
    make_adb.frame.loc[0, column] = value
    with pytest.warns(UserWarning, match="Discarded 1 NIST lines"):
        adb = make_adb.create()
    np.testing.assert_array_equal(adb.nu_lines, [1000.0, 1001.0])


def test_nist_missing_columns_and_empty_selection(make_adb):
    make_adb.frame.drop(columns="A", inplace=True)
    with pytest.raises(ValueError, match="missing required columns: A"):
        make_adb.create()
    make_adb.frame["A"] = 1e6
    with pytest.warns(UserWarning, match="No NIST lines"):
        adb = make_adb.create(nurange=[1100.0, 1200.0])
    assert adb.nu_lines.size == 0


@pytest.mark.parametrize("species", ["Fe", "CO_I", "Fe_IIII", "Xx_I", "H_III", 26])
def test_nist_rejects_invalid_species_before_fetch(make_adb, species):
    with pytest.raises(ValueError, match="species|ion stage"):
        AdbNist(species)
    assert not make_adb.calls


def test_nist_rejects_missing_partition_function_before_fetch(make_adb):
    with pytest.raises(ValueError, match="supported partition functions"):
        AdbNist("Fe_IV")
    assert not make_adb.calls


def test_nist_missing_partition_table_entry(make_adb, monkeypatch):
    temperatures, partition = api.load_pf_Barklem2016()
    partition = partition.loc[partition["T[K]"] != "Fe_II"]
    monkeypatch.setattr(api, "load_pf_Barklem2016", lambda: (temperatures, partition))
    with pytest.raises(ValueError, match="No Barklem partition function.*Fe_II"):
        make_adb.create()
    assert not make_adb.calls


def test_nist_rejects_mismatched_cache_species(make_adb):
    make_adb.frame.loc[0, "species"] = "Fe_I"
    with pytest.raises(ValueError, match="other than the requested 'Fe_II'"):
        make_adb.create()


@pytest.mark.parametrize("kwargs", [
    {"nurange": []}, {"nurange": [np.nan, 1000]}, {"nurange": [1000, 1000]},
    {"margin": -1}, {"crit": np.inf},
])
def test_nist_rejects_invalid_selection(make_adb, kwargs):
    with pytest.raises(ValueError):
        make_adb.create(**kwargs)
    assert not make_adb.calls


def test_nist_irwin_reference_and_runtime_partition_are_consistent(make_adb):
    from exojax.database.core_atom.pf import partfn_Fe
    from exojax.utils.constants import ccgs, hcperk

    make_adb.frame["species"] = "Fe_I"
    adb = AdbNist("Fe_I", Irwin=True)
    temperature = 1800.0
    expected = (
        -np.asarray(adb.A) * np.asarray(adb.gupper)
        * np.exp(-hcperk * np.asarray(adb.elower) / temperature)
        * np.expm1(-hcperk * adb.nu_lines / temperature)
        / (8 * np.pi * ccgs * adb.nu_lines**2 * float(partfn_Fe(temperature)))
    )
    actual = line_strength(
        temperature, adb.logsij0, adb.nu_lines, adb.elower,
        adb.qr_interp_lines(temperature, adb.Tref), adb.Tref,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0.0)


def test_nist_direct_requires_broadening_and_checks_callback_shape(make_adb):
    adb = make_adb.create()
    grid = np.linspace(999.8, 1002.2, 61)
    with pytest.raises(ValueError, match="NIST requires an explicit atomic_broadening"):
        OpaDirect(adb, grid)
    with pytest.raises(TypeError, match="must be callable"):
        OpaDirect(adb, grid, atomic_broadening=0.1)
    with pytest.raises(ValueError, match="only NIST, VALD, and Kurucz"):
        OpaDirect(SimpleNamespace(dbtype="exomol"), grid, atomic_broadening=lambda T, P: 0.1)
    for width in (0.1, jnp.ones(2), jnp.ones((1, 3))):
        opa = OpaDirect(adb, grid, atomic_broadening=lambda T, P: width)
        with pytest.raises(ValueError, match="atomic_broadening must return shape"):
            jax.jit(opa.xsvector)(1800.0, 0.1)


def test_nist_direct_values_and_gradients_with_explicit_broadening(make_adb):
    adb = make_adb.create()
    grid = np.linspace(999.8, 1002.2, 61)

    def broadening(T, P):
        return jnp.array([0.1, 0.2, 0.3]) * P * (296.0 / T)**0.7 + 0.002

    opa = OpaDirect(adb, grid, atomic_broadening=broadening)
    temperatures, pressures = jnp.array([1300.0, 1800.0]), jnp.array([0.01, 0.1])
    matrix = jax.jit(opa.xsmatrix)(temperatures, pressures)
    vectors = jnp.stack([opa.xsvector(T, P) for T, P in zip(temperatures, pressures)])
    np.testing.assert_allclose(matrix, vectors, rtol=1e-12, atol=0.0)
    assert np.all(np.isfinite(matrix)) and np.all(matrix > 0)
    strengths = line_strength(
        1800.0, adb.logsij0, adb.nu_lines, adb.elower,
        adb.qr_interp_lines(1800.0, adb.Tref), adb.Tref,
    )
    expected = xsvector(
        opa.opainfo, doppler_sigma(adb.nu_lines, 1800.0, adb.line_masses),
        broadening(1800.0, 0.1), strengths,
    )
    np.testing.assert_allclose(matrix[1], expected, rtol=1e-12, atol=0.0)
    index = np.argmin(np.abs(grid - 1000.0))
    fn = jax.jit(lambda T, P: jnp.log(opa.xsvector(T, P)[index]))
    actual = jax.jit(jax.grad(fn, argnums=(0, 1)))(1800.0, 0.1)
    matrix_fn = lambda T, P: jnp.log(opa.xsmatrix(
        jnp.array([T, 1300.0]), jnp.array([P, 0.01])
    )[0, index])
    matrix_gradient = jax.jit(jax.grad(matrix_fn, argnums=(0, 1)))(1800.0, 0.1)
    expected_gradient = (
        (fn(1800.1, 0.1) - fn(1799.9, 0.1)) / 0.2,
        (fn(1800.0, 0.10001) - fn(1800.0, 0.09999)) / 0.00002,
    )
    assert np.all(np.isfinite(actual)) and np.all(np.abs(actual) > 1e-8)
    np.testing.assert_allclose(actual, expected_gradient, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(matrix_gradient, actual, rtol=1e-12, atol=0.0)
    assert opa == OpaDirect(adb, grid, atomic_broadening=broadening)
    assert opa != OpaDirect(adb, grid, atomic_broadening=lambda T, P: broadening(T, P))


def test_nist_direct_accepts_explicit_doppler_only_model(make_adb):
    adb = make_adb.create()
    opa = OpaDirect(
        adb, np.linspace(999.8, 1002.2, 61),
        atomic_broadening=lambda T, P: jnp.zeros_like(adb.A),
    )
    opacity = jax.jit(opa.xsvector)(1800.0, 0.1)
    assert np.all(np.isfinite(opacity)) and np.all(opacity >= 0)
    assert np.any(opacity > 0)


def test_nist_radis_parser_fetch_and_cache_roundtrip(monkeypatch, tmp_path):
    radis_api = pytest.importorskip("radis.api.nistapi")
    from radis.api.dbmanager import DatabaseManager

    source = tmp_path / "nist_OI.tsv"
    source.write_text(
        "ritz_wl_air(nm)\tAki(s^-1)\tAcc\tEi(cm-1)\tEk(cm-1)\tg_i\tg_k\n"
        "777.1944\t3.69e7\tA\t73768.200\t86631.454\t5\t7\n"
        "777.4166\t3.69e7\tA\t73768.200\t86627.778\t5\t5\n"
        "777.5388\t3.69e7\tA\t73768.200\t86625.757\t5\t3\n"
    )
    downloads = []

    def download(manager, urls, targets, total):
        downloads.append(urls)
        opener = SimpleNamespace(open=lambda: source.open("rb"))
        return manager.parse_to_local_file(opener, urls[0], targets[0], pbar_active=False)

    monkeypatch.setattr(DatabaseManager, "is_registered", lambda self: False)
    monkeypatch.setattr(radis_api.NISTDatabaseManager, "download_and_parse", download)
    monkeypatch.setattr(radis_api.NISTDatabaseManager, "register", lambda *args: None)
    kwargs = dict(nurange=[12850, 12870], local_databases=str(tmp_path / "cache"))
    first = AdbNist("O_I", **kwargs)
    second = AdbNist("O_I", **kwargs)
    assert len(downloads) == 1
    np.testing.assert_allclose(first.nu_lines, [12857.557, 12859.578, 12863.254])
    np.testing.assert_array_equal(first.nu_lines, second.nu_lines)
    np.testing.assert_array_equal(first.A, np.full(3, 3.69e7))
    np.testing.assert_array_equal(first.jupper, [1.0, 2.0, 3.0])
    opa = OpaDirect(
        first, np.linspace(12850, 12870, 81),
        atomic_broadening=lambda T, P: jnp.full_like(first.A, 0.01),
    )
    opacity = opa.xsvector(5000.0, 0.1)
    assert np.all(np.isfinite(opacity)) and np.all(opacity > 0)
