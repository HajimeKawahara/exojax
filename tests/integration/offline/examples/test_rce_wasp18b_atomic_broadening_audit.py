"""Verify native pressure substitution and audit isolation on small fixtures."""

from copy import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_prepare_rce_wasp18b_atomic import arguments, line
from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield importlib.import_module("rce_wasp18b_atomic_broadening_audit")
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("species,code,profile", [
    ("Fe1", "26.00", "voigt"), ("Na1", "11.00", "alkali_subvoigt"),
    ("K1", "19.00", "alkali_subvoigt")])
def test_native_effective_pressure_matches_layer_specific_partial_pressures(audit, tmp_path, species, code, profile):
    from exojax.database.kurucz.api import AdbKurucz
    from exojax.opacity import OpaDirect

    # Exercise both tabulated vdW and the Unsold fallback in one line list.
    rows = [line(wave, code) for wave in (499., 500., 501.)]
    rows[1] = rows[1][:92] + "  0.00" + rows[1][98:]
    source = tmp_path / "fe.all"
    source.write_text("".join(rows))
    fixed = np.array([0., .16, .84])
    adb = AdbKurucz(source, vmr_fraction=fixed)
    actual_adb = copy(adb)
    fractions = np.array([[.10, .10, .70], [.90, .08, .01]])
    actual_adb.vmrH, actual_adb.vmrHe, actual_adb.vmrHH = map(jnp.asarray, fractions.T)
    grid = np.linspace(19930., 20060., 48)
    temperature, pressure = jnp.array([1600., 4300.]), np.array([.001, 100.])
    effective = audit.effective_pressure(pressure, fractions, fixed)
    assert effective[-1] > pressure[-1]  # Native evaluation has no saved-P clamp.
    native = audit.atomic.BatchedAtomicTeacher(adb, grid, species, profile, 2)
    expected = OpaDirect(actual_adb, grid, line_profile=profile).xsmatrix(temperature, jnp.asarray(pressure))
    result = native.xsmatrix(temperature, jnp.asarray(effective))
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=0.)
    fixed_result = native.xsmatrix(temperature, jnp.asarray(pressure))
    assert np.max(np.abs(np.asarray(result / fixed_result) - 1.)) > .01
    np.testing.assert_array_equal(audit.effective_pressure(pressure, np.tile(fixed, (2, 1)), fixed), pressure)


def test_pairing_uses_four_layer_batches_and_original_endpoint_weights(audit):
    from exojax.opacity.ckd.core import compute_ckd_from_xsmatrix, gauss_legendre_grid

    class Teacher:
        ready, method = True, "fixture"
        nu_grid = np.linspace(1., 3., 9)

        def __init__(self):
            self.calls = []

        def xsmatrix(self, temperature, pressure):
            self.calls.append((np.asarray(temperature).copy(), np.asarray(pressure).copy()))
            return temperature[:, None] * pressure[:, None] * (1. + jnp.asarray(self.nu_grid)[None, :]**2)

    native = Teacher()
    teacher = audit.atomic.base.CroppedTeacher(native, native.nu_grid, slice(None), {})
    temperature, pressure = np.arange(6.) + 1500., np.arange(6.) + 1.
    effective = 1.1 * pressure
    edges = np.array([[1., 2.], [2., 3.]])
    logk, counts = audit.paired_native_ckd(teacher, temperature, pressure, effective, edges, 4)
    assert counts == [5, 5]  # The shared endpoint occurs in both bands.
    assert [len(t) for t, _ in native.calls] == [4, 4, 4]
    np.testing.assert_array_equal(np.concatenate([p for _, p in native.calls]), np.r_[pressure, effective])
    g, _ = gauss_legendre_grid(4)
    reference = temperature[:, None] * pressure[:, None] * (1. + native.nu_grid[:5][None, :]**2)
    np.testing.assert_allclose(logk[0, ..., 0], compute_ckd_from_xsmatrix(reference, g), rtol=2e-15)
    np.testing.assert_allclose(logk[1] - logk[0], np.log(1.1), rtol=2e-13)


@pytest.mark.parametrize("species,code", [("Fe1", "26.00"), ("Na1", "11.00"), ("K1", "19.00")])
def test_native_chunk_reproduces_frozen_preparation_at_original_nodes(audit, tmp_path, species, code):
    source = tmp_path / "fe.all"
    source.write_text(line(500., code))
    argv = arguments(audit.atomic, tmp_path) + ["--mdb-path", str(source), "--species", species]
    plan = audit.atomic.make_plan(audit.atomic.parser_with_defaults().parse_args(argv))
    audit.atomic.prepare_chunk(plan, 0)
    temperature = np.sort(plan["settings"]["temperature_grid_K"])
    pressure = np.asarray(plan["settings"]["pressure_grid_bar"])
    result, edges, metadata = audit.native_chunk(plan, 0, temperature, pressure, pressure)
    path = Path(plan["output_dir"]) / species / "chunk_00000/ckd.npz"
    with np.load(path, allow_pickle=False) as saved:
        reference = saved["log_kggrid"][np.arange(2), np.arange(2)]
        np.testing.assert_array_equal(edges, saved["band_edges"])
    np.testing.assert_allclose(result[0], reference, rtol=0., atol=3e-14)
    np.testing.assert_array_equal(result[0], result[1])
    assert metadata["number_of_lines"] == 1


def test_atom_replacement_keeps_other_species_and_physical_columns(audit):
    from exojax.opacity.ckd.mixing import mix_ckd_rorr

    depths = jnp.arange(1., 25.).reshape(3, 2, 2, 2)
    original = np.asarray(depths).copy()
    abundance_columns = jnp.array([3., 7.])
    fe = jnp.array([[[.1, .2], [.3, .4]], [[.5, .6], [.7, .8]]])
    continuum, weights = jnp.ones((2, 2)), jnp.array([.4, .6])
    actual = audit.mix_with_atom(depths, fe, abundance_columns, continuum, weights, 1)
    direct = depths.at[1].set(fe * abundance_columns[:, None, None])
    expected = mix_ckd_rorr(direct, weights) + continuum[:, None, :]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(depths, original)


@pytest.mark.parametrize("scattering", [False, True])
def test_chunk_flux_uses_production_transfer_without_reapplying_irradiation(audit, modules, tmp_path, scattering):
    _, column, chemistry = fixture_manifest(modules, tmp_path)
    column.settings = replace(column.settings, stellar_scattering=scattering)
    temperature, bottom, parameters = jnp.array([2300., 2700.]), 3000., jnp.array([0., .6, -.69])
    broadband = importlib.import_module("_rce_broadband")
    edges = jnp.array([[3500., 3700.], [3700., 4500.]])
    column.band_nodes, column.band_weights = broadband.make_band_quadrature(edges, 4)
    column.band_widths = jnp.diff(edges, axis=1).ravel()
    _, column.g_weights = audit.gauss_legendre_grid(4)
    column.star_bands = broadband.band_stellar_flux(column.stellar.nu_cm1, column.stellar.surface_flux_per_cm1, edges)
    column.ckd = {name: SimpleNamespace(xstensor_ckd=lambda t, p: jnp.broadcast_to(
        jnp.arange(1., 5.)[None, :, None]*1e-24, (2, 4, 2))) for name in column.lbl}
    depth = column.optical_depth(temperature, chemistry)
    expected = column.fluxes(temperature, bottom, parameters, chemistry)
    actual = audit.flux_at_depth(column, depth, column.band_nodes, column.star_bands,
        temperature, bottom, parameters, chemistry)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.)
    partial = audit.flux_at_depth(column, depth[..., :1], column.band_nodes[:1], column.star_bands[:1],
        temperature, bottom, parameters, chemistry)
    np.testing.assert_allclose(partial, np.asarray(expected)[..., :1], rtol=2e-12, atol=0.)


def test_checkpoint_reuses_bytes_and_rejects_changed_identity_or_coefficients(audit, tmp_path):
    calls = []

    def calculate():
        calls.append(True)
        return np.arange(8.).reshape(2, 1, 2, 2), np.array([[1., 2.], [2., 3.]]), {"number_of_lines": 3}

    first = audit.cached_native_chunk(tmp_path, "original", 7, calculate)
    old_bytes = (tmp_path / "chunk_00007.json").read_bytes()
    again = audit.cached_native_chunk(tmp_path, "original", 7, calculate)
    assert len(calls) == 1
    np.testing.assert_array_equal(first[0], again[0])
    assert (tmp_path / "chunk_00007.json").read_bytes() == old_bytes
    with pytest.raises(ValueError, match="Changed"):
        audit.cached_native_chunk(tmp_path, "new-state", 7, calculate)
    (tmp_path / "chunk_00007.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed"):
        audit.cached_native_chunk(tmp_path, "original", 7, calculate)


@pytest.mark.parametrize("fractions", [[[.8, .3, .1]], [[-.1, .2, .8]], [[np.nan, 0., 0.]]])
def test_reject_invalid_all_particle_fractions(audit, fractions):
    with pytest.raises(ValueError, match="all-particle"):
        audit.effective_pressure([1.], fractions, [0., .16, .84])
