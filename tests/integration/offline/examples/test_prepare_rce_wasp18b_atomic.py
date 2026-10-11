"""Check atomic batching against native opacity and persist resumable tables."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_atomic")


def line(wavelength, species):
    fields = [(0, 11, f"{wavelength:11.4f}"), (11, 18, " -1.000"),
        (18, 24, f"{species:>6}"), (24, 36, f"{100.:12.3f}"),
        (36, 41, "  0.5"), (52, 64, f"{100. + 1e7 / wavelength:12.3f}"),
        (64, 69, "  1.5"), (80, 86, "  8.00"), (86, 92, " -6.00"), (92, 98, " -7.00")]
    row = [" "] * 160
    for start, stop, value in fields:
        row[start:stop] = value
    return "".join(row) + "\n"


def arguments(preparation, tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema_version": preparation.base.SCHEMA_VERSION,
        "column": {"pressure_boundaries_bar": [.001, .01, .1]}}))
    return ["--database-root", str(tmp_path / "database"), "--inventory", str(inventory),
        "--species", "Fe1", "--nu-min", "19990", "--nu-max", "20000",
        "--resolved-nu-min", "19990", "--resolved-nu-max", "20000",
        "--wing-padding", "1", "--samples-per-band", "32",
        "--ckd-resolution", "1000", "--bands-per-chunk", "2",
        "--temperature-nodes", "2", "--ng", "2"]


def test_grid_contract_and_dry_run(preparation, tmp_path, capsys):
    from exogibbs.thermo.standard import prepare_fastchem_thermodynamics

    assert set(preparation.ATOMIC_LINES) <= set(prepare_fastchem_thermodynamics().chemical_setup.species)
    argv = arguments(preparation, tmp_path)
    args = preparation.parser_with_defaults().parse_args(argv)
    plan = preparation.make_plan(args)
    args.mdb_path = Path(plan["mdb_path"])
    assert plan["settings"] == preparation.base.make_plan(args)["settings"]
    assert preparation.main(argv + ["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["atomic_preparation"]["line_profile"] == "voigt"
    assert not Path(plan["output_dir"]).exists()
    assert not Path(plan["mdb_path"]).exists()
    args.species, args.line_profile, args.atomic_line_padding = "Na1", "auto", 8999.
    with pytest.raises(ValueError, match="9000"):
        preparation.make_plan(args)


@pytest.mark.parametrize("species,code,profile", [
    ("Fe1", "26.00", "voigt"), ("Na1", "11.00", "alkali_subvoigt"),
    ("K1", "19.00", "alkali_subvoigt")])
def test_batched_native_profiles_and_temperature_jvp(preparation, tmp_path, species, code, profile):
    from exojax.database.kurucz.api import AdbKurucz
    from exojax.opacity import OpaDirect

    source = tmp_path / "lines.all"
    source.write_text("".join(line(wave, code) for wave in (499., 500., 501.)))
    adb = AdbKurucz(source, vmr_fraction=[.25, .15, .60])
    grid = np.linspace(19930., 20060., 42)
    native = OpaDirect(adb, grid, line_profile=profile)
    teacher = preparation.BatchedAtomicTeacher(adb, grid, species, profile, 2)
    pressure = jnp.array([.001, 10.])
    temperature = jnp.array([1500., 4500.])
    actual, slope = jax.jvp(lambda t: teacher.xsmatrix(t, pressure), (temperature,), (jnp.ones_like(temperature),))
    expected, expected_slope = jax.jvp(lambda t: native.xsmatrix(t, pressure), (temperature,), (jnp.ones_like(temperature),))
    assert np.all(np.isfinite(actual)) and np.all(actual > 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
    # Fusion changes rounding in the Voigt JVP; scale each atmospheric layer
    # by its peak derivative so nearly cancelling entries do not set tolerance.
    scale = np.max(np.abs(expected_slope), axis=1, keepdims=True)
    np.testing.assert_allclose(slope / scale, expected_slope / scale, rtol=1e-11, atol=1e-11)
    compiled_native = jax.jit(lambda t: native.xsmatrix(t, pressure))
    _, compiled_slope = jax.jvp(compiled_native, (temperature,), (jnp.ones_like(temperature),))
    np.testing.assert_allclose(slope / scale, compiled_slope / scale, rtol=1e-12, atol=1e-12)
    if species == "Fe1":
        # 1500 K is a partition-function interpolation knot: a centered
        # difference there measures two different slopes. Use an interior T.
        interior = jnp.array([1600., 4500.])
        fun = lambda t: teacher.xsmatrix(t, pressure)
        _, derivative = jax.jvp(fun, (interior,), (jnp.ones_like(interior),))
        step = .3
        finite = (-fun(interior + 2 * step) + 8 * fun(interior + step)
                  - 8 * fun(interior - step) + fun(interior - 2 * step)) / (12 * step)
        scale = np.max(np.abs(derivative), axis=1, keepdims=True)
        # The native approximate Voigt/custom-JVP pair itself differs from
        # centered differences by 1.3e-7 on this normalized scale.
        np.testing.assert_allclose(derivative / scale, finite / scale, rtol=0., atol=5e-7)


@pytest.mark.parametrize("wavelength,expected_lines", [(500., 1), (900., 0)])
def test_chunk_persistence_dispatch_metadata_and_resume(preparation, tmp_path, monkeypatch, wavelength, expected_lines):
    from exojax.opacity import OpaDiffgrid

    source = tmp_path / "gf2600.all"
    source.write_text(line(wavelength, "26.00"))
    argv = arguments(preparation, tmp_path) + ["--mdb-path", str(source)]
    calls = []

    def child(command, check):
        assert check and Path(command[1]).resolve() == Path(preparation.__file__).resolve()
        calls.append(command)
        assert preparation.main(command[2:]) == 0

    monkeypatch.setattr(preparation, "subprocess", SimpleNamespace(run=child))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    assert preparation.main(argv) == 0
    assert len(calls) == 1
    plan = preparation.make_plan(preparation.parser_with_defaults().parse_args(argv))
    records = preparation.completed_chunks(plan)
    assert records[0]["provenance"]["number_of_lines"] == expected_lines
    complete = json.loads((Path(plan["output_dir"]) / "Fe1/complete.json").read_text())
    diffgrid = OpaDiffgrid.from_saved_opa(complete["files"]["diffgrid.npz"]["path"])
    assert diffgrid.teacher_method == "atomic_lpf_batched"
    assert diffgrid.teacher_profile_kernel is None
    assert preparation.main(argv) == 0 and len(calls) == 1
    changed = dict(plan, atomic_preparation=dict(plan["atomic_preparation"], line_profile="changed"))
    with pytest.raises(ValueError, match="different physics"):
        preparation.completed_chunks(changed)
    source.write_text(line(501., "26.00"))
    with pytest.raises(ValueError, match="Changed source"):
        preparation.completed_chunks(plan)
