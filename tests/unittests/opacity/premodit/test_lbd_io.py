"""Portable LBD artifacts preserve spectra without retaining line lists."""

from pathlib import Path
import json
import os
import runpy
import subprocess
import sys

import h5py
import numpy as np
import pytest

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity import OpaPremodit, load_lbd, save_lbd
from exojax.opacity._common.profconv import calc_xsection_from_lsd_zeroscan
from exojax.opacity.premodit import premodit


EXAMPLE = Path(__file__).resolve().parents[4] / "examples/read_premodit_lbd.py"


@pytest.fixture
def make_opa(isolated_test_environment):
    def build(diffmode=2, profile_kernel="analytic"):
        snapshot = MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol",
                molmass=28.0,
                T_gQT=np.array([200.0, 600.0, 900.0, 1500.0]),
                gQT=np.array([50.0, 150.0, 260.0, 550.0]),
            ),
            lines=Lines(
                nu_lines=np.array([1001.1, 1002.2]),
                elower=np.array([40.0, 170.0]),
                line_strength_ref_original=np.array([1e-21, 2e-21]),
            ),
            n_Texp=np.array([0.5, 0.7]),
            alpha_ref=np.array([0.04, 0.08]),
        )
        opa = OpaPremodit.from_snapshot(
            snapshot,
            np.geomspace(1000.0, 1004.0, 16),
            diffmode=diffmode,
            profile_kernel=profile_kernel,
            broadening_resolution={"mode": "minmax", "value": None},
        )
        opa.manual_setting(100.0, 800.0, 1000.0, Tmin=700.0, Tmax=1200.0)
        return opa

    return build


@pytest.fixture
def metadata():
    return {
        "mol_name": "CO",
        "key_iso_ll": "12C-16O__synthetic",
        "DOI": "synthetic-test-data",
        "Date_ID": "test-v1",
        "broadener": "H2",
        "valid_wavenumber_range": [1000.5, 1003.5],
        "provenance": {
            "source": "Two synthetic transitions; not an ExoMol release.",
            "loader": {"name": "synthetic", "version": "1"},
        },
    }


@pytest.mark.parametrize("diffmode", [0, 1, 2])
@pytest.mark.parametrize("profile_kernel", ["analytic", "real_space"])
def test_reader_and_standalone_consumer_preserve_spectra(
    tmp_path, make_opa, metadata, diffmode, profile_kernel
):
    opa = make_opa(diffmode, profile_kernel)
    expected = {
        (temperature, pressure): np.asarray(opa.xsvector(temperature, pressure))
        for temperature in (800.0, 1000.0, 1200.0)
        for pressure in (0.01, 1.0, 10.0)
    }
    energies = opa._get_info_tuple()[1]
    expected_lsd = {}
    for temperature in (800.0, 1000.0, 1200.0):
        qr = opa.pf_provider.qr_single(temperature, opa.Tref)
        if diffmode == 0:
            lsd = premodit.unbiased_lsd_zeroth(
                opa.lbd_coeff[0], temperature, opa.Tref, opa.nu_grid, energies, qr
            )
        else:
            unbias = premodit.unbiased_lsd_first if diffmode == 1 else premodit.unbiased_lsd_second
            lsd = unbias(
                opa.lbd_coeff, temperature, opa.Tref, opa.Twt, opa.nu_grid, energies, qr
            )
        expected_lsd[temperature] = np.asarray(lsd)
    path = tmp_path / "co.premodit.h5"
    save_lbd(opa, path, metadata=metadata)
    arrays, loaded_meta = load_lbd(path)
    consumer = runpy.run_path(str(EXAMPLE))
    standalone_arrays, standalone_meta = consumer["load_lbd"](path)

    np.testing.assert_array_equal(arrays["lbd_coeff"], opa.lbd_coeff)
    assert arrays["lbd_coeff"].dtype == np.float64
    assert np.isneginf(arrays["lbd_coeff"][0]).any()
    if diffmode:
        assert (arrays["lbd_coeff"][1:] < 0).any()
    for name in arrays:
        np.testing.assert_array_equal(standalone_arrays[name], arrays[name])
    for name in ("mol_name", "key_iso_ll", "DOI", "Date_ID"):
        assert loaded_meta[name] == standalone_meta[name] == metadata[name]
    provenance = loaded_meta["provenance"]
    assert provenance["broadener"] == "H2"
    assert provenance["valid_wavenumber_range"] == metadata["valid_wavenumber_range"]
    assert provenance["source"] == metadata["provenance"]["source"]

    with h5py.File(path, "r") as handle:
        assert handle["premodit/lbd_coeff"].compression == "gzip"
        assert handle["premodit/lbd_coeff"].shuffle
        assert set(handle["premodit"]) == {
            "lbd_coeff", "nu_grid", "elower_grid", "ngamma_ref_grid",
            "n_Texp_grid", "multi_index_uniqgrid",
        }
        assert set(handle["partition_function"]) == {"temperature", "Q"}

    # Evaluate from the exported arrays after discarding the original calculator.
    del opa
    path.unlink()  # Both readers must own their arrays and close the file.
    p = standalone_meta["premodit"]
    nu = standalone_arrays["nu_grid"]
    pmarray = (-1.0) ** np.arange(len(nu) + 1)
    for (temperature, pressure), expected_xs in expected.items():
        lsd, ngamma, nsigma = consumer["evaluation_inputs"](
            standalone_arrays, standalone_meta, temperature, pressure
        )
        np.testing.assert_allclose(lsd, expected_lsd[temperature], rtol=2e-12, atol=0)
        actual = calc_xsection_from_lsd_zeroscan(
            lsd, p["R"], pmarray, nsigma, nu, np.log(ngamma),
            profile_kernel=profile_kernel,
        )
        # FFT cancellation in weak bins is limited by the peak opacity scale.
        roundoff = 32 * np.finfo(np.float64).eps * np.max(np.abs(expected_xs))
        np.testing.assert_allclose(actual, expected_xs, rtol=2e-12, atol=roundoff)


def test_reader_import_does_not_load_compute_backends():
    script = (
        "import sys; from exojax.opacity import load_lbd; "
        "assert not {'jax', 'radis', 'pyexocross'} & sys.modules.keys()"
    )
    env = dict(os.environ, PYTHONPATH=str(EXAMPLE.parents[1] / "src"))
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_existing_file_is_preserved(tmp_path, make_opa, metadata):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        save_lbd(make_opa(), path, metadata=metadata)
    assert path.read_bytes() == b"existing artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_write_does_not_publish_or_leave_temporary_file(
    tmp_path, make_opa, metadata, monkeypatch
):
    opa = make_opa()
    original = h5py.Group.create_dataset

    def fail_on_coefficients(group, name, *args, **kwargs):
        if name == "lbd_coeff":
            raise OSError("Simulated disk failure")
        return original(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", fail_on_coefficients)
    with pytest.raises(OSError, match="Simulated disk failure"):
        save_lbd(opa, tmp_path / "incomplete.h5", metadata=metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda opa: setattr(opa, "ready", False),
        lambda opa: setattr(opa, "nstitch", 2),
        lambda opa: setattr(opa, "dbtype", "hitran"),
        lambda opa: setattr(opa, "lbd_coeff", opa.lbd_coeff.astype(np.float32)),
    ],
    ids=["not-ready", "stitched", "hitran", "float32"],
)
def test_writer_rejects_unsupported_calculators(tmp_path, make_opa, metadata, change):
    opa = make_opa()
    change(opa)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        save_lbd(opa, tmp_path / "invalid.h5", metadata=metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["missing-broadener", "invalid-range", "reserved-provenance"])
def test_writer_requires_unambiguous_metadata(tmp_path, make_opa, metadata, change):
    if change == "missing-broadener":
        del metadata["broadener"]
    elif change == "invalid-range":
        metadata["valid_wavenumber_range"] = [990.0, 1003.0]
    else:
        metadata["provenance"]["abundance_convention"] = "preweighted"
    with pytest.raises((ValueError, KeyError)):
        save_lbd(make_opa(), tmp_path / "invalid.h5", metadata=metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "path, attribute, value",
    [
        ("/", "format", "another-format"),
        ("/", "schema_version", "2.0"),
        ("premodit/lbd_coeff", "encoding", "all-logarithmic"),
        ("premodit/lbd_coeff", "axes", "elower,broadening_pair,wavenumber,derivative_order"),
        ("premodit/nu_grid", "units", "m^-1"),
        ("premodit", "pressure_unit", "Pa"),
        ("premodit", "diffmode", 1.5),
        ("premodit", "Tref", None),
    ],
)
def test_reader_rejects_incompatible_contracts(
    tmp_path, make_opa, metadata, path, attribute, value
):
    artifact = tmp_path / "invalid.h5"
    save_lbd(make_opa(), artifact, metadata=metadata)
    with h5py.File(artifact, "r+") as handle:
        if value is None:
            del handle[path].attrs[attribute]
        else:
            handle[path].attrs[attribute] = value
    with pytest.raises(ValueError):
        load_lbd(artifact)


@pytest.mark.parametrize(
    "corruption",
    ["shape", "index", "q-range", "nan-grid", "infinite-derivative", "irregular-nu"],
)
def test_reader_rejects_invalid_numerical_data(tmp_path, make_opa, metadata, corruption):
    path = tmp_path / "invalid.h5"
    save_lbd(make_opa(), path, metadata=metadata)
    with h5py.File(path, "r+") as handle:
        if corruption == "shape":
            coeff = handle["premodit/lbd_coeff"]
            attributes = dict(coeff.attrs)
            shortened = coeff[:, :-1]
            del handle["premodit/lbd_coeff"]
            handle.create_dataset("premodit/lbd_coeff", data=shortened).attrs.update(attributes)
        elif corruption == "index":
            handle["premodit/multi_index_uniqgrid"][0, 0] = 100000
        elif corruption == "q-range":
            handle["partition_function/temperature"][:] = [100.0, 200.0, 300.0, 400.0]
        elif corruption == "nan-grid":
            handle["premodit/ngamma_ref_grid"][0] = np.nan
        elif corruption == "infinite-derivative":
            handle["premodit/lbd_coeff"][1, 0, 0, 0] = -np.inf
        else:
            handle["premodit/nu_grid"][3] += 0.01
    with pytest.raises(ValueError):
        load_lbd(path)


def test_same_major_version_and_new_exojax_version_are_portable(tmp_path, make_opa, metadata):
    path = tmp_path / "future-minor.h5"
    save_lbd(make_opa(), path, metadata=metadata)
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "1.1"
        provenance = json.loads(handle["provenance_json"].asstr()[()])
        provenance["writer"]["exojax_version"] = "999.0.0"
        handle["provenance_json"][()] = json.dumps(provenance)
    arrays, _ = load_lbd(path)
    assert arrays["lbd_coeff"].shape[0] == 3
    consumer = runpy.run_path(str(EXAMPLE))
    standalone_arrays, _ = consumer["load_lbd"](path)
    np.testing.assert_array_equal(standalone_arrays["lbd_coeff"], arrays["lbd_coeff"])
