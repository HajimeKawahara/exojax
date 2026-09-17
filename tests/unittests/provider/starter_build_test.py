"""Scientific preservation checks for the starter ExoMolOP subset."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "build_starter_opacity",
    Path(__file__).resolve().parents[3] / "tools" / "build_starter_opacity.py",
)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


@pytest.fixture
def h2o_source(tmp_path):
    source = tmp_path / builder.H2O_URL.rsplit("/", 1)[1]
    centers = np.array([500, 1000, 1800, 2200, 4000, 8000, 9900, 11000, 13000.0])
    temperatures = np.array([200, 400, 800, 1200, 1600, 2000.0])
    pressures = np.array([1e-6, 5e-6, 1e-4, 1, 20, 100.0])
    pressure, temperature, band, sample = np.indices((6, 6, 9, 3))
    # Distinct axis contributions reveal accidental axis swaps and offsets.
    coefficients = (pressure * 10000 + temperature * 1000 + band * 10 + sample) * 1e-25
    coefficients[2, 2, 3, 1] = 0.0
    arrays = {
        "kcoeff": coefficients,
        "bin_centers": centers,
        "bin_edges": np.r_[250.0, 0.5 * (centers[:-1] + centers[1:]), 14000.0],
        "t": temperatures,
        "p": pressures,
        "samples": np.array([0.1, 0.5, 0.9]),
        "weights": np.array([0.2, 0.6, 0.2]),
        "mol_mass": np.array([18.0]),
    }
    with h5py.File(source, "w") as handle:
        for key, data in arrays.items():
            handle.create_dataset(key, data=data)
        for key, units in {
            "kcoeff": "cm^2/molecule", "bin_centers": "cm^-1",
            "bin_edges": "cm^-1", "t": "K", "p": "bar",
        }.items():
            handle[key].attrs["units"] = units
        handle.attrs["source_note"] = "Synthetic ExoMolOP preservation fixture"
        handle.create_dataset("DOI", data=np.array([b"10.1093/mnras/sty1877"]))
        handle.create_dataset("method", data=np.array([b"petit_samples"]))
        handle.create_dataset("ngauss", data=3)
    return source, arrays


def test_crop_preserves_native_axis_order_guards_and_coefficients(tmp_path, h2o_source):
    source, original = h2o_source
    destination = tmp_path / "subset.h5"
    builder.crop_h2o(source, destination)

    with h5py.File(destination) as subset:
        # T/P endpoints must bracket the advertised range on native grid nodes.
        np.testing.assert_array_equal(subset["t"][:], [400, 800, 1200, 1600])
        np.testing.assert_array_equal(subset["p"][:], [5e-6, 1e-4, 1, 20])
        # Retain one native guard band on both sides of 2000--10000 cm^-1.
        np.testing.assert_array_equal(
            subset["bin_centers"][:], [1800, 2200, 4000, 8000, 9900, 11000]
        )
        np.testing.assert_array_equal(subset["bin_edges"][:], original["bin_edges"][2:9])
        np.testing.assert_array_equal(
            subset["kcoeff"][:], original["kcoeff"][1:5, 1:5, 2:8, :]
        )
        assert subset["kcoeff"].dtype == original["kcoeff"].dtype
        assert subset["kcoeff"][1, 1, 1, 1] == 0.0
        for key in ("samples", "weights", "mol_mass"):
            np.testing.assert_array_equal(subset[key][:], original[key])


def test_crop_preserves_units_and_source_attribution(tmp_path, h2o_source):
    source, _ = h2o_source
    destination = tmp_path / "subset.h5"
    builder.crop_h2o(source, destination)

    with h5py.File(source) as original, h5py.File(destination) as subset:
        assert subset.attrs["source_note"] == original.attrs["source_note"]
        for key in ("kcoeff", "bin_centers", "bin_edges", "t", "p"):
            assert dict(subset[key].attrs) == dict(original[key].attrs)
        for key in ("DOI", "method", "ngauss"):
            np.testing.assert_array_equal(subset[key][()], original[key][()])
        assert subset["mol_name"][0] == b"H2O"


@pytest.mark.parametrize("axis", ["bin_centers", "t", "p"])
def test_crop_rejects_incomplete_coverage_without_output(tmp_path, h2o_source, axis):
    source, _ = h2o_source
    with h5py.File(source, "r+") as handle:
        grid = handle[axis][:]
        grid += {"bin_centers": 2000.0, "t": 400.0, "p": 1e-4}[axis]
        handle[axis][:] = grid
    destination = tmp_path / "subset.h5"

    with pytest.raises(ValueError, match="cover"):
        builder.crop_h2o(source, destination)

    assert not destination.exists()
