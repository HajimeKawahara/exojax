"""Check finite-state normalization, isolated assets and input pinning."""

import bz2
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def partition(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_hcn_partition")


@pytest.fixture
def source(partition, tmp_path):
    directory = tmp_path / "original"
    directory.mkdir()
    stem = partition.STEM
    (directory / f"{stem}.def").write_text("4000 # Maximum temperature of linelist\n")
    states = ("1 0 6 0 NaN + e 0 0 0 0 0\n"
              "2 5705 6 0 NaN + e 1 0 0 0 0\n"
              "3 10000 18 1 NaN - f * * * * *\n")
    (directory / f"{stem}.states.bz2").write_bytes(bz2.compress(states.encode()))
    temperature = np.array([1., 296., 1500., 2500., 4000.])
    energy, weight, labels = partition.read_states(directory / f"{stem}.states.bz2")
    q = partition.partition_parts(temperature, energy, weight, labels)[:, 0]
    np.savetxt(directory / f"{stem}.pf", np.column_stack((temperature, q)))
    hashes = {ext: partition.sha256(directory / f"{stem}.{ext}") for ext in partition.SOURCE_SHA256}
    return directory, hashes


def test_direct_sum_keeps_unknown_states_and_nuclear_spin_once(partition):
    energy, weight, labels = np.array([0., 5705., 10000.]), np.array([6., 6., 18.]), np.array(["0", "1", "*"])
    temperatures = np.array([1500., 4000., 4500.])
    actual = partition.partition_parts(temperatures, energy, weight, labels)
    expected = weight[None, :] * np.exp(-partition.hcperk * energy[None, :] / temperatures[:, None])
    np.testing.assert_allclose(actual[:, 0], expected.sum(axis=1), rtol=0., atol=0.)
    np.testing.assert_array_equal(actual[:, 1:], expected)
    assert actual[-1, 0] > actual[-2, 0]  # No 4000 K clamp.
    assert np.all(actual[:, 3] > 0.)
    with pytest.raises(ValueError, match="finite positive"):
        partition.partition_parts([np.nan], energy, weight, labels)


def test_asset_preserves_originals_and_rejects_modified_resume(partition, source, tmp_path):
    directory, hashes = source
    output, record = partition.prepare_asset(directory, tmp_path / "derived", expected_sha256=hashes)
    assert hashes == {ext: partition.sha256(directory / f"{partition.STEM}.{ext}") for ext in hashes}
    assert partition.sha256(output / f"{partition.STEM}.pf.original") == hashes["pf"]
    q = np.loadtxt(output / f"{partition.STEM}.pf")
    np.testing.assert_array_equal(q[:, 0], np.arange(1., 4501.))
    assert record["declared_line_list_maximum_temperature_K"] == 4000.
    assert record["label_counts"] == {"0": 1, "1": 1, "*": 1}
    repeated, resumed = partition.prepare_asset(directory, tmp_path / "derived", expected_sha256=hashes)
    assert repeated == output and resumed == record
    (output / f"{partition.STEM}.pf").write_text("changed")
    with pytest.raises(ValueError, match="Changed finite-state"):
        partition.prepare_asset(directory, tmp_path / "derived", expected_sha256=hashes)


def test_incompatible_pf_is_rejected_before_writing(partition, source, tmp_path):
    directory, hashes = source
    path = directory / f"{partition.STEM}.pf"
    values = np.loadtxt(path)
    values[:, 1] *= 2.  # A doubled spin factor must not be silently accepted.
    np.savetxt(path, values)
    hashes["pf"] = partition.sha256(path)
    with pytest.raises(ValueError, match="does not reproduce"):
        partition.prepare_asset(directory, tmp_path / "bad", expected_sha256=hashes)
    assert not (tmp_path / "bad").exists()


def test_normalization_distinguishes_chemical_and_partial_state_ratios(partition, source, tmp_path):
    directory, hashes = source
    output, _ = partition.prepare_asset(directory, tmp_path / "derived", expected_sha256=hashes)
    actual = partition.normalization_diagnostics(output, [4500.], [2e-6], [1e-6])
    assert actual["chemical_hnc_over_hcn"] == [.5]
    np.testing.assert_allclose(actual["combined_number_fraction"], [3e-6])
    assert actual["assigned_state_hnc_over_hcn"][0] != .5
    assert actual["unassigned_partition_fraction"][0] > 0.
    with pytest.raises(ValueError, match="outside"):
        partition.normalization_diagnostics(output, [4501.], [2e-6], [1e-6])
