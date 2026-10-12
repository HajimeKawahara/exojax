"""Streaming caches must reproduce native RADIS tables and publish atomically."""

import bz2
import hashlib
import importlib
import io
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from exojax.database.exomol.api import MdbExomol
from exojax.test.data import get_testdata_filename


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("prepare_rce_wasp18b_line_cache")


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    import requests

    def no_download(*args, **kwargs):
        pytest.fail("The bundled fixture attempted a download")

    monkeypatch.setattr(MdbExomol, "download", no_download)
    monkeypatch.setattr(MdbExomol, "is_registered", lambda self: False)
    monkeypatch.setattr(requests.sessions.Session, "request", no_download)
    source = Path(get_testdata_filename("CO")) / "12C-16O" / "SAMPLE"
    result = []
    for name in ("reference", "streamed"):
        path = tmp_path / name / "CO" / "12C-16O" / "SAMPLE"
        path.mkdir(parents=True)
        for item in source.iterdir():
            if item.name.endswith((".bz2", ".def", ".pf", ".broad")):
                shutil.copy2(item, path / item.name)
        result.append(path)
    return result


def inputs(directory, optional=False):
    from radis.api.exomolapi import read_def, read_pf, read_states

    stem = "12C-16O__SAMPLE"
    definition = read_def(directory / f"{stem}.def")
    states = read_states(directory / f"{stem}.states.bz2", definition, engine="csv", skip_optional_data=not optional)
    pf = read_pf(directory / f"{stem}.pf")
    return directory / f"{stem}.trans.bz2", states, definition, np.interp(296., pf["T"], pf["QT"])


@pytest.mark.parametrize("columns,optional", [(4, False), (3, False), (4, True), (3, True)])
def test_streamed_cache_matches_native_radis(helper, datasets, columns, optional):
    reference, streamed = datasets
    if columns == 3:
        for directory in datasets:
            path = directory / "12C-16O__SAMPLE.trans.bz2"
            data = pd.read_csv(path, sep=r"\s+", header=None)
            with bz2.open(path, "wt") as file:
                data.iloc[:, :3].to_csv(file, sep=" ", header=False, index=False)
    MdbExomol(reference, [1., 1.e9], engine="pytables", broadf_download=False,
              gpu_transfer=False, activation=False, optional_quantum_states=optional)
    raw, states, definition, partition = inputs(streamed, optional)
    result = helper.build_transition_cache(raw, states, definition, partition,
                                          rows_per_chunk=73, optional_quantum_states=optional)
    expected = pd.read_hdf(reference / "12C-16O__SAMPLE.trans.h5", key="df")
    actual = pd.read_hdf(result["path"], key="df")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert result["rows"] == len(expected)
    assert not result["reused"]
    with pd.HDFStore(result["path"], "r") as store:
        storer = store.get_storer("df")
        assert "nu_lines" in storer.data_columns
        assert "nu_lines" in storer.table.colindexes
        assert "index" in storer.table.colindexes
    path = Path(result["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stamp = path.stat().st_mtime_ns
    assert helper.build_transition_cache(raw, states, definition, partition, rows_per_chunk=17)["reused"]
    assert path.stat().st_mtime_ns == stamp
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not list(streamed.glob("*.part"))


def test_failed_batch_never_publishes_cache(helper, datasets, monkeypatch):
    from radis.api import exomolapi

    directory = datasets[1]
    raw, states, definition, partition = inputs(directory)
    original = exomolapi.pickup_gE
    count = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted conversion")
        return original(*args, **kwargs)

    monkeypatch.setattr(exomolapi, "pickup_gE", fail_second_batch)
    with pytest.raises(RuntimeError, match="interrupted conversion"):
        helper.build_transition_cache(raw, states, definition, partition, rows_per_chunk=10)
    assert not raw.with_suffix(".h5").exists()
    assert not list(directory.glob("*.part"))


def test_mixed_wavenumbers_are_rejected_before_publication(helper, datasets):
    directory = datasets[1]
    raw, states, definition, partition = inputs(directory)
    with bz2.open(raw, "rt") as file:
        lines = [file.readline().split(), file.readline().split()]
    with bz2.open(raw, "wt") as file:
        file.write(" ".join(lines[0][:3]) + "\n" + " ".join(lines[1]) + "\n")
    with pytest.raises(ValueError, match="Mixed missing/present"):
        helper.build_transition_cache(raw, states, definition, partition, rows_per_chunk=1)
    assert not raw.with_suffix(".h5").exists()
    assert not list(directory.glob("*.part"))


def test_interrupted_raw_download_is_not_published(helper, datasets, monkeypatch):
    from exojax.database.exomol import _pyexocross_download

    directory = datasets[1]
    raw = directory / "12C-16O__SAMPLE.trans.bz2"
    raw.unlink()

    class BrokenDownload(io.BytesIO):
        headers = {}

        def read(self, size=-1):
            if self.tell():
                raise OSError("interrupted download")
            return super().read(size)

    monkeypatch.setattr(_pyexocross_download, "urlopen", lambda *args, **kwargs: BrokenDownload(b"BZh9partial"))
    args = helper.parser_with_defaults().parse_args(["--species", "C1O1", "--mdb-path", str(directory)])
    with pytest.raises(OSError, match="interrupted download"):
        helper.prepare(args)
    assert not raw.exists()
    assert not raw.with_suffix(".h5").exists()
    assert not list(directory.glob("*.part"))


def test_dry_run_does_not_create_database(helper, tmp_path):
    root = tmp_path / "not_created"
    assert helper.main(["--database-root", str(root), "--dry-run"]) == 0
    assert not root.exists()


def test_empty_transition_source_is_not_published(helper, datasets):
    directory = datasets[1]
    raw, states, definition, partition = inputs(directory)
    raw.write_bytes(bz2.compress(b""))
    with pytest.raises(ValueError, match="contains no rows"):
        helper.build_transition_cache(raw, states, definition, partition, rows_per_chunk=10)
    assert not raw.with_suffix(".h5").exists()
    assert not list(directory.glob("*.part"))
