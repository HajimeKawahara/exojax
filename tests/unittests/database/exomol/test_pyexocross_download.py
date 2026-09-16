"""Offline checks for dataset-specific ExoMol downloads."""

import bz2
from io import BytesIO
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError

import pytest

from exojax.database.exomol import _pyexocross_download as download


class Response(BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def _definition(count=1, maximum=2000):
    return (
        f"{count} # No. of transition files\n"
        f"{maximum} # Maximum wavenumber (in cm-1)\n"
    ).encode()


def _requests(monkeypatch, responses):
    requested = []

    def open_url(url, timeout):
        assert timeout == 60
        requested.append(url)
        response = responses[url]
        if isinstance(response, Exception):
            raise response
        return Response(response)

    monkeypatch.setattr(download, "urlopen", open_url)
    return requested


def _local_dataset(tmp_path, count=1, maximum=2000):
    path = tmp_path / "CO" / "12C-16O" / "Li2015"
    path.mkdir(parents=True)
    stem = "12C-16O__Li2015"
    (path / f"{stem}.def").write_bytes(_definition(count, maximum))
    (path / f"{stem}.pf").write_text("296 1\n1000 2\n")
    (path / f"{stem}.states.bz2").write_bytes(bz2.compress(b"1 0 1 0\n"))
    return path, stem


def test_downloads_only_named_dataset_and_requested_broadener(tmp_path, monkeypatch):
    path = tmp_path / "CO" / "12C-16O" / "Li2015"
    stem = "12C-16O__Li2015"
    base = "https://www.exomol.com/db/CO/12C-16O/"
    responses = {
        base + f"Li2015/{stem}.def": _definition(),
        base + f"Li2015/{stem}.pf": b"296 1\n1000 2\n",
        base + f"Li2015/{stem}.states.bz2": bz2.compress(b"1 0 1 0\n"),
        base + f"Li2015/{stem}.trans.bz2": bz2.compress(b"2 1 1\n"),
        base + "12C-16O__He.broad": b"a0 0.07 0.5 0\n",
    }
    requested = _requests(monkeypatch, responses)
    assert download.ensure_exomol_files(path, [100, 200], bkgdatm="He") == path
    assert set(requested) == set(responses)
    assert (path / "12C-16O__He.broad").read_bytes() == responses[base + "12C-16O__He.broad"]
    assert not list(path.glob("*.part"))


def test_segment_download_uses_overlap_and_reuses_existing_files(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path, count=4, maximum=4000)
    existing = path / f"{stem}__01000-02000.trans.bz2"
    existing.write_bytes(bz2.compress(b"2 1 1\n"))
    filename = f"{stem}__02000-03000.trans.bz2"
    url = f"https://www.exomol.com/db/CO/12C-16O/Li2015/{filename}"
    requested = _requests(monkeypatch, {url: bz2.compress(b"3 1 1\n")})
    download.ensure_exomol_files(path, [1500, 2500], broadf_download=False)
    assert requested == [url]
    assert (path / filename).exists()


@pytest.mark.parametrize("molecule,isotope", [("CO", "12C-16O"), ("H2O", "1H2-16O")])
def test_bundled_range_is_complete_offline(tmp_path, monkeypatch, molecule, isotope):
    from exojax.test.data import get_testdata_filename

    source = Path(get_testdata_filename(molecule)) / isotope / "SAMPLE"
    target = tmp_path / molecule / isotope / "SAMPLE"
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.name.endswith((".def", ".pf", ".bz2", ".broad")):
            shutil.copy2(file, target / file.name)
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(target, [4330, 4360], broadf_download=False)
    assert not requested


@pytest.mark.parametrize("options", [{"broadf_download": False}, {"broadf": False}])
def test_disabled_broadening_never_requests_missing_file(tmp_path, monkeypatch, options):
    path, stem = _local_dataset(tmp_path)
    (path / f"{stem}.trans.bz2").write_bytes(bz2.compress(b"2 1 1\n"))
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(path, [100, 200], **options)
    assert not requested


def test_absent_broadener_can_use_definition_defaults(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path)
    (path / f"{stem}.trans.bz2").write_bytes(bz2.compress(b"2 1 1\n"))
    url = "https://www.exomol.com/db/CO/12C-16O/12C-16O__H2.broad"
    requested = _requests(monkeypatch, {url: HTTPError(url, 404, "Not found", {}, None)})
    download.ensure_exomol_files(path, [100, 200])
    assert requested == [url]
    assert not (path / "12C-16O__H2.broad").exists()


def test_required_file_failure_is_not_silenced(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path)
    url = f"https://www.exomol.com/db/CO/12C-16O/Li2015/{stem}.trans.bz2"
    _requests(monkeypatch, {url: HTTPError(url, 404, "Not found", {}, None)})
    with pytest.raises(HTTPError):
        download.ensure_exomol_files(path, [100, 200], broadf_download=False)
    assert not (path / f"{stem}.trans.bz2").exists()


@pytest.mark.parametrize("payload", [b"", b"<html>missing</html>", b"not compressed"])
def test_bad_download_never_leaves_a_cached_file(tmp_path, monkeypatch, payload):
    target = tmp_path / "data.trans.bz2"
    _requests(monkeypatch, {"https://example.test/data": payload})
    with pytest.raises(ValueError):
        download._download_file("https://example.test/data", target)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_interrupted_download_removes_partial_file(tmp_path, monkeypatch):
    class InterruptedResponse(Response):
        def read(self, size):
            if self.tell():
                raise OSError("Connection interrupted")
            return super().read(size)

    monkeypatch.setattr(download, "urlopen", lambda *a, **k: InterruptedResponse(b"BZhpartial"))
    target = tmp_path / "data.trans.bz2"
    with pytest.raises(OSError, match="interrupted"):
        download._download_file("https://example.test/data", target)
    assert not list(tmp_path.iterdir())


def test_truncated_download_is_not_cached(tmp_path, monkeypatch):
    response = Response(b"BZhpartial")
    response.headers["Content-Length"] = "1000"
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="Incomplete"):
        download._download_file("https://example.test/data", tmp_path / "data.trans.bz2")
    assert not list(tmp_path.iterdir())


def test_uncompressed_local_files_are_reused(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path)
    (path / f"{stem}.states.bz2").unlink()
    (path / f"{stem}.states").write_text("1 0 1 0\n")
    (path / f"{stem}.trans").write_text("2 1 1\n")
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(path, [100, 200], broadf_download=False)
    assert not requested


def test_json_definition_is_reused_without_text_download(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path, count=4, maximum=4000)
    (path / f"{stem}.def").unlink()
    (path / f"{stem}.def.json").write_text(json.dumps({
        "dataset": {"transitions": {"number_of_transition_files": 4, "max_wavenumber": 4000}}
    }))
    (path / f"{stem}__01000-02000.trans.bz2").write_bytes(bz2.compress(b"2 1 1\n"))
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(path, [1500, 1600], broadf_download=False)
    assert not requested


def test_json_definition_takes_precedence_over_text_coverage(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path, count=1)
    (path / f"{stem}.def.json").write_text(json.dumps({
        "dataset": {"transitions": {"number_of_transition_files": 4, "max_wavenumber": 4000}}
    }))
    (path / f"{stem}__01000-02000.trans.bz2").write_bytes(bz2.compress(b"2 1 1\n"))
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(path, [1500, 1600], broadf_download=False)
    assert not requested


def test_isotope_directory_broadening_is_reused_offline(tmp_path, monkeypatch):
    path, stem = _local_dataset(tmp_path)
    (path / f"{stem}.trans.bz2").write_bytes(bz2.compress(b"2 1 1\n"))
    broad_file = path.parent / "12C-16O__He.broad"
    broad_file.write_text("a0 0.07 0.5 0\n")
    requested = _requests(monkeypatch, {})
    download.ensure_exomol_files(path, [100, 200], bkgdatm="He")
    assert not requested
    assert not (path / broad_file.name).exists()


@pytest.mark.parametrize("nurange", [[float("nan"), 200], [100, float("nan")]])
def test_nan_range_fails_before_creating_or_requesting_files(tmp_path, monkeypatch, nurange):
    path = tmp_path / "CO" / "12C-16O" / "Li2015"
    requested = _requests(monkeypatch, {})
    with pytest.raises(ValueError, match="without NaN"):
        download.ensure_exomol_files(path, nurange)
    assert not requested
    assert not list(tmp_path.iterdir())


def test_unbounded_range_selects_all_segments(tmp_path):
    path, stem = _local_dataset(tmp_path, count=2, maximum=2000)
    assert download._transition_names(path / f"{stem}.def", stem, [-float("inf"), float("inf")]) == [
        f"{stem}__00000-01000.trans.bz2", f"{stem}__01000-02000.trans.bz2"
    ]


def test_vtt_irregular_segments_are_preserved(tmp_path):
    definition = tmp_path / "1H-2H-16O__VTT.def"
    definition.write_bytes(_definition(count=16, maximum=26000))
    assert download._transition_names(definition, "1H-2H-16O__VTT", [2300, 2400]) == [
        "1H-2H-16O__VTT__02250-02750.trans.bz2"
    ]
