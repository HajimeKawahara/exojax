import hashlib
import io
import json

import pytest

from exojax.provider import starter


DATASET = "co-diffgrid-v1"
BASE_URL = "https://example.test/opacity"


@pytest.fixture
def server(monkeypatch):
    payloads = {
        "opacity.npz": b"opacity data" * 100_000,
        "metadata.json": b'{"molecule": "CO"}',
    }
    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "citation": "Example source citation",
        "files": [
            {
                "name": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in payloads.items()
        ],
    }
    resources = dict(payloads)
    calls = []

    def urlopen(url, timeout):
        assert timeout > 0
        assert url.startswith(f"{BASE_URL}/{DATASET}/")
        name = url.removeprefix(f"{BASE_URL}/{DATASET}/")
        calls.append(name)
        if name == "manifest.json":
            return io.BytesIO(json.dumps(manifest).encode())
        data = resources[name]
        return data() if callable(data) else io.BytesIO(data)

    monkeypatch.setattr(starter, "urlopen", urlopen)
    return manifest, payloads, resources, calls


def test_download_and_offline_cache_reuse(tmp_path, server, monkeypatch):
    manifest, payloads, _, calls = server
    directory = starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)

    assert directory == tmp_path / DATASET
    assert json.loads((directory / "manifest.json").read_text()) == manifest
    for name, data in payloads.items():
        assert (directory / name).read_bytes() == data
    assert calls == ["manifest.json", *payloads]

    def offline(*args, **kwargs):
        pytest.fail("A complete cache must work offline")

    monkeypatch.setattr(starter, "urlopen", offline)
    assert starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL) == directory


@pytest.mark.parametrize("damage", ["missing", "size", "checksum"])
def test_only_missing_or_corrupt_payload_is_downloaded(tmp_path, server, damage):
    _, payloads, _, calls = server
    directory = starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    path = directory / "opacity.npz"
    if damage == "missing":
        path.unlink()
    elif damage == "size":
        path.write_bytes(b"truncated")
    else:
        path.write_bytes(b"x" * len(payloads[path.name]))
    calls.clear()

    starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)

    assert path.read_bytes() == payloads[path.name]
    assert calls == [path.name]


@pytest.mark.parametrize("failure", ["truncated", "oversized", "checksum", "interrupted"])
def test_failed_download_is_not_installed_and_retry_succeeds(tmp_path, server, failure):
    _, payloads, resources, calls = server
    name = "metadata.json"
    if failure == "truncated":
        resources[name] = payloads[name][:-1]
    elif failure == "oversized":
        resources[name] = payloads[name] + b"x"
    elif failure == "checksum":
        resources[name] = b"x" * len(payloads[name])
    else:
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection interrupted")
                return super().read(5)

        resources[name] = lambda: Interrupted(payloads[name])

    with pytest.raises((OSError, ValueError)):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)

    directory = tmp_path / DATASET
    assert (directory / "opacity.npz").read_bytes() == payloads["opacity.npz"]
    assert not (directory / name).exists()
    assert not (directory / "manifest.json").exists()
    assert not list(directory.glob("*.part"))

    resources[name] = payloads[name]
    calls.clear()
    starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    assert calls == ["manifest.json", name]
    assert (directory / name).read_bytes() == payloads[name]
    assert (directory / "manifest.json").is_file()


def test_failed_repair_preserves_cached_manifest_and_payload(tmp_path, server):
    _, payloads, resources, _ = server
    directory = starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    old_manifest = (directory / "manifest.json").read_bytes()
    path = directory / "opacity.npz"
    path.write_bytes(b"old corrupt cache")
    resources[path.name] = b"incorrect remote data"

    with pytest.raises(ValueError, match="mismatch"):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)

    assert path.read_bytes() == b"old corrupt cache"
    assert (directory / "manifest.json").read_bytes() == old_manifest
    assert (directory / "metadata.json").read_bytes() == payloads["metadata.json"]
    assert not list(directory.glob("*.part"))


@pytest.mark.parametrize(
    "dataset",
    ["../escape", "a/b", r"a\b", "", ".", "..", "a?query", "a#fragment", "/absolute"],
)
def test_rejects_unsafe_dataset_before_network_access(tmp_path, monkeypatch, dataset):
    def unexpected_network(*args, **kwargs):
        pytest.fail("Invalid dataset must be rejected before network access")

    monkeypatch.setattr(starter, "urlopen", unexpected_network)
    with pytest.raises(ValueError, match="dataset"):
        starter.fetch_starter_opacity(dataset, tmp_path, base_url=BASE_URL)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "filename",
    [
        "../escape", "/absolute", "a/b", r"a\b", "..", "", "manifest.json",
        "Manifest.json", "a?query", "a#fragment",
    ],
)
def test_rejects_unsafe_manifest_filename(tmp_path, server, filename):
    manifest, _, _, calls = server
    manifest["files"][0]["name"] = filename
    with pytest.raises(ValueError, match="filename"):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    assert calls == ["manifest.json"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"dataset": "another-dataset"},
        {"files": []},
        {"files": [None]},
    ],
)
def test_rejects_invalid_manifest(tmp_path, server, change):
    manifest, _, _, calls = server
    manifest.update(change)
    with pytest.raises(ValueError):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    assert calls == ["manifest.json"]


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "bad"}, {"size_bytes": -1}, {"size_bytes": True},
        {"size_bytes": 1.5},
    ],
)
def test_rejects_invalid_file_metadata(tmp_path, server, change):
    manifest, _, _, calls = server
    manifest["files"][0].update(change)
    with pytest.raises(ValueError, match="size or SHA256"):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    assert calls == ["manifest.json"]


def test_rejects_duplicate_manifest_filenames(tmp_path, server):
    manifest, _, _, calls = server
    manifest["files"].append(manifest["files"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        starter.fetch_starter_opacity(DATASET, tmp_path, base_url=BASE_URL)
    assert calls == ["manifest.json"]
