"""Download small, versioned opacity datasets for the introductory examples."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


DEFAULT_BASE_URL = "https://secondearths.sakura.ne.jp/exojax/data/opacity"
_CHUNK_SIZE = 1024 * 1024


def _validate_manifest(manifest, dataset):
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("dataset") != dataset
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ValueError(f"Invalid starter opacity manifest for {dataset!r}")

    names = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest file must be an object")
        name = entry.get("name")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or name.lower() == "manifest.json"
            or name in names
        ):
            raise ValueError(f"Invalid or duplicate manifest filename: {name!r}")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise ValueError(f"Invalid size or SHA256 for {name!r}")
        names.add(name)


def _cached_file_matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def _download_file(url, path, entry):
    temporary = None
    try:
        with urlopen(url, timeout=120) as response, tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: response.read(_CHUNK_SIZE), b""):
                size += len(chunk)
                if size > entry["size_bytes"]:
                    raise ValueError(f"Downloaded size exceeds manifest for {path.name}")
                digest.update(chunk)
                handle.write(chunk)
            if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Downloaded size or SHA256 mismatch for {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_starter_opacity(dataset, cache_dir=None, *, base_url=DEFAULT_BASE_URL):
    """Fetch and verify a starter dataset, returning its local directory.

    Args:
        dataset: Immutable dataset ID, for example ``"h2o-ckd-v1"`` or
            ``"co-diffgrid-v1"``. Updated tables must use a new ID.
        cache_dir: Cache root. Defaults to ``~/.cache/exojax/opacity``.
        base_url: HTTP(S) directory containing the dataset directories.

    The server must provide ``{base_url}/{dataset}/manifest.json`` with
    ``schema_version=1``, a matching ``dataset``, and a ``files`` list of
    ``name``, ``size_bytes``, and lowercase ``sha256`` entries. Other metadata
    is preserved. Filenames must be flat, without directory components.

    Valid cached files are reused without network access. Missing or corrupt
    files are downloaded to temporary files and installed only after their
    size and SHA256 match the manifest. The manifest is saved after all files
    pass verification; immutable dataset IDs allow its subsequent offline use.
    Network errors propagate, and malformed manifests or payloads raise
    ``ValueError``. A changed server copy cannot silently replace cached data.
    """
    if (
        not isinstance(dataset, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", dataset) is None
    ):
        raise ValueError(
            "dataset must be a lowercase name containing letters, digits, '-' or '_'"
        )
    parsed_url = urlsplit(base_url)
    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(
            "base_url must be an HTTP(S) directory URL without a query or fragment"
        )

    root = Path(cache_dir) if cache_dir is not None else Path("~/.cache/exojax/opacity")
    directory = root.expanduser() / dataset
    manifest_path = directory / "manifest.json"
    dataset_url = f"{base_url.rstrip('/')}/{dataset}"
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
    else:
        with urlopen(f"{dataset_url}/manifest.json", timeout=120) as response:
            manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    _validate_manifest(manifest, dataset)

    directory.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        path = directory / entry["name"]
        if not _cached_file_matches(path, entry):
            _download_file(f"{dataset_url}/{entry['name']}", path, entry)

    if not manifest_path.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".part", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(manifest_bytes)
            os.replace(temporary, manifest_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory
