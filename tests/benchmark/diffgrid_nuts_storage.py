"""Internal, offline storage helpers for the DiffGrid NUTS benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(path, lambda stream: stream.write(data))


def write_npz(path: Path, **arrays) -> None:
    """Write an archive atomically without silently adding a filename suffix."""
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays cannot be saved in benchmark archives.")
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def read_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(Path(path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"Benchmark metadata must be a JSON object: {path}")
    version = metadata.get("schema_version")
    if type(version) is not int or version not in (1, SCHEMA_VERSION):
        raise ValueError(f"Unsupported benchmark schema version {version!r}: {path}")
    if version == SCHEMA_VERSION and metadata.get("status") != "completed":
        raise ValueError(
            f"Benchmark artifact is not completed ({metadata.get('status')!r}): {path}"
        )
    if version == 1:
        missing = [
            name for name in ("provenance", "status", "samples") if name not in metadata
        ]
        for name in missing:
            metadata[name] = None
        metadata["legacy_missing_fields"] = missing
        metadata["legacy_notice"] = (
            "Schema 1 did not record provenance, completion status, or raw samples; "
            "missing information cannot be reconstructed."
        )
    return metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in (".", "..")
        or any(character in value for character in ("/", "\\", "\0"))
    ):
        raise ValueError(
            "Run ID must be one nonempty path component without separators."
        )
    return value


def result_paths(output_dir: Path, method: str, run_id: str | None = None):
    if method not in ("premodit", "diffgrid"):
        raise ValueError(f"Unknown benchmark method: {method!r}")
    output_dir = Path(output_dir)
    if run_id is None:
        return {
            "result": output_dir / f"{method}.json",
            "samples": output_dir / f"{method}_samples.npz",
        }
    directory = output_dir / "runs" / validate_run_id(run_id) / method
    return {"result": directory / "result.json", "samples": directory / "samples.npz"}


def reserve_result(output_dir: Path, method: str, run_id: str | None = None):
    """Reserve a named method run; retain legacy output-directory overwrite behavior."""
    paths = result_paths(output_dir, method, run_id)
    paths["result"].parent.mkdir(parents=True, exist_ok=run_id is None)
    return paths


def _sample_arrays(samples, extra_fields, parameter_order):
    order = list(parameter_order)
    if (
        not samples
        or any(not isinstance(name, str) for name in order)
        or len(order) != len(set(order))
        or set(order) != set(samples)
    ):
        raise ValueError("Parameter order must contain every sample name exactly once.")
    arrays = {}
    groups = {"samples": {}, "extra_fields": {}}
    chain_shape = None
    for group, values in (("samples", samples), ("extra_fields", extra_fields)):
        names = order if group == "samples" else list(values)
        for index, name in enumerate(names):
            if not isinstance(name, str):
                raise ValueError("Sample and extra-field names must be strings.")
            array = np.asarray(values[name])
            if array.dtype.hasobject or array.ndim < 2:
                raise ValueError(
                    f"{group}/{name} requires a non-object (chain, draw, ...) array."
                )
            if chain_shape is None:
                chain_shape = array.shape[:2]
                if min(chain_shape) < 1:
                    raise ValueError(
                        "Sample chain and draw dimensions must be nonempty."
                    )
            if array.shape[:2] != chain_shape:
                raise ValueError(
                    f"{group}/{name} has inconsistent chain/draw dimensions."
                )
            key = f"{group}_{index}"
            arrays[key] = array
            groups[group][name] = {
                "archive_key": key,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
    return arrays, groups, order, list(chain_shape)


def save_samples(path: Path, samples, extra_fields, parameter_order):
    arrays, groups, order, chain_shape = _sample_arrays(
        samples, extra_fields, parameter_order
    )
    write_npz(path, **arrays)
    return {
        "filename": Path(path).name,
        "sha256": sha256(path),
        "parameter_order": order,
        "chain_shape": chain_shape,
        **groups,
    }


def load_samples(path: Path, manifest: dict[str, Any]):
    """Restore raw chains only after validating the archive and its array contract."""
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Sample archive digest does not match result metadata.")
    restored = {"samples": {}, "extra_fields": {}}
    expected_keys = []
    with np.load(path, allow_pickle=False) as archive:
        for group, values in restored.items():
            for name, entry in manifest[group].items():
                key = entry["archive_key"]
                expected_keys.append(key)
                if key not in archive:
                    raise ValueError(f"Missing sample archive field: {key}")
                array = archive[key]
                if (
                    list(array.shape) != entry["shape"]
                    or array.dtype.str != entry["dtype"]
                ):
                    raise ValueError(
                        f"Sample shape or dtype does not match metadata: {name}"
                    )
                values[name] = array
        if len(expected_keys) != len(set(expected_keys)) or set(archive.files) != set(
            expected_keys
        ):
            raise ValueError("Sample archive fields do not match result metadata.")
    _, _, _, chain_shape = _sample_arrays(
        restored["samples"], restored["extra_fields"], manifest["parameter_order"]
    )
    if chain_shape != manifest["chain_shape"]:
        raise ValueError("Sample chain/draw dimensions do not match result metadata.")
    return restored["samples"], restored["extra_fields"]


def _json_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def collect_provenance(repo_root: Path, input_paths, settings: dict[str, Any]):
    """Record executed inputs and revision identity without revisiting large databases."""
    repo_root = Path(repo_root).resolve()

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout

    git_info = {
        "commit": None,
        "dirty": None,
        "status": None,
        "tracked_diff_sha256": None,
        "unavailable_reason": None,
    }
    tracked = None
    try:
        git_info["commit"] = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain=v1", "--untracked-files=all").decode()
        git_info.update(
            dirty=bool(status),
            status=status,
            tracked_diff_sha256=hashlib.sha256(
                git("diff", "--binary", "HEAD", "--")
            ).hexdigest(),
        )
        tracked = {
            os.fsdecode(path) for path in git("ls-files", "-z").split(b"\0") if path
        }
    except (OSError, subprocess.CalledProcessError) as error:
        git_info["unavailable_reason"] = str(error)
    git_info["fixed_head_reproducible"] = (
        git_info["commit"] is not None
        and git_info["dirty"] is False
        and git_info["unavailable_reason"] is None
    )

    paths = {
        Path(path).resolve(): "execution_code"
        if Path(path).suffix == ".py"
        else "input"
        for path in input_paths
    }
    # Hash the common source tree so lazy method-specific imports cannot make
    # identical revisions appear to use different code.
    for path in (repo_root / "src" / "exojax").rglob("*.py"):
        paths.setdefault(path.resolve(), "source_code")
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (name == "exojax" or name.startswith("exojax.")) and filename:
            path = Path(filename).resolve()
            if path.suffix == ".py":
                paths.setdefault(path, "imported_code")
    files = []
    for path, role in sorted(paths.items(), key=lambda item: str(item[0])):
        try:
            relative = str(path.relative_to(repo_root))
        except ValueError:
            relative = str(path)
        entry = {
            "path": relative,
            "role": role,
            "sha256": None,
            "tracked": relative in tracked if tracked is not None else None,
            "unavailable_reason": None,
        }
        try:
            entry["sha256"] = sha256(path)
        except OSError as error:
            entry["unavailable_reason"] = str(error)
        files.append(entry)
    dependencies = {}
    for name in (
        "exojax",
        "numpy",
        "jax",
        "jaxlib",
        "numpyro",
        "arviz",
        "scipy",
        "radis",
        "numba",
        "pandas",
        "h5py",
        "tables",
        "json-tricks",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("JAX_", "XLA_"))
        or key
        in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NUMBA_DISABLE_JIT")
    }
    code_files = [entry for entry in files if entry["role"] != "input"]
    code_sha256 = _json_hash(
        [{key: entry[key] for key in ("path", "sha256")} for entry in code_files]
    )
    input_sha256 = _json_hash(
        [
            {key: entry[key] for key in ("path", "sha256")}
            for entry in files
            if entry["role"] == "input"
        ]
    )
    git_info["fixed_head_reproducible"] = git_info["fixed_head_reproducible"] and all(
        entry["tracked"] is True and entry["sha256"] is not None for entry in code_files
    )
    git_info["reproducibility_scope"] = (
        "Execution code only; prepared artifacts and external inputs are recorded separately."
    )
    settings_sha256 = _json_hash(settings)
    return {
        "git": git_info,
        "files": files,
        "code_sha256": code_sha256,
        "input_sha256": input_sha256,
        "settings_sha256": settings_sha256,
        "execution_sha256": _json_hash(
            {
                "code": code_sha256,
                "inputs": input_sha256,
                "settings": settings_sha256,
            }
        ),
        "settings": settings,
        "dependencies": dependencies,
        "environment": environment,
    }
