"""Stream pinned prepared Diffgrid chunks through the existing P7 transfer.

Only coordinates are loaded when assembling the RCE column. Coefficient
archives are verified and loaded one spectral chunk at a time after the RCE
solve. This host orchestration is a primal forward path, not a retrieval AD
interface. The complete emergent grid is retained before applying P1 or a
Gaussian response, including response stencils crossing chunk boundaries.
"""

from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from _rce_observation import pixel_flux_ratio
from _rce_wasp18b import response_samples
from _rce_wasp18b_assets import _sha256
from exojax.opacity import OpaDiffgrid
from exojax.opacity.diffgrid.io import DIFFGRID_SCHEMA_VERSION, _UNITS
from exojax.opacity.io.serialization import (
    _ensure_compatible_versions, _sha256_array, _validate_schema,
)


COORDINATES = ("nu_grid", "pressure_grid", "temperature_grid",
               "inverse_temperature_grid", "log_cross_section_floor")
COEFFICIENTS = ("log_cross_section_grid", "log_cross_section_derivative_grid")


def pressure_interpolation(source, target):
    """Return fixed log-pressure weights without extrapolation or node drift."""
    source, target = np.asarray(source), np.asarray(target)
    for name, grid in (("source", source), ("target", target)):
        if (grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid) & (grid > 0))
                or np.any(np.diff(grid) <= 0)):
            raise ValueError(f"Use at least two finite increasing positive {name} pressures")
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("Pressure resampling forbids extrapolation outside the source centers")
    upper = np.minimum(np.searchsorted(source, target, side="left"), source.size - 1)
    exact = source[upper] == target
    lower = np.where(exact, upper, upper - 1)
    interval = np.log(source[upper]) - np.log(source[lower])
    weight = np.where(exact, 0., (np.log(target) - np.log(source[lower])) / np.where(exact, 1., interval))
    return lower, upper, weight


def resample_pressure_values(values, mapping):
    """Apply identical linear weights to log cross sections or their slopes."""
    lower, upper, weight = mapping
    shape = (len(weight),) + (1,) * (values.ndim - 1)
    # Exact native nodes select the same row twice, preserving its values.
    return values[lower] + jnp.asarray(weight).reshape(shape) * (values[upper] - values[lower])


def _pressure_indices(indices, size):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.size < 2 or indices.dtype.kind not in "iu"
            or indices[0] < 0 or indices[-1] >= size or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("Choose at least two increasing native pressure indices")
    return indices


def _entry_path(entry, parent):
    path = (parent / Path(entry["path"]).expanduser()).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing streaming asset: {path}")
    digest = entry.get("sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"Invalid streaming SHA256: {path}")
    return path


def _verify(path, digest):
    if _sha256(path) != digest:
        raise ValueError(f"SHA256 mismatch for streaming asset: {path}")


def _read_coordinates(path, metadata_path):
    """Validate pinned NPZ coordinate arrays without unpacking coefficients."""
    meta = json.loads(metadata_path.read_text())
    _validate_schema(meta["schema_version"], False)
    _ensure_compatible_versions(meta["exojax_version"])
    if (meta["diffgrid_schema_version"] != DIFFGRID_SCHEMA_VERSION
            or meta["opa_type"] != "OpaDiffgrid" or meta["units"] != _UNITS):
        raise ValueError("Unsupported streaming Diffgrid schema, type, or units")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in COORDINATES}
        names = set(archive.files)
    required = set(COORDINATES + COEFFICIENTS)
    if not required <= names or names - required - {"wav"}:
        raise ValueError("Unexpected streaming Diffgrid array names")
    for field in ("array_shapes", "array_dtypes", "array_digests"):
        if set(meta[field]) != names:
            raise ValueError("Streaming Diffgrid metadata and payload names differ")
    # NPY headers are small even when a coefficient array spans gigabytes.
    with zipfile.ZipFile(path) as archive:
        for name in names:
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported streaming NPY header version")
            if list(shape) != meta["array_shapes"][name] or str(dtype) != meta["array_dtypes"][name]:
                raise ValueError("Streaming Diffgrid shape or dtype metadata mismatch")
            if name in required and dtype != np.dtype("float64"):
                raise ValueError("P7 streaming requires float64 Diffgrid arrays")
    for name, value in arrays.items():
        if _sha256_array(value) != meta["array_digests"][name]:
            raise ValueError(f"Streaming coordinate digest mismatch: {name}")
    nu, pressure, temperature, inverse, floor = (arrays[name] for name in COORDINATES)
    for name, value, minimum in (("nu_grid", nu, 2), ("pressure_grid", pressure, 1),
                                 ("temperature_grid", temperature, 2),
                                 ("inverse_temperature_grid", inverse, 2)):
        if value.ndim != 1 or value.size < minimum or not np.all(np.isfinite(value) & (value > 0)):
            raise ValueError(f"Invalid streaming coordinate: {name}")
    if (np.any(np.diff(nu) <= 0) or np.any(np.diff(pressure) <= 0)
            or np.any(np.diff(temperature) >= 0) or np.any(np.diff(inverse) <= 0)
            or inverse.shape != temperature.shape
            or not np.allclose(inverse * temperature, 1., rtol=1e-12, atol=0)):
        raise ValueError("Streaming coordinates must be ordered and reciprocal")
    expected = [pressure.size, temperature.size, nu.size]
    if any(meta["array_shapes"][name] != expected for name in COEFFICIENTS):
        raise ValueError("Streaming coefficient dimensions disagree with coordinates")
    if (floor.shape != () or not np.isfinite(floor)
            or not np.log(np.finfo(float).tiny) <= floor <= np.log(np.finfo(float).max)):
        raise ValueError("Invalid streaming cross-section floor")
    return arrays, meta


class MetadataOnlyDiffgrid:
    """Coordinate contract for a column whose resolved calculation is deferred."""

    def __init__(self, arrays):
        self.nu_grid = arrays["nu_grid"]
        self.diffgrid_info = SimpleNamespace(**{key: value for key, value in arrays.items()
                                                if key != "nu_grid"})

    def check_pressure_grid(self, pressure):
        expected = self.diffgrid_info.pressure_grid
        if np.shape(pressure) != expected.shape or not np.allclose(pressure, expected, rtol=1e-7, atol=0):
            raise ValueError("Streaming Diffgrid pressure grid differs from the column")

    def xsmatrix(self, temperature):
        raise RuntimeError("Use the streaming resolved path with metadata-only opacities")

    def pressure_subset(self, indices):
        """Select stored pressure coordinates without interpolation."""
        indices = _pressure_indices(indices, len(self.diffgrid_info.pressure_grid))
        arrays = dict(vars(self.diffgrid_info), nu_grid=self.nu_grid)
        arrays["pressure_grid"] = arrays["pressure_grid"][indices]
        return MetadataOnlyDiffgrid(arrays)

    def pressure_resample(self, pressure):
        """Expose target coordinates; native archives stay unchanged."""
        pressure_interpolation(self.diffgrid_info.pressure_grid, pressure)
        arrays = dict(vars(self.diffgrid_info), nu_grid=self.nu_grid)
        arrays["pressure_grid"] = np.asarray(pressure).copy()
        return MetadataOnlyDiffgrid(arrays)

    def temperature_subset(self, indices):
        """Select stored inverse-temperature knots and retain their derivatives."""
        indices = _temperature_indices(indices, len(self.diffgrid_info.temperature_grid))
        arrays = dict(vars(self.diffgrid_info), nu_grid=self.nu_grid)
        for key in ("temperature_grid", "inverse_temperature_grid"):
            arrays[key] = arrays[key][indices]
        return MetadataOnlyDiffgrid(arrays)


def _temperature_indices(indices, size):
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.size < 2 or indices.dtype.kind not in "iu"
            or indices[0] < 0 or indices[-1] >= size or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("Choose at least two increasing native temperature indices")
    return indices


@dataclass(frozen=True)
class ChunkArchive:
    path: Path
    metadata_path: Path
    sha256: str
    metadata_sha256: str


class StreamingResolvedPlan:
    """Validated coordinates and immutable pins for matching spectral chunks."""

    def __init__(self, species, chunks, nu, grids, provenance, pressure, temperature=None):
        self.species, self.chunks = tuple(species), chunks
        self.nu, self.grids, self.provenance = nu, grids, provenance
        self.source_pressure = np.asarray(pressure)
        self.pressure_indices = np.arange(len(pressure))
        self.target_pressure, self.pressure_mapping = None, None
        self.source_temperature = None if temperature is None else np.asarray(temperature)
        self.temperature_indices = None
        self._column, self._calculate = None, None

    def temperature_subset(self, indices):
        """Select original Hermite knots at read time; source archives stay pinned."""
        if self.source_temperature is None:
            raise ValueError("Streaming plan has no native temperature coordinates")
        original = (np.arange(len(self.source_temperature)) if self.temperature_indices is None
                    else self.temperature_indices)
        indices = _temperature_indices(indices, len(original))
        subset = copy(self)
        subset.temperature_indices = original[indices]
        subset.provenance = dict(self.provenance, temperature_subset=dict(
            source_temperature_K=self.source_temperature.tolist(),
            source_temperature_sha256=_sha256_array(self.source_temperature),
            source_archive_indices=subset.temperature_indices.tolist(),
            selected_temperature_K=self.source_temperature[subset.temperature_indices].tolist(),
            policy="Select original log cross sections and inverse-temperature derivatives; use the existing Hermite interpolation and final clip",
            scope="Held-out native temperature-node restoration; accuracy between all native nodes is not certified",
        ), all_chunk_coefficients_verified=False)
        subset._column, subset._calculate = None, None
        return subset

    def pressure_subset(self, indices):
        """Keep every source pin; slice only after strict archive validation."""
        if self.target_pressure is not None:
            indices = _pressure_indices(indices, len(self.target_pressure))
            return self.pressure_resample(self.target_pressure[indices])
        indices = _pressure_indices(indices, len(self.pressure_indices))
        subset = copy(self)
        subset.pressure_indices = self.pressure_indices[indices]
        subset.provenance = dict(self.provenance,
            source_pressure_indices=subset.pressure_indices.tolist(),
            pressure_subset_policy="Exact stored pressure slices; no interpolation",
            all_chunk_coefficients_verified=False)
        subset._column, subset._calculate = None, None
        return subset

    def pressure_resample(self, pressure):
        """Interpolate the selected native nodes without changing archive pins."""
        source = self.source_pressure[self.pressure_indices]
        mapping = pressure_interpolation(source, pressure)
        refined = copy(self)
        refined.target_pressure = np.asarray(pressure).copy()
        refined.pressure_mapping = (self.pressure_indices[mapping[0]],
                                    self.pressure_indices[mapping[1]], mapping[2])
        refined.provenance = {key: value for key, value in self.provenance.items()
                              if key not in ("source_pressure_indices", "pressure_subset_policy")}
        refined.provenance.update(
            pressure_resampling=dict(
                source_pressure_bar=source.tolist(),
                source_archive_indices=self.pressure_indices.tolist(),
                target_pressure_bar=refined.target_pressure.tolist(),
                source_pressure_sha256=_sha256_array(source),
                target_pressure_sha256=_sha256_array(refined.target_pressure),
                lower_indices=mapping[0].tolist(), upper_indices=mapping[1].tolist(),
                upper_weights=mapping[2].tolist(),
                policy="Linear in log P for log cross section and its inverse-temperature derivative; native nodes exact; no extrapolation",
                scope="Interpolation of the selected stored nodes; accuracy between the full native pressure nodes is not certified",
            ), all_chunk_coefficients_verified=False)
        refined._column, refined._calculate = None, None
        return refined

    def top_flux(self, column, temperature, bottom, parameters, chemistry):
        """Reuse column.fluxes on small runtime tables and return full host flux."""
        self.provenance["all_chunk_coefficients_verified"] = False
        if tuple(column.lbl) != self.species or not np.array_equal(np.asarray(column.nu), self.nu):
            raise ValueError("Streaming plan does not match the column species or grid")
        expected_pressure = (self.source_pressure[self.pressure_indices] if self.target_pressure is None
                             else self.target_pressure)
        if (np.shape(column.pressure) != expected_pressure.shape
                or not np.allclose(column.pressure, expected_pressure, rtol=1e-7, atol=0)):
            raise ValueError("Streaming pressure selection does not match the column")
        if self.source_temperature is not None:
            expected_temperature = (self.source_temperature if self.temperature_indices is None
                                    else self.source_temperature[self.temperature_indices])
            if any(not np.array_equal(opacity.diffgrid_info.temperature_grid, expected_temperature)
                   for opacity in column.lbl.values()):
                raise ValueError("Streaming temperature selection does not match the column")
        if (not bool(column.temperature_valid(temperature, bottom, parameters))
                or not bool(jnp.all(chemistry.diagnostics.valid))):
            raise ValueError("Streaming spectrum requires a valid physical state")

        if self._column is not column:
            @jax.jit
            def calculate(nu, stellar, tables, temperature, bottom, parameters, chemistry):
                # Reuse every existing opacity, continuum, boundary and RT formula.
                # The original column and its solver remain unchanged.
                local = copy(column)
                local.nu, local.star_resolved = nu, stellar
                return local.fluxes(temperature, bottom, parameters, chemistry,
                                    resolved=True, resolved_tables=tables).upward[0]
            self._column, self._calculate = column, calculate

        result = []
        for index, (chunk, nu) in enumerate(zip(self.chunks, self.grids)):
            tables = {}
            for name in self.species:
                entry = chunk[name]
                _verify(entry.path, entry.sha256)
                _verify(entry.metadata_path, entry.metadata_sha256)
                opa = OpaDiffgrid.from_saved_opa(str(entry.path))
                if not np.array_equal(opa.nu_grid, nu):
                    raise ValueError("Streaming chunk grid changed after setup")
                opa.check_pressure_grid(self.source_pressure)
                info = opa.diffgrid_info
                if self.pressure_mapping is None:
                    values = tuple(getattr(info, key)[self.pressure_indices] for key in COEFFICIENTS)
                else:
                    values = tuple(resample_pressure_values(getattr(info, key), self.pressure_mapping)
                                   for key in COEFFICIENTS)
                inverse = info.inverse_temperature_grid
                if self.temperature_indices is not None:
                    if not np.array_equal(info.temperature_grid, self.source_temperature):
                        raise ValueError("Streaming temperature coordinates changed after setup")
                    inverse = inverse[self.temperature_indices]
                    values = tuple(value[:, self.temperature_indices] for value in values)
                tables[name] = (inverse, *values, info.log_cross_section_floor)
            # Explicit table arguments avoid retaining each archive as a JIT constant.
            values = np.asarray(self._calculate(jnp.asarray(nu), jnp.asarray(column.stellar.at(nu)),
                                                tables, temperature, bottom, parameters, chemistry))
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite streamed emergent spectrum")
            result.append(values[1:] if index else values)
            del tables, opa, info
        self.provenance["all_chunk_coefficients_verified"] = True
        return np.concatenate(result)

    def prediction(self, column, top_flux, resolving_power=None):
        """Apply the unchanged global observation operator across chunk seams."""
        column.check_response_coverage(resolving_power)
        if np.shape(top_flux) != self.nu.shape or not np.all(np.isfinite(top_flux)):
            raise ValueError("Supply the complete finite streamed top flux")
        return pixel_flux_ratio(
            column.W, response_samples(self.nu, top_flux, column.pixel_nu, resolving_power),
            response_samples(column.stellar.nu_cm1, column.stellar.surface_flux_per_cm1,
                             column.pixel_nu, resolving_power), column.area_ratio,
        )


def load_streaming_opacities(manifest):
    """Return metadata-only opacities and a chunk plan from a verified manifest.

    The preparation record is itself a manifest-pinned asset. Its child JSON
    records pin each chunk NPZ and sidecar. Coordinate digests and child JSON
    hashes are checked at setup; full coefficient-file hashes and the standard
    strict Diffgrid reader are checked on every streaming use.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("Enable JAX x64 before loading streaming metadata")
    descriptors, species_chunks, common_grids = {}, {}, None
    pins = {}
    for name, specification in manifest.config["opacity"]["lbl"].items():
        if specification["format"] != "diffgrid":
            raise ValueError("Streaming requires prepared Diffgrid NPZ archives")
        asset, meta_asset = specification["asset"], specification["metadata_asset"]
        merged, sidecar = manifest.paths[asset], manifest.paths[meta_asset]
        if merged.suffix != ".npz" or sidecar != merged.with_name(merged.stem + "_metadata.json"):
            raise ValueError("Streaming requires the pinned sibling Diffgrid sidecar")
        _verify(sidecar, manifest.sha256[meta_asset])
        arrays, metadata = _read_coordinates(merged, sidecar)
        descriptors[name] = MetadataOnlyDiffgrid(arrays)
        pins[str(merged)], pins[str(sidecar)] = manifest.sha256[asset], manifest.sha256[meta_asset]
        preparation_key = specification.get("preparation_asset", name + "_preparation")
        preparation = manifest.paths[preparation_key]
        _verify(preparation, manifest.sha256[preparation_key])
        pins[str(preparation)] = manifest.sha256[preparation_key]
        record = json.loads(preparation.read_text())
        for filename, path, digest in (("diffgrid.npz", merged, manifest.sha256[asset]),
                                       ("diffgrid_metadata.json", sidecar, manifest.sha256[meta_asset])):
            entry = record["files"][filename]
            if _entry_path(entry, preparation.parent) != path or entry["sha256"] != digest:
                raise ValueError("Preparation record does not pin the merged manifest asset")
        fingerprint = record["plan"]["grid_fingerprint"]
        if metadata["user_meta"]["grid_fingerprint"] != fingerprint:
            raise ValueError("Merged and preparation grid fingerprints differ")
        chunks, grids = [], []
        for child in record["chunk_records"]:
            child_path = _entry_path(child, preparation.parent)
            _verify(child_path, child["sha256"])
            pins[str(child_path)] = child["sha256"]
            content = json.loads(child_path.read_text())
            if (content["grid_fingerprint"] != fingerprint
                    or content["mdb_path"] != record["plan"]["mdb_path"]):
                raise ValueError("Chunk and merged preparation settings differ")
            files = content["files"]
            if "diffgrid.npz" not in files:
                if "diffgrid_metadata.json" in files:
                    raise ValueError("Incomplete streaming chunk archive")
                continue
            path = _entry_path(files["diffgrid.npz"], child_path.parent)
            meta_path = _entry_path(files["diffgrid_metadata.json"], child_path.parent)
            if meta_path != path.with_name(path.stem + "_metadata.json"):
                raise ValueError("Streaming chunk metadata must be its sibling sidecar")
            _verify(meta_path, files["diffgrid_metadata.json"]["sha256"])
            coordinates, meta = _read_coordinates(path, meta_path)
            if meta["user_meta"]["grid_fingerprint"] != fingerprint:
                raise ValueError("Chunk grid fingerprint disagrees with preparation")
            for field in COORDINATES[1:]:
                if not np.array_equal(coordinates[field], arrays[field]):
                    raise ValueError(f"Chunk coordinate differs from merged archive: {field}")
            nu = coordinates["nu_grid"]
            if grids and grids[-1][-1] != nu[0]:
                raise ValueError("Ordered streaming chunks must share exactly one endpoint")
            grids.append(nu)
            chunks.append(ChunkArchive(path, meta_path, files["diffgrid.npz"]["sha256"],
                                       files["diffgrid_metadata.json"]["sha256"]))
            pins[str(path)], pins[str(meta_path)] = chunks[-1].sha256, chunks[-1].metadata_sha256
        if not grids or not np.array_equal(np.concatenate([grid[1:] if i else grid
                                                           for i, grid in enumerate(grids)]), arrays["nu_grid"]):
            raise ValueError("Streaming chunks must reconstruct the exact merged grid")
        if common_grids is None:
            common_grids = grids
        elif len(common_grids) != len(grids) or any(not np.array_equal(a, b) for a, b in zip(common_grids, grids)):
            raise ValueError("All species must share identical streaming chunk grids")
        species_chunks[name] = chunks
    if not descriptors:
        raise ValueError("Streaming requires a nonempty resolved species mapping")
    chunks = [{name: entries[i] for name, entries in species_chunks.items()}
              for i in range(len(common_grids))]
    nu = next(iter(descriptors.values())).nu_grid
    provenance = {"mode": "prepared_diffgrid_chunks", "input_manifest_sha256": manifest.sha256["__manifest__"],
                  "file_sha256": pins, "chunk_count": len(chunks),
                  "all_chunk_coefficients_verified": False,
                  "coefficient_verification": "SHA256 and strict Diffgrid validation on each streaming read",
                  "seam_policy": "Keep the preceding chunk endpoint, matching prepare_rce_wasp18b_opacity.merge_species"}
    info = next(iter(descriptors.values())).diffgrid_info
    return descriptors, StreamingResolvedPlan(descriptors, chunks, nu, common_grids, provenance,
                                               info.pressure_grid, info.temperature_grid)
