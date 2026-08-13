import hashlib
import json

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import pytest

from exojax import __version__
from exojax.opacity import OpaDiffgrid
from exojax.opacity import saveopa
from exojax.opacity.io.serialization import _sha256_array
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog


ARRAY_KEYS = (
    "nu_grid",
    "pressure_grid",
    "temperature_grid",
    "inverse_temperature_grid",
    "log_cross_section_grid",
    "log_cross_section_derivative_grid",
    "log_cross_section_floor",
)


class _AnalyticTeacher:
    method = "analytic"
    ready = True
    wavelength_order = "descending"
    molmass = 28.01

    def __init__(self, dtype=np.float32):
        self.nu_grid = np.asarray([1000.0, 1001.0, 1002.0], dtype=dtype)
        self.wav = nu2wav(
            self.nu_grid,
            wavelength_order=self.wavelength_order,
            unit="AA",
        )
        self.resolution = resolution_eslog(self.nu_grid)

    def xsmatrix(self, temperature, pressure):
        temperature = jnp.asarray(temperature)
        pressure = jnp.asarray(pressure)
        nu_grid = jnp.asarray(self.nu_grid, dtype=temperature.dtype)
        temperature_factor = jnp.exp(-700.0 / temperature)
        pressure_factor = 1.0 + 0.05 * jnp.log1p(pressure)
        wavenumber_factor = 1.0 + 0.02 * (nu_grid - nu_grid[0])
        return (
            jnp.asarray(1.0e-22, dtype=temperature.dtype)
            * temperature_factor[:, None]
            * pressure_factor[:, None]
            * wavenumber_factor[None, :]
        )


class _MinimalAnalyticTeacher:
    method = "minimal-analytic"
    ready = True
    nu_grid = np.asarray([1000.0, 1001.0], dtype=np.float32)

    def xsmatrix(self, temperature, pressure):
        value = jnp.exp(-500.0 / jnp.asarray(temperature))
        return jnp.broadcast_to(value[:, None], (len(pressure), len(self.nu_grid)))


@pytest.fixture(scope="module")
def diffgrid_case():
    teacher = _AnalyticTeacher()
    pressure_grid = np.asarray([0.1, 1.0], dtype=np.float32)
    temperature_grid = np.asarray([700.0, 1000.0, 1500.0], dtype=np.float32)
    opa = OpaDiffgrid(
        teacher,
        temperature_grid,
        pressure_grid,
        min_cross_section=1.0e-30,
    )
    return opa, teacher, pressure_grid


def _metadata_path(npz_path):
    return npz_path.with_name(npz_path.stem + "_metadata.json")


def _read_npz_archive(npz_path):
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    with open(_metadata_path(npz_path), "r") as stream:
        metadata = json.load(stream)
    return arrays, metadata


def _write_npz_archive(npz_path, arrays, metadata):
    np.savez_compressed(npz_path, **arrays)
    with open(_metadata_path(npz_path), "w") as stream:
        json.dump(metadata, stream, indent=4)


def _digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    return digest.hexdigest()


def _refresh_array_metadata(metadata, key, array):
    """Update the common metadata spellings used by opacity archives."""
    array = np.asarray(array)
    for container_name in ("array_digests", "digests"):
        container = metadata.get(container_name)
        if isinstance(container, dict) and key in container:
            container[key] = _digest(array)
    digest_key = f"{key}_digest"
    if digest_key in metadata:
        metadata[digest_key] = _digest(array)

    for container_name in ("array_shapes", "shapes"):
        container = metadata.get(container_name)
        if isinstance(container, dict) and key in container:
            container[key] = list(array.shape)
    for container_name in ("array_dtypes", "dtypes"):
        container = metadata.get(container_name)
        if isinstance(container, dict) and key in container:
            container[key] = str(array.dtype)


def _assert_same_diffgrid(expected, actual):
    assert actual.ready is True
    assert actual.method == "diffgrid"
    assert actual.teacher_method == expected.teacher_method
    assert actual.opainfo is actual.diffgrid_info
    for key in ARRAY_KEYS:
        actual_array = (
            actual.nu_grid
            if key == "nu_grid"
            else getattr(actual.diffgrid_info, key)
        )
        expected_array = (
            expected.nu_grid
            if key == "nu_grid"
            else getattr(expected.diffgrid_info, key)
        )
        np.testing.assert_array_equal(
            np.asarray(actual_array),
            np.asarray(expected_array),
        )


def test_npz_roundtrip_preserves_tables_metadata_and_optional_attributes(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = tmp_path / "analytic_diffgrid"
    aux = {
        "labels": ["CO", 1],
        "nested": {"values": np.asarray([1, 2], dtype=np.int32)},
    }
    user_meta = {"artifact": "offline-test", "revision": 2}

    saveopa(opa, str(path), format="npz", aux=aux, extra_meta=user_meta)
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    _assert_same_diffgrid(opa, loaded)
    assert loaded.aux == {
        "labels": ["CO", 1],
        "nested": {"values": [1, 2]},
    }
    assert loaded.user_meta == user_meta
    assert loaded.molmass == pytest.approx(opa.molmass)
    assert loaded.wavelength_order == opa.wavelength_order
    np.testing.assert_array_equal(loaded.wav, opa.wav)
    assert loaded.resolution == pytest.approx(opa.resolution)

    arrays, metadata = _read_npz_archive(path.with_suffix(".npz"))
    assert set(ARRAY_KEYS) <= set(arrays)
    assert metadata["schema_version"] == "ioopa@1"
    assert metadata["diffgrid_schema_version"] == "diffgrid@1"
    assert metadata["opa_type"] == "OpaDiffgrid"
    assert metadata["exojax_version"] == __version__
    assert metadata["opa_state"]["teacher_method"] == "analytic"
    assert metadata["units"] == {
        "wavenumber": "cm^-1",
        "pressure": "bar",
        "temperature": "K",
        "inverse_temperature": "K^-1",
        "cross_section": "cm^2",
        "log_cross_section_derivative": "K",
    }
    for key in ARRAY_KEYS:
        assert metadata["array_shapes"][key] == list(arrays[key].shape)
        assert metadata["array_dtypes"][key] == str(arrays[key].dtype)
        assert metadata["array_digests"][key] == _sha256_array(arrays[key])


def test_zarr_roundtrip_preserves_tables(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = tmp_path / "analytic_diffgrid"
    aux = {"provider": "external-catalog"}
    user_meta = {"artifact": "zarr-test"}

    saveopa(
        opa,
        str(path),
        format="zarr",
        aux=aux,
        extra_meta=user_meta,
    )
    saveopa(
        opa,
        str(path),
        format="npz",
        aux={"provider": "npz-sibling"},
    )
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".zarr")

    _assert_same_diffgrid(opa, loaded)
    assert loaded.aux == aux
    assert loaded.user_meta == user_meta
    assert loaded.molmass == pytest.approx(opa.molmass)
    assert loaded.wavelength_order == opa.wavelength_order
    np.testing.assert_array_equal(loaded.wav, opa.wav)
    assert loaded.resolution == pytest.approx(opa.resolution)


def test_loads_schema_compliant_npz_from_external_producer(tmp_path):
    arrays = {
        "nu_grid": np.asarray([1000.0, 1001.0], dtype=np.float32),
        "pressure_grid": np.asarray([1.0, 0.1], dtype=np.float32),
        "temperature_grid": np.asarray(
            [1500.0, 1000.0, 700.0], dtype=np.float32
        ),
    }
    arrays["inverse_temperature_grid"] = (
        np.asarray(1.0, dtype=np.float32) / arrays["temperature_grid"]
    )
    arrays["log_cross_section_grid"] = np.asarray(
        [
            [[-52.0, -51.0], [-50.0, -49.0], [-48.0, -47.0]],
            [[-51.0, -50.0], [-49.0, -48.0], [-47.0, -46.0]],
        ],
        dtype=np.float32,
    )
    arrays["log_cross_section_derivative_grid"] = np.zeros(
        (2, 3, 2), dtype=np.float32
    )
    arrays["log_cross_section_floor"] = np.asarray(-80.0, dtype=np.float32)
    metadata = {
        "schema_version": "ioopa@1",
        "diffgrid_schema_version": "diffgrid@1",
        "opa_type": "OpaDiffgrid",
        "exojax_version": __version__,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "created_at": "2026-01-01T00:00:00+00:00",
        "units": {
            "wavenumber": "cm^-1",
            "pressure": "bar",
            "temperature": "K",
            "inverse_temperature": "K^-1",
            "cross_section": "cm^2",
            "log_cross_section_derivative": "K",
        },
        "array_shapes": {
            name: list(array.shape) for name, array in arrays.items()
        },
        "array_dtypes": {
            name: str(array.dtype) for name, array in arrays.items()
        },
        "array_digests": {
            name: _digest(array) for name, array in arrays.items()
        },
        "opa_state": {
            "teacher_method": "external-producer",
            "optional_attributes": {"molmass": 28.01},
        },
        "aux": {"catalog": "independent"},
        "user_meta": {"license": "test-only"},
    }
    path = tmp_path / "external_diffgrid.npz"
    _write_npz_archive(path, arrays, metadata)

    loaded = OpaDiffgrid.from_saved_opa(str(path))

    assert loaded.teacher_method == "external-producer"
    assert loaded.molmass == pytest.approx(28.01)
    assert loaded.aux == {"catalog": "independent"}
    assert loaded.user_meta == {"license": "test-only"}
    result = loaded.xsmatrix(
        np.asarray([1500.0, 700.0], dtype=np.float32),
        arrays["pressure_grid"],
    )
    assert result.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(result)))


def test_generic_saveopa_dispatches_diffgrid(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = tmp_path / "generic_dispatch"

    saveopa(opa, str(path), format="npz")

    assert path.with_suffix(".npz").is_file()
    assert _metadata_path(path.with_suffix(".npz")).is_file()


@pytest.mark.parametrize(
    "suffix,format",
    [(".npz", "zarr"), (".zarr", "npz")],
)
def test_save_rejects_path_suffix_conflicting_with_format(
    tmp_path, diffgrid_case, suffix, format
):
    opa, _, _ = diffgrid_case

    with pytest.raises(ValueError, match="suffix.*conflicts"):
        saveopa(opa, str(tmp_path / f"artifact{suffix}"), format=format)


def test_save_rejects_metadata_it_cannot_reload(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    original_teacher_method = opa.teacher_method
    try:
        opa.teacher_method = 1
        with pytest.raises(ValueError, match="teacher_method"):
            saveopa(opa, str(tmp_path / "invalid_provenance"), format="npz")
    finally:
        opa.teacher_method = original_teacher_method


def test_loaded_diffgrid_does_not_require_teacher(tmp_path):
    teacher = _AnalyticTeacher()
    opa = OpaDiffgrid(
        teacher,
        np.asarray([700.0, 1000.0, 1500.0], dtype=np.float32),
        np.asarray([0.1, 1.0], dtype=np.float32),
        min_cross_section=1.0e-30,
    )
    path = tmp_path / "teacher_independent"
    saveopa(opa, str(path), format="npz")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the saved diffgrid must not call its teacher")

    teacher.xsmatrix = fail_if_called
    del opa
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    result = loaded.xsmatrix(jnp.asarray([800.0, 1200.0]))
    assert result.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(result)))
    assert not hasattr(loaded, "base_opa")


def test_optional_attribute_absence_is_preserved(tmp_path):
    opa = OpaDiffgrid(
        _MinimalAnalyticTeacher(),
        np.asarray([700.0, 1000.0, 1500.0], dtype=np.float32),
        np.asarray([0.1], dtype=np.float32),
        min_cross_section=1.0e-30,
    )
    path = tmp_path / "minimal_attributes"
    saveopa(opa, str(path), format="npz")

    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    for attribute in ("wavelength_order", "wav", "resolution", "molmass"):
        assert not hasattr(loaded, attribute)


def test_loaded_xsmatrix_matches_at_nodes_and_interior(tmp_path, diffgrid_case):
    opa, _, pressure_grid = diffgrid_case
    path = tmp_path / "evaluation_roundtrip"
    saveopa(opa, str(path), format="npz")
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    at_nodes = np.asarray([700.0, 1500.0], dtype=np.float32)
    interior = np.asarray([800.0, 1200.0], dtype=np.float32)
    for temperature in (at_nodes, interior):
        np.testing.assert_allclose(
            loaded.xsmatrix(temperature, pressure_grid),
            opa.xsmatrix(temperature, pressure_grid),
            rtol=2.0e-6,
            atol=0.0,
        )


def test_loaded_diffgrid_supports_jax_transformations(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = tmp_path / "jax_transformations"
    saveopa(opa, str(path), format="npz")
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")
    temperature = jnp.asarray([800.0, 1200.0])
    tangent = jnp.asarray([0.5, -0.25])

    compiled = jax.jit(loaded.xsmatrix)(temperature)
    batched = jax.vmap(loaded.xsmatrix)(
        jnp.stack([temperature, temperature + jnp.asarray([10.0, -10.0])])
    )

    def objective(value):
        return jnp.sum(jnp.log(loaded.xsmatrix(value)))

    gradient = jax.grad(objective)(temperature)
    value, directional_derivative = jax.jvp(
        objective, (temperature,), (tangent,)
    )

    assert compiled.shape == (2, 3)
    assert batched.shape == (2, 2, 3)
    for result in (compiled, batched, gradient, value, directional_derivative):
        assert np.all(np.isfinite(np.asarray(result)))


def test_loaded_diffgrid_preserves_pressure_validation(tmp_path, diffgrid_case):
    opa, _, pressure_grid = diffgrid_case
    path = tmp_path / "pressure_validation"
    saveopa(opa, str(path), format="npz")
    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    with pytest.raises(ValueError, match="rebuild the table"):
        loaded.xsmatrix(np.asarray([800.0, 1200.0]), pressure_grid * 1.01)
    with pytest.raises(ValueError, match="shape does not match"):
        loaded.xsmatrix(np.asarray([800.0, 1200.0]), pressure_grid[:-1])


@pytest.mark.parametrize(
    "saved_pressure",
    [
        np.asarray([1.0, 0.1], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    ],
)
def test_load_preserves_pressure_layer_order_without_uniqueness_requirement(
    tmp_path, diffgrid_case, saved_pressure
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "pressure_layer_order")
    arrays, metadata = _read_npz_archive(path)
    arrays["pressure_grid"] = saved_pressure
    _refresh_array_metadata(metadata, "pressure_grid", saved_pressure)
    _write_npz_archive(path, arrays, metadata)

    loaded = OpaDiffgrid.from_saved_opa(str(path))

    np.testing.assert_array_equal(loaded.pressure_grid, saved_pressure)
    result = loaded.xsmatrix(
        np.asarray([800.0, 1200.0], dtype=np.float32), saved_pressure
    )
    assert np.all(np.isfinite(np.asarray(result)))


def _saved_npz(tmp_path, opa, name="archive"):
    path = tmp_path / name
    saveopa(opa, str(path), format="npz")
    return path.with_suffix(".npz")


def test_load_rejects_missing_required_array(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "missing_array")
    arrays, metadata = _read_npz_archive(path)
    del arrays["log_cross_section_derivative_grid"]
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(KeyError, match="log_cross_section_derivative_grid"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_wrong_table_shape(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "wrong_shape")
    arrays, metadata = _read_npz_archive(path)
    arrays["log_cross_section_derivative_grid"] = arrays[
        "log_cross_section_derivative_grid"
    ][..., :-1]
    _refresh_array_metadata(
        metadata,
        "log_cross_section_derivative_grid",
        arrays["log_cross_section_derivative_grid"],
    )
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="shape"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_fractional_shape_metadata(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "fractional_shape")
    arrays, metadata = _read_npz_archive(path)
    metadata["array_shapes"]["pressure_grid"] = [2.9]
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="Invalid shape metadata"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_optional_array_declared_but_missing(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "missing_optional_array")
    arrays, metadata = _read_npz_archive(path)
    del arrays["wav"]
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="missing from the payload.*wav"):
        OpaDiffgrid.from_saved_opa(str(path))


@pytest.mark.parametrize(
    "key,index",
    [
        ("pressure_grid", (0,)),
        ("temperature_grid", (0,)),
        ("inverse_temperature_grid", (0,)),
        ("log_cross_section_grid", (0, 0, 0)),
        ("log_cross_section_derivative_grid", (0, 0, 0)),
    ],
)
def test_load_rejects_nonfinite_arrays(tmp_path, diffgrid_case, key, index):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, f"nonfinite_{key}")
    arrays, metadata = _read_npz_archive(path)
    arrays[key] = arrays[key].copy()
    arrays[key][index] = np.nan
    _refresh_array_metadata(metadata, key, arrays[key])
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="finite"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_inconsistent_temperature_coordinates(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "inconsistent_coordinates")
    arrays, metadata = _read_npz_archive(path)
    arrays["temperature_grid"] = arrays["temperature_grid"].copy()
    arrays["temperature_grid"][1] *= 1.1
    _refresh_array_metadata(metadata, "temperature_grid", arrays["temperature_grid"])
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="temperature"):
        OpaDiffgrid.from_saved_opa(str(path))


@pytest.mark.parametrize("floor", [np.nan, np.inf])
def test_load_rejects_invalid_floor(tmp_path, diffgrid_case, floor):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "invalid_floor")
    arrays, metadata = _read_npz_archive(path)
    arrays["log_cross_section_floor"] = np.asarray(
        floor, dtype=arrays["log_cross_section_floor"].dtype
    )
    _refresh_array_metadata(
        metadata, "log_cross_section_floor", arrays["log_cross_section_floor"]
    )
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="floor|finite"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_floor_below_active_normal_range(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "unrepresentable_floor")
    arrays, metadata = _read_npz_archive(path)
    dtype = arrays["log_cross_section_grid"].dtype
    arrays["log_cross_section_floor"] = np.asarray(
        np.log(np.finfo(dtype).tiny) - 1.0,
        dtype=dtype,
    )
    _refresh_array_metadata(
        metadata,
        "log_cross_section_floor",
        arrays["log_cross_section_floor"],
    )
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="smallest normal"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_floor_above_active_finite_range(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "oversized_floor")
    arrays, metadata = _read_npz_archive(path)
    dtype = arrays["log_cross_section_grid"].dtype
    arrays["log_cross_section_floor"] = np.asarray(
        np.log(np.finfo(dtype).max) + 1.0,
        dtype=dtype,
    )
    _refresh_array_metadata(
        metadata,
        "log_cross_section_floor",
        arrays["log_cross_section_floor"],
    )
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="largest finite"):
        OpaDiffgrid.from_saved_opa(str(path))


def test_load_rejects_digest_mismatch(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "digest_mismatch")
    arrays, metadata = _read_npz_archive(path)
    arrays["pressure_grid"] = arrays["pressure_grid"].copy()
    arrays["pressure_grid"][0] *= 1.01
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="[Dd]igest"):
        OpaDiffgrid.from_saved_opa(str(path))


@pytest.mark.parametrize(
    "metadata_key,value,message",
    [
        ("opa_type", "OpaPremodit", "opa_type|OpaDiffgrid"),
        (
            "diffgrid_schema_version",
            "diffgrid@999",
            "Diffgrid schema|diffgrid.*schema",
        ),
    ],
)
def test_load_rejects_wrong_type_or_diffgrid_schema(
    tmp_path, diffgrid_case, metadata_key, value, message
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, f"wrong_{metadata_key}")
    arrays, metadata = _read_npz_archive(path)
    metadata[metadata_key] = value
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match=message):
        OpaDiffgrid.from_saved_opa(str(path))
    if metadata_key == "diffgrid_schema_version":
        with pytest.raises(ValueError, match=message):
            OpaDiffgrid.from_saved_opa(str(path), allow_downgrade=True)


def test_common_schema_allow_downgrade_policy(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "common_schema")
    arrays, metadata = _read_npz_archive(path)
    metadata["schema_version"] = "ioopa@0"
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="schema"):
        OpaDiffgrid.from_saved_opa(str(path))

    loaded = OpaDiffgrid.from_saved_opa(str(path), allow_downgrade=True)
    assert loaded.ready is True


def test_exojax_version_strictness(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "version_policy")
    arrays, metadata = _read_npz_archive(path)
    metadata["exojax_version"] = "0.0.external"
    _write_npz_archive(path, arrays, metadata)

    with pytest.raises(ValueError, match="strict=False"):
        OpaDiffgrid.from_saved_opa(str(path))

    loaded = OpaDiffgrid.from_saved_opa(str(path), strict=False)
    assert loaded.ready is True


def test_dtype_conversion_requires_non_strict_load(tmp_path, diffgrid_case):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "float64_payload")
    arrays, metadata = _read_npz_archive(path)
    for key in ARRAY_KEYS:
        arrays[key] = np.asarray(arrays[key], dtype=np.float64)
    arrays["inverse_temperature_grid"] = (
        np.asarray(1.0, dtype=np.float64) / arrays["temperature_grid"]
    )
    for key in ARRAY_KEYS:
        _refresh_array_metadata(metadata, key, arrays[key])
    _write_npz_archive(path, arrays, metadata)

    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="dtype|64-bit|strict=False"):
            OpaDiffgrid.from_saved_opa(str(path))

        loaded = OpaDiffgrid.from_saved_opa(str(path), strict=False)
        assert loaded.log_cross_section_grid.dtype == jnp.float32
        assert np.all(
            np.isfinite(np.asarray(loaded.xsmatrix(jnp.asarray([800.0, 1200.0]))))
        )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_roundtrip_accepts_integer_temperature_nodes(tmp_path):
    opa = OpaDiffgrid(
        _MinimalAnalyticTeacher(),
        np.asarray([700, 1000, 1500], dtype=np.int32),
        np.asarray([0.1], dtype=np.float32),
        min_cross_section=1.0e-30,
    )
    path = tmp_path / "integer_temperature_nodes"
    saveopa(opa, str(path), format="npz")

    loaded = OpaDiffgrid.from_saved_opa(str(path) + ".npz")

    np.testing.assert_array_equal(loaded.temperature_grid, opa.temperature_grid)
    result = loaded.xsmatrix(np.asarray([1000.0], dtype=np.float32))
    assert np.all(np.isfinite(np.asarray(result)))


def test_non_strict_load_rejects_table_overflow_in_active_dtype(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "overflowing_table")
    arrays, metadata = _read_npz_archive(path)
    value_grid = np.asarray(
        arrays["log_cross_section_grid"], dtype=np.float64
    )
    value_grid[0, 0, 0] = 2.0 * float(np.finfo(np.float32).max)
    arrays["log_cross_section_grid"] = value_grid
    _refresh_array_metadata(metadata, "log_cross_section_grid", value_grid)
    _write_npz_archive(path, arrays, metadata)

    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="finite.*active JAX dtype"):
            OpaDiffgrid.from_saved_opa(str(path), strict=False)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_non_strict_load_rejects_temperature_nodes_collapsed_by_jax_dtype(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "collapsed_nodes")
    arrays, metadata = _read_npz_archive(path)
    temperature_grid = np.asarray(
        [1500.0, 1000.00001, 1000.0], dtype=np.float64
    )
    arrays["temperature_grid"] = temperature_grid
    arrays["inverse_temperature_grid"] = 1.0 / temperature_grid
    for key in (
        "temperature_grid",
        "inverse_temperature_grid",
        "log_cross_section_grid",
        "log_cross_section_derivative_grid",
        "log_cross_section_floor",
        "pressure_grid",
    ):
        arrays[key] = np.asarray(arrays[key], dtype=np.float64)
        _refresh_array_metadata(metadata, key, arrays[key])
    _write_npz_archive(path, arrays, metadata)

    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="active JAX dtype|unique|distinct"):
            OpaDiffgrid.from_saved_opa(str(path), strict=False)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_load_rejects_wavenumber_nodes_collapsed_by_jax_dtype(
    tmp_path, diffgrid_case
):
    opa, _, _ = diffgrid_case
    path = _saved_npz(tmp_path, opa, "collapsed_wavenumbers")
    arrays, metadata = _read_npz_archive(path)
    arrays["nu_grid"] = np.asarray(
        [1000.0, 1000.00001, 1002.0], dtype=np.float64
    )
    _refresh_array_metadata(metadata, "nu_grid", arrays["nu_grid"])
    _write_npz_archive(path, arrays, metadata)

    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="nu_grid.*distinct.*active JAX"):
            OpaDiffgrid.from_saved_opa(str(path))
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
