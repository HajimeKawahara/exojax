"""Offline API checks for selectable PreMODIT profile kernels."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity import OpaDiffgrid, OpaPremodit, saveopa
from exojax.opacity.diffgrid.diagnostics import compare_diffgrid_with_teacher
from exojax.utils.grids import wavenumber_grid


def _teacher(*, diffmode=0, **kwargs):
    snapshot = MDBSnapshot(
        meta=MDBMeta(
            dbtype="exomol",
            molmass=18.0,
            T_gQT=np.asarray([300.0, 1000.0, 2000.0]),
            gQT=np.asarray([1.0, 2.0, 4.0]),
        ),
        lines=Lines(
            nu_lines=np.asarray([999.0, 1000.0, 1002.0]),
            elower=np.asarray([20.0, 350.0, 900.0]),
            line_strength_ref_original=np.asarray([2.0e-23, 4.0e-23, 3.0e-23]),
        ),
        n_Texp=np.asarray([0.5, 0.5, 0.5]),
        alpha_ref=np.asarray([0.06, 0.06, 0.06]),
    )
    nu_grid, _, _ = wavenumber_grid(
        995.0, 1005.0, 32, unit="cm-1", xsmode="premodit"
    )
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid,
        diffmode=diffmode,
        broadening_resolution={"mode": "single", "value": (0.06, 0.5)},
        **kwargs,
    )
    opa.manual_setting(dE=300.0, Tref=1000.0, Twt=1200.0, Tmin=500.0, Tmax=1800.0)
    return opa


@pytest.mark.parametrize("diffmode", [0, 1, 2])
@pytest.mark.parametrize("profile_kernel", ["analytic", "real_space"])
def test_vector_matrix_and_gradients_for_each_kernel(diffmode, profile_kernel):
    original = _teacher(diffmode=diffmode)
    assert original.profile_kernel == "analytic"
    opa = original.with_profile_kernel(profile_kernel)
    assert opa.lbd_coeff is original.lbd_coeff
    assert opa.pre_modit_info is original.pre_modit_info
    assert opa is not original
    assert original.profile_kernel == "analytic"
    assert (opa == original) == (profile_kernel == "analytic")

    temperature = jnp.asarray([850.0, 1100.0])
    pressure = jnp.asarray([0.3, 1.0])
    vectors = jnp.stack(
        [opa.xsvector(t, p) for t, p in zip(temperature, pressure)]
    )
    matrix = jax.jit(opa.xsmatrix)(temperature, pressure)
    np.testing.assert_allclose(matrix, vectors, rtol=1.0e-12, atol=1.0e-35)
    if profile_kernel == "analytic":
        np.testing.assert_array_equal(
            matrix, jax.jit(original.xsmatrix)(temperature, pressure)
        )
    else:
        assert np.min(matrix) >= 0.0

    # The weighted linear observable probes both temperature and pressure AD
    # without adding a log floor or any extra nonsmooth operation.
    weights = jnp.linspace(0.5, 1.5, len(opa.nu_grid))

    def objective(parameters):
        return jnp.sum(weights * opa.xsvector(parameters[0], parameters[1])) / 1.0e-22

    parameters = jnp.asarray([850.0, 0.3])
    gradient = jax.jit(jax.grad(objective))(parameters)
    steps = np.asarray([1.0e-2, 1.0e-5])
    finite_difference = []
    for index, step in enumerate(steps):
        offset = np.zeros(2)
        offset[index] = step
        finite_difference.append(
            (objective(parameters + offset) - objective(parameters - offset))
            / (2.0 * step)
        )
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(
        gradient, finite_difference, rtol=3.0e-5, atol=1.0e-12
    )


def test_constructor_selects_real_space_and_rejects_invalid_mode():
    opa = _teacher(profile_kernel="real_space")
    reference = _teacher().with_profile_kernel("real_space")
    np.testing.assert_array_equal(
        opa.xsvector(850.0, 0.3), reference.xsvector(850.0, 0.3)
    )
    with pytest.raises(ValueError, match="profile_kernel"):
        _teacher(profile_kernel="unknown")
    with pytest.raises(ValueError, match="profile_kernel"):
        opa.with_profile_kernel("unknown")


def test_stitching_keeps_real_space_default():
    opa = _teacher(nstitch=2)
    assert opa.profile_kernel == "real_space"
    np.testing.assert_array_equal(
        opa.xsvector(850.0, 0.3),
        opa.with_profile_kernel("real_space").xsvector(850.0, 0.3),
    )
    with pytest.raises(ValueError, match="real_space"):
        _teacher(nstitch=2, profile_kernel="analytic")
    with pytest.raises(ValueError, match="real_space"):
        opa.with_profile_kernel("analytic")


@pytest.mark.parametrize(
    ("teacher_mode", "options", "expected_mode"),
    [
        ("analytic", {}, "real_space"),
        ("analytic", {"profile_kernel": "analytic"}, "analytic"),
        ("analytic", {"profile_kernel": None}, "analytic"),
        ("real_space", {"profile_kernel": None}, "real_space"),
    ],
)
def test_diffgrid_selects_kernel_without_mutating_compiled_teacher(
    teacher_mode, options, expected_mode
):
    teacher = _teacher(profile_kernel=teacher_mode)
    temperature = jnp.asarray([800.0, 1200.0])
    pressure = np.asarray([0.3, 1.0])
    compiled_teacher = jax.jit(teacher.xsmatrix)
    before = compiled_teacher(temperature, pressure)
    coefficients = teacher.lbd_coeff

    diffgrid = OpaDiffgrid(
        teacher, np.asarray([800.0, 1200.0]), pressure, **options
    )

    assert diffgrid.teacher_profile_kernel == expected_mode
    assert teacher.profile_kernel == teacher_mode
    assert teacher.lbd_coeff is coefficients
    np.testing.assert_array_equal(compiled_teacher(temperature, pressure), before)
    np.testing.assert_allclose(
        teacher.xsmatrix(temperature, pressure), before, rtol=1.0e-12
    )
    selected_teacher = teacher.with_profile_kernel(expected_mode)
    np.testing.assert_allclose(
        diffgrid.xsmatrix(temperature),
        selected_teacher.xsmatrix(temperature, pressure),
        rtol=1.0e-12,
        atol=1.0e-35,
    )
    if expected_mode != teacher.profile_kernel:
        with pytest.raises(ValueError, match="with_profile_kernel"):
            compare_diffgrid_with_teacher(diffgrid, teacher, temperature)
    summary = compare_diffgrid_with_teacher(diffgrid, selected_teacher, temperature)
    assert summary.maximum_absolute_log_cross_section_error < 1.0e-12


@pytest.mark.parametrize("format", ["npz", "zarr"])
@pytest.mark.parametrize("profile_kernel", ["analytic", "real_space"])
def test_diffgrid_persists_selected_teacher_kernel(tmp_path, format, profile_kernel):
    teacher = _teacher()
    temperature = np.asarray([800.0, 1200.0])
    pressure = np.asarray([0.3, 1.0])
    diffgrid = OpaDiffgrid(
        teacher, temperature, pressure, profile_kernel=profile_kernel
    )
    path = tmp_path / "diffgrid"
    saveopa(diffgrid, str(path), format=format)
    loaded = OpaDiffgrid.from_saved_opa(str(path.with_suffix("." + format)))
    assert loaded.teacher_profile_kernel == profile_kernel
    np.testing.assert_array_equal(
        loaded.xsmatrix(temperature), diffgrid.xsmatrix(temperature)
    )


@pytest.mark.parametrize("nstitch", [1, 2])
@pytest.mark.parametrize("format", ["npz", "zarr"])
def test_legacy_premodit_kernel_is_inferred_from_stitching(tmp_path, nstitch, format):
    teacher = _teacher(nstitch=nstitch)
    path = tmp_path / "legacy"
    saveopa(teacher, str(path), format=format)
    if format == "npz":
        metadata_path = tmp_path / "legacy_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["opa_state"]["profile_kernel"]
        metadata_path.write_text(json.dumps(metadata))
    else:
        import zarr

        group = zarr.open(str(path.with_suffix(".zarr")), mode="a")
        state = dict(group.attrs["opa_state"])
        del state["profile_kernel"]
        group.attrs["opa_state"] = state

    loaded = OpaPremodit.from_saved_opa(str(path.with_suffix("." + format)))

    assert loaded.profile_kernel == ("analytic" if nstitch == 1 else "real_space")
    np.testing.assert_array_equal(
        loaded.xsvector(850.0, 0.3), teacher.xsvector(850.0, 0.3)
    )
