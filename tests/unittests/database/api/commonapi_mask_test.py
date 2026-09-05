"""Regression tests for masks shared by HITRAN and HITEMP databases."""

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database._common.commonapi import MdbCommonHitempHitran


@pytest.mark.parametrize("gpu_transfer", [False, True])
@pytest.mark.parametrize("optional_fields", [False, True])
def test_mask_preserves_isotope_partition_functions_and_line_arrays(
    gpu_transfer, optional_fields
):
    mdb = MdbCommonHitempHitran.__new__(MdbCommonHitempHitran)
    fields = [
        "nu_lines",
        "line_strength_ref_original",
        "delta_air",
        "A",
        "n_air",
        "gamma_air",
        "gamma_self",
        "elower",
        "gpp",
    ]
    for field in fields:
        setattr(mdb, field, np.arange(1.0, 5.0))
    mdb.logsij0 = np.log(mdb.line_strength_ref_original)
    fields.append("logsij0")
    mdb.gpu_transfer = gpu_transfer
    if gpu_transfer:
        mdb.dev_nu_lines = jnp.array(mdb.nu_lines)
        mdb.logsij0 = jnp.array(mdb.logsij0)
        fields.append("dev_nu_lines")
    mdb.with_error = optional_fields
    if optional_fields:
        for broadener in ("h2", "he", "co2", "h2o"):
            for parameter in ("n", "gamma"):
                field = f"{parameter}_{broadener}"
                setattr(mdb, field, np.arange(1.0, 5.0))
                fields.append(field)
        mdb.ierr = np.array([123456, 234567, 345678, 456789])
        mdb.add_error()
        fields.extend(
            [
                "ierr",
                "nu_lines_err",
                "line_strength_ref_err",
                "gamma_air_err",
                "gamma_self_err",
                "n_air_err",
                "delta_air_err",
            ]
        )
    mdb.isoid = np.array([1, 2, 3, 2])
    mdb.uniqiso = np.array([1, 2, 3])
    mdb.T_gQT = jnp.array(
        [[100.0, 300.0, 900.0], [100.0, 400.0, 1000.0], [100.0, 500.0, 1100.0]]
    )
    mdb.gQT = jnp.array([[1.0, 3.0, 9.0], [2.0, 10.0, 40.0], [5.0, 30.0, 120.0]])

    mask = mdb.isoid != 1
    expected = {field: np.asarray(getattr(mdb, field))[mask] for field in fields}
    expected_qr = np.asarray(mdb.qr_interp_lines(800.0, 296.0))[mask]
    expected_qt = np.asarray(mdb.QT_interp(2, 800.0))
    expected_gqt = np.asarray(mdb.gQT)[1:]
    expected_temperatures = np.asarray(mdb.T_gQT)[1:]

    mdb.apply_mask_mdb(mask)

    for field, values in expected.items():
        np.testing.assert_array_equal(getattr(mdb, field), values)
    np.testing.assert_array_equal(mdb.isoid, [2, 3, 2])
    np.testing.assert_array_equal(mdb.uniqiso, [2, 3])
    np.testing.assert_array_equal(mdb.gQT, expected_gqt)
    np.testing.assert_array_equal(mdb.T_gQT, expected_temperatures)
    np.testing.assert_allclose(mdb.qr_interp_lines(800.0, 296.0), expected_qr)
    np.testing.assert_allclose(mdb.QT_interp(2, 800.0), expected_qt)
    assert hasattr(mdb, "dev_nu_lines") == gpu_transfer

    mdb.apply_mask_mdb(mdb.isoid == 3)
    np.testing.assert_array_equal(mdb.uniqiso, [3])
    np.testing.assert_allclose(mdb.qr_interp_lines(800.0, 296.0), expected_qr[1:2])
