"""Regression tests for ExoMol masks with and without device arrays."""

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.exomol.api import MdbExomol


@pytest.mark.parametrize("gpu_transfer", [False, True])
def test_apply_mask_keeps_exomol_line_arrays_aligned(gpu_transfer):
    mdb = MdbExomol.__new__(MdbExomol)
    mdb.gpu_transfer = gpu_transfer
    fields = [
        "A",
        "nu_lines",
        "gamma_natural",
        "alpha_ref",
        "n_Texp",
        "elower",
        "jlower",
        "jupper",
        "line_strength_ref_original",
        "gpp",
    ]
    for field in fields:
        setattr(mdb, field, np.arange(1.0, 5.0))
    mdb.logsij0 = np.log(mdb.line_strength_ref_original)
    mdb.gamma_natural = jnp.array(mdb.gamma_natural)
    fields.append("logsij0")
    if gpu_transfer:
        mdb.generate_jnp_arrays()
        fields.append("dev_nu_lines")

    mask = np.array([False, True, False, True])
    expected = {field: np.asarray(getattr(mdb, field))[mask] for field in fields}
    mdb.apply_mask_mdb(mask)

    for field, values in expected.items():
        np.testing.assert_array_equal(getattr(mdb, field), values)
    assert hasattr(mdb, "dev_nu_lines") == gpu_transfer

    mdb.apply_mask_mdb(np.array([False, True]))
    for field, values in expected.items():
        np.testing.assert_array_equal(getattr(mdb, field), values[1:])
