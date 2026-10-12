"""LSD accumulation must preserve each line's order, including cancellation."""

import numpy as np
import pytest

from exojax.opacity._common.lsd import npadd3D_direct1D


def sequential_reference(a, w, cx, ix, cy, iy, cz, iz, sumx=1., sumz=1.):
    """Accumulate four ordered passes independently of NumPy advanced indexing."""
    conjugate_cx, conjugate_cz = sumx - cx, sumz - cz
    passes = (
        (ix, iz, w * conjugate_cx * cy * conjugate_cz),
        (ix + 1, iz, w * cx * cy * conjugate_cz),
        (ix, iz + 1, w * conjugate_cx * cy * cz),
        (ix + 1, iz + 1, w * cx * cy * cz),
    )
    for x, z, values in passes:
        for i, j, k, value in zip(*np.broadcast_arrays(x, iy, z, values)):
            a[i, j, k] += value
    return a


@pytest.mark.parametrize("sumx,sumz", [(1., 1.), (.75, 0.)])
def test_duplicate_scatter_keeps_pass_and_line_order(sumx, sumz):
    rng = np.random.default_rng(187)
    count, shape = 3000, (7, 4, 6)
    indices = [rng.integers(0, n, count, dtype=np.intp)
               for n in (shape[0] - 1, shape[1], shape[2] - 1)]
    weight = rng.choice([1e20, 1., -1e20, -1., 1e-20], count)
    cx, cy, cz = rng.random((3, count))
    initial = rng.standard_normal(shape)
    args = (weight, cx, indices[0], cy, indices[1], cz, indices[2], sumx, sumz)
    expected = sequential_reference(initial.copy(), *args)
    actual = npadd3D_direct1D(initial.copy(), *args)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["noncontiguous", "negative", "scalar", "int32", "float32", "empty"])
def test_general_scatter_routes_preserve_values_and_destination(mode):
    dtype = np.float32 if mode == "float32" else np.float64
    left, right = np.ones((6, 8, 6), dtype=dtype), np.ones((6, 8, 6), dtype=dtype)
    if mode == "noncontiguous":
        left, right = left[:, ::2, :], right[:, ::2, :]
    ix = np.array([0, 1, 1, 2, 1], dtype=np.intp)
    iy, iz = ix.copy(), ix.copy()
    if mode == "negative":
        ix[0], iy[1], iz[2] = -1, -2, -3
    elif mode == "scalar":
        iy = 0
    elif mode == "int32":
        ix = ix.astype(np.int32)
    elif mode == "empty":
        ix, iy, iz = (np.array([], dtype=np.intp) for _ in range(3))
    args = (np.ones(ix.shape), .3, ix, .7, iy, .9, iz)
    expected = sequential_reference(left, *args)
    actual = npadd3D_direct1D(right, *args)
    assert actual is right
    np.testing.assert_array_equal(actual, expected)


def test_invalid_neighbor_keeps_index_error():
    indices = np.array([2], dtype=np.intp)
    with pytest.raises(IndexError):
        npadd3D_direct1D(np.ones((3, 3, 3)), np.ones(1), .3, indices,
                        .7, indices, .9, indices)
