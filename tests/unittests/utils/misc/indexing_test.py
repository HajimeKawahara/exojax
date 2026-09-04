from exojax.utils.indexing import uniqidx
from exojax.utils.indexing import find_or_add_index
from exojax.utils.indexing import uniqidx_neibouring
from exojax.utils.indexing import unique_rows
from exojax.utils.indexing import get_smooth_index
from exojax.utils.indexing import get_value_at_smooth_index
import numpy as np
import pytest


def test_uniqidx_2D():
    a = np.array([[4, 1], [7, 1], [7, 2], [7, 1], [8, 0], [4, 1]])
    uidx, val = uniqidx(a)
    ref = np.array([0, 1, 2, 1, 3, 0])
    assert np.all(uidx - ref == 0.0)
    refval = np.array([[4, 1], [7, 1], [7, 2], [8, 0]])
    assert np.all(val - refval == 0.0)


def test_uniqidx_3D():
    a = np.array([[4, 1, 2], [7, 1, 2], [7, 2, 2], [7, 1, 2], [8, 0, 2], [4, 1, 1]])
    uidx, val = uniqidx(a)
    ref = np.array([1, 2, 3, 2, 4, 0])
    assert np.all(uidx - ref == 0.0)
    refval = np.array([[4, 1, 1], [4, 1, 2], [7, 1, 2], [7, 2, 2], [8, 0, 2]])
    assert np.all(val - refval == 0.0)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64])
@pytest.mark.parametrize(
    "rows, expected_indices, expected_rows",
    [
        (
            [[2], [-1], [2], [0], [-1], [-1]],
            [2, 0, 2, 1, 0, 0],
            [[-1], [0], [2]],
        ),
        (
            [[2, -1], [-1, 2], [2, -1], [0, -2], [-1, 2], [-1, 2]],
            [2, 0, 2, 1, 0, 0],
            [[-1, 2], [0, -2], [2, -1]],
        ),
        (
            [[2, -1, 3], [-1, 2, -3], [2, -1, 3], [0, -2, 1],
             [-1, 2, -3], [-1, 2, 1]],
            [3, 0, 3, 2, 0, 1],
            [[-1, 2, -3], [-1, 2, 1], [0, -2, 1], [2, -1, 3]],
        ),
    ],
)
def test_uniqidx_signed_duplicates(dtype, rows, expected_indices, expected_rows):
    a = np.array(rows, dtype=dtype)

    uidx, values = uniqidx(a)

    assert uidx.shape == (len(a),)
    assert uidx.dtype == np.dtype(int)
    assert values.dtype == a.dtype
    np.testing.assert_array_equal(uidx, expected_indices)
    np.testing.assert_array_equal(values, expected_rows)
    np.testing.assert_array_equal(values[uidx], a)
    np.testing.assert_array_equal(values, unique_rows(a))


def test_find_or_add_index():
    # case for existed in index_array
    a = np.array([[4, 1], [7, 1], [7, 2], [8, 0]])
    index_found, b = find_or_add_index([4, 1], a)
    assert index_found == 0
    assert np.all(a - b == 0)

    # case for not existed in index_array
    index_found, c = find_or_add_index([4, 4], a)
    assert index_found == 4
    ref = np.array([[4, 1], [7, 1], [7, 2], [8, 0], [4, 4]])
    assert np.all(c - ref == 0)


def test_uniqidx_neibouring():
    a = np.array([[4, 1], [7, 1], [7, 2], [8, 0], [4, 1]])
    udix, neighbor_indices, multi_index_update = uniqidx_neibouring(a)
    assert np.all(udix == [0, 1, 2, 3, 0])
    index_of_uidx = 3  # can be 0,1 ... , np.max(udix)
    assert np.all(multi_index_update[index_of_uidx] == [8, 0])
    i, j, k = neighbor_indices[index_of_uidx, :]
    assert np.all(
        multi_index_update[i] == multi_index_update[index_of_uidx] + np.array([1, 0])
    )
    assert np.all(
        multi_index_update[j] == multi_index_update[index_of_uidx] + np.array([0, 1])
    )
    assert np.all(
        multi_index_update[k] == multi_index_update[index_of_uidx] + np.array([1, 1])
    )


def test_uniqidx_neibouring_full_map():
    a = np.array([[0, 0], [-1, 0], [0, 0], [-1, -1], [0, 1]])

    uidx, neighbors, multi_index = uniqidx_neibouring(a)

    np.testing.assert_array_equal(uidx, [2, 1, 2, 0, 3])
    np.testing.assert_array_equal(
        neighbors, [[4, 1, 2], [2, 5, 3], [6, 3, 7], [7, 8, 9]]
    )
    np.testing.assert_array_equal(
        multi_index,
        [[-1, -1], [-1, 0], [0, 0], [0, 1], [0, -1],
         [-1, 1], [1, 0], [1, 1], [0, 2], [1, 2]],
    )
    np.testing.assert_array_equal(multi_index[uidx], a)


def test_unique_rows():
    a = np.array([[4, 1], [7, 1], [7, 2], [8, 0], [4, 1]])
    u = unique_rows(a)
    assert np.all(np.unique(a, axis=0) == u)


def test_get_value_at_smooth_index():
    array = np.array([10, 20, 30, 40, 50])
    smooth_index = np.array([0.5, 1.7, 2.5, 3.5])
    result = get_value_at_smooth_index(array, smooth_index)
    expected = np.array([15, 27, 35, 45])
    assert np.allclose(result, expected)


def test_get_smooth_index():
    xp = np.array([10, 20, 30, 40, 50])
    x = np.array([15, 27, 35, 45])
    result = get_smooth_index(xp, x)
    expected = np.array([0.5, 1.7, 2.5, 3.5])
    assert np.allclose(result, expected)


if __name__ == "__main__":
    # test_uniqidx_2D()
    # test_uniqidx_3D()
    # test_find_or_add_index()
    # test_uniqidx_neibouring()
    test_get_value_at_smooth_index()
    test_get_smooth_index()
