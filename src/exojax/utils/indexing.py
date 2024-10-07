""" index manipulation

* (np)getix provides the contribution and index.

"""

import numpy as np
import tqdm
import warnings
import jax.numpy as jnp


def get_smooth_index(xp, x):
    """get smooth index

    Args:
        xp: x grid
        x: x array
    Returns:
        float: smooth index
    """
    findex = jnp.arange(len(xp), dtype=float)
    smooth_index = jnp.interp(x, xp, findex)
    return smooth_index

def get_value_at_smooth_index(array, smooth_index):
    """get value at smooth index position (e.g. cloud base) from an array

    Args:
        array (float): array, such as log pressures or temperatures
        smooth_index (float): smooth index

    Returns:
        float: value at cloud base
    """

    ind = smooth_index.astype(int)
    # ind = jnp.clip(ind, 0, len(array) - 2)
    res = smooth_index - jnp.floor(smooth_index)
    return (1.0 - res) * array[ind] + res * array[ind + 1]


def getix(x, xv):
    """jnp version of getix.

    Args:
        x: x array
        xv: x grid, should be ascending order

    Returns:
        cont (contribution)
        index (index)

    Note:
        cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.

    Example:

        >>> from exojax.utils.indexing import getix
        >>> import jax.numpy as jnp
        >>> y=jnp.array([1.1,4.3])
        >>> yv=jnp.arange(6)
        >>> getix(y,yv)
        (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))
    """
    indarr = jnp.arange(len(xv))
    pos = jnp.interp(x, xv, indarr)
    cont, index = jnp.modf(pos)
    return cont, index.astype(int)


def npgetix(x, xv):
    """numpy version of getix.

    Args:
        x: x array
        xv: x grid, should be ascending order

    Returns:
        cont (contribution)
        index (index)

    Note:
        cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """
    indarr = np.arange(len(xv))
    pos = np.interp(x, xv, indarr)
    cont, index = np.modf(pos)
    return cont, index.astype(int)


def unique_rows(x):
    """memory saved version of np.unique(,axis=0)

    Notes:
        Originally from a snippet/Answer #4 (https://discuss.dizzycoding.com/find-unique-rows-in-numpy-array/?amp=1)

    Args:
        x (2D array): 2D array (N x M), need to be C-contiguous


    Returns:
        2D array: unique 2D array (N' x M), where N' <= N, removed duplicated M vector.
    """
    if not x.data.contiguous:
        warnings.warn("input should be contiguous.", UserWarning)
    uniq = np.unique(x.view(x.dtype.descr * x.shape[1]))
    return uniq.view(x.dtype).reshape(-1, x.shape[1])


def uniqidx(input_array):
    """compute indices based on uniq values of the input M-dimensional array.

    Args:
        input_array: input array (N,M), will use unique M-dim vectors

    Returns:
        unique index, unique value

    Examples:

        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])
        >>> uidx, uval=uniqidx(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]

    """
    N, _ = np.shape(input_array)
    uniqvals = unique_rows(input_array)
    uidx = np.zeros(N, dtype=int)
    uidx_p = np.where(input_array == uniqvals[0], True, False)
    uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = 0
    for i, uv in enumerate(tqdm.tqdm(uniqvals[1:], desc="uniqidx")):
        uidx_p = np.where(input_array == uv, True, False)
        uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = i + 1
    return uidx, uniqvals


def uniqidx_neibouring(index_array):
    """compute indices based on uniq values of the input index array and input index + one vector

    Args:
        index_array: input index array (N,M), will use unique M-dim vectors

    Returns:
        unique index (udix)
        neibouring index (nidx) for udix [N_uidx, 3]
        multi index as a function of nidx

    """
    uidx, multi_index = uniqidx(index_array)
    multi_index_update = np.copy(multi_index)
    Nuidx = np.max(uidx) + 1
    neighbor_indices = np.zeros((Nuidx, 3), dtype=int)
    for i in range(0, Nuidx):
        neighbor_indices[i, 0], multi_index_update = find_or_add_index(
            multi_index[i, :] + np.array([1, 0]), multi_index_update
        )
        neighbor_indices[i, 1], multi_index_update = find_or_add_index(
            multi_index[i, :] + np.array([0, 1]), multi_index_update
        )
        neighbor_indices[i, 2], multi_index_update = find_or_add_index(
            multi_index[i, :] + np.array([1, 1]), multi_index_update
        )

    return uidx, neighbor_indices, multi_index_update


def find_or_add_index(new_index, index_array):
    """find a position of a new index in index_array, if not exisited add the new index in index_array

    Args:
        new_index: new index investigated
        index_array: index array

    Returns:
        position, index_array updated

    """
    uidx_p = np.where(index_array == new_index, True, False)
    mask = np.array(np.prod(uidx_p, axis=1), dtype=bool)
    ni = np.where(mask == True)[0]
    if len(ni) == 0:
        index_array = np.vstack([index_array, new_index])
        return len(index_array) - 1, index_array
    else:
        return ni[0], index_array


if __name__ == "__main__":
    a = np.array([[4, 1], [7, 1], [7, 2], [8, 0], [4, 1]])
    udix, neighbor_indices, multi_index_update = uniqidx_neibouring(a)
