from exojax.utils.indexing import uniqidx, find_or_add_index, uniqidx_neibouring
import numpy as np


def test_uniqidx_2D():
    a = np.array([[4, 1], [7, 1], [7, 2], [7, 1], [8, 0], [4, 1]])
    uidx, val = uniqidx(a)
    ref = np.array([0, 1, 2, 1, 3, 0])
    assert np.all(uidx - ref == 0.0)
    refval = np.array([[4, 1], [7, 1], [7, 2], [8, 0]])
    assert np.all(val - refval == 0.0)


def test_uniqidx_3D():
    a = np.array([[4, 1, 2], [7, 1, 2], [7, 2, 2], [7, 1, 2], [8, 0, 2],
                  [4, 1, 1]])
    uidx, val = uniqidx(a)
    ref = np.array([1, 2, 3, 2, 4, 0])
    assert np.all(uidx - ref == 0.0)
    refval = np.array([[4, 1, 1], [4, 1, 2], [7, 1, 2], [7, 2, 2], [8, 0, 2]])
    assert np.all(val - refval == 0.0)


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
    assert np.all(udix == [0,1,2,3,0])
    index_of_uidx = 3 # can be 0,1 ... , np.max(udix) 
    assert np.all(multi_index_update[index_of_uidx] == [8,0])
    i,j,k=neighbor_indices[index_of_uidx,:]
    assert np.all(multi_index_update[i] == multi_index_update[index_of_uidx] + np.array([1,0])) 
    assert np.all(multi_index_update[j] == multi_index_update[index_of_uidx] + np.array([0,1])) 
    assert np.all(multi_index_update[k] == multi_index_update[index_of_uidx] + np.array([1,1])) 
    
if __name__ == "__main__":
    #test_uniqidx_2D()
    #test_uniqidx_3D()
    #test_find_or_add_index()
    test_uniqidx_neibouring() 