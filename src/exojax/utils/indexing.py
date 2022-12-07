import numpy as np
import tqdm
import warnings

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
        warnings.warn("input should be contiguous.",UserWarning)
    uniq = np.unique(x.view(x.dtype.descr * x.shape[1]))
    return uniq.view(x.dtype).reshape(-1, x.shape[1])

def uniqidx(input_array):
    """ compute indices based on uniq values of the input M-dimensional array.                                                   
                                                                                                                        
    Args:                                                                                                               
        input_array: input array (N,M), will use unique M-dim vectors                                                             
                                                                                                                        
    Returns:                                                                                                            
        unique index, unique value                                                                                      
                                                                                                                        
    Examples:                                                                                                           
                                                                                                                        
        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])                                                           
        >>> uidx, uval=uniqidx(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]                                        
                                                                                                                        
    """
    N, _ = np.shape(input_array)
    #uniqvals = np.unique(input_array, axis=0)
    uniqvals = unique_rows(input_array)
    uidx = np.zeros(N, dtype=int)
    uidx_p = np.where(input_array == uniqvals[0], True, False)
    uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = 0
    for i, uv in enumerate(tqdm.tqdm(uniqvals[1:], desc="uniqidx")):
        uidx_p = np.where(input_array == uv, True, False)
        uidx[np.array(np.prod(uidx_p, axis=1), dtype=bool)] = i + 1
    return uidx, uniqvals


def uniqidx_neibouring(index_array):
    """ compute indices based on uniq values of the input index array and input index + one vector  
                                                                                                                        
    Args:                                                                                                               
        index_array: input index array (N,M), will use unique M-dim vectors                                                             
                                                                                                                        
    Returns:                                                                                                            
        unique index (udix)
        neibouring index (nidx) for udix [N_uidx, 3]
        multi index as a function of nidx
                                                                                                                        
    """
    uidx, multi_index = uniqidx(index_array)
    multi_index_update=np.copy(multi_index)
    Nuidx = np.max(uidx)+1
    neighbor_indices=np.zeros((Nuidx,3),dtype=int)
    for i in range(0, Nuidx):
        neighbor_indices[i,0], multi_index_update = find_or_add_index(multi_index[i,:]+np.array([1,0]), multi_index_update)
        neighbor_indices[i,1], multi_index_update = find_or_add_index(multi_index[i,:]+np.array([0,1]), multi_index_update)
        neighbor_indices[i,2], multi_index_update = find_or_add_index(multi_index[i,:]+np.array([1,1]), multi_index_update)
        
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
        index_array = np.vstack([index_array,new_index])
        return len(index_array) - 1, index_array
    else:
        return ni[0], index_array

if __name__ == "__main__":
    a = np.array([[4, 1], [7, 1], [7, 2], [8, 0], [4, 1]])
    udix, neighbor_indices, multi_index_update = uniqidx_neibouring(a)
