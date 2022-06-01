import numpy as np
import tqdm

def uniqidx_2D(a):
    """ compute indices based on uniq values of the input array (2D).                                                   
                                                                                                                        
    Args:                                                                                                               
        a: input array (N,M), will use unique M-dim vectors                                                             
                                                                                                                        
    Returns:                                                                                                            
        unique index, unique value                                                                                      
                                                                                                                        
    Examples:                                                                                                           
                                                                                                                        
        >>> a=np.array([[4,1],[7,1],[7,2],[7,1],[8,0],[4,1]])                                                           
        >>> uidx, uval=uniqidx_2D(a) #->[0,1,2,1,3,0], [[4,1],[7,1],[7,2],[8,0]]                                        
                                                                                                                        
    """
    N,_=np.shape(a)
    uniqvals=np.unique(a,axis=0)
    uidx=np.zeros(N,dtype=int)
    uidx_p=np.where(a==uniqvals[0], True, False)
    uidx[np.array(np.prod(uidx_p,axis=1),dtype=bool)]=0
    for i,uv in enumerate(tqdm.tqdm(uniqvals[1:],desc="uniqidx")):
        uidx_p=np.where(a==uv, True, False)
        uidx[np.array(np.prod(uidx_p,axis=1),dtype=bool)]=i+1
    return uidx, uniqvals
