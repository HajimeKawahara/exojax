"""generate partition function grid using radis.levels.partfunc

    * Why do we need this? ExoJAX requires the derivative of the partition function 
    (ratio) qr(T). Our implimentation uses jax.numpy.interp of the given partition 
    fucntion grid as a function of T.

"""
import warnings
import numpy as np

def generate_partition_function_grid(T_gQT,
                                     molecid,
                                     iso,
                                     partition_function_algorithm="TIPS"):
    if partition_function_algorithm == "TIPS":
        warnings.warn("Use TIPS as a partition function", UserWarning)
        from radis.levels.partfunc import PartFuncTIPS
        Q = PartFuncTIPS(molecid, iso)
    else:
        assert ValueError("partition_function_algorithm is invalid. Use TIPS.")
        
    gQT=[]
    for T in T_gQT:
        gQT.append(Q.at(T=T))
    return np.array(gQT)