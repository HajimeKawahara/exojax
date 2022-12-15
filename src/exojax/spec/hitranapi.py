"""API for HITRAN and HITEMP outside HAPI."""
import numpy as np
from radis.db.classes import get_molecule
from radis.db.classes import get_molecule_identifier
import jax.numpy as jnp
from contextlib import redirect_stdout
import os
with redirect_stdout(open(os.devnull, 'w')):
    import hapi

def molecid_hitran(molec):
    """molec id from Hitran/Hitemp filename or molecule name or molecid itself.

    Args:
       molec: Hitran/Hitemp filename or molecule name or molec id itself.

    Return:
       int: molecid (HITRAN molecular id)
    """
    try:
        hitf = molec.split('_')
        molecid = int(hitf[0])
        return molecid
    except:
        try:
            return get_molecule_identifier(molec)
        except:
            try:
                return get_molecule_identifier(get_molecule(molec))
            except:
                raise ValueError(
                    'Not valid HITRAN/Hitemp file or molecular id or molecular name.'
                )


def make_partition_function_grid_hitran(M, I_list):
    """HITRAN/HITEMP IO for partition function

    Args:
        M: HITRAN molecule number
        I_list: HITRAN isotopologue number list

    Returns:
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT
    """
    gQT = []
    T_gQT = []
    len_idx = []
    for I in I_list:
        gQT.append(hapi.TIPS_2017_ISOQ_HASH[(M, I)])
        T_gQT.append(hapi.TIPS_2017_ISOT_HASH[(M, I)])
        len_idx.append(len(hapi.TIPS_2017_ISOQ_HASH[(M, I)]))

    # pad gQT and T_gQT with the last element
    len_max = np.max(len_idx)
    for idx, iso in enumerate(I_list):
        l_add = [gQT[idx][-1]] * (len_max - len(gQT[idx]))
        gQT[idx] = np.append(gQT[idx], l_add)

        l_add = [T_gQT[idx][-1]] * (len_max - len(T_gQT[idx]))
        T_gQT[idx] = np.append(T_gQT[idx], l_add)

    return jnp.array(gQT), jnp.array(T_gQT)


