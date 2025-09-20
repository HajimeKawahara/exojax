"""Quantum States Module

"""
import numpy as np


def m_transition_state(jlower, branch):
    """compute m-value of the transition from rotational state

    Args:
        jlower (int): lower rotational quantum state
        branch (int): jupper - jlower

    Return:
        int: m

    Notes:
        m = Jlower + 1 for R-branch (branch=1)
        m = Jlower  for P- and Q- branch (branch=-1 and 0)

    """
    return (2*jlower + branch + 1)//2

def branch_to_number(branch_str, fillvalue=None):
    """convert branch: object to number (float)
    Args:
        branch_str (np.ndarray): Branch type (R, P, Q)
    Returns:
        np.ndarray: jupper - jlower
    """
    branch_str = np.asarray(branch_str, dtype=object)
    is_nan = np.vectorize(lambda x: isinstance(x, float) and np.isnan(x))(branch_str)

    if fillvalue is None:
        valid = {"R", "P", "Q"}
        if np.any(is_nan) or not np.all(np.isin(branch_str[~is_nan], list(valid))):
            raise ValueError("Invalid or NaN branch values found.")

    branch = np.full(branch_str.shape, fillvalue, dtype=float)

    branch[branch_str == "R"] = 1
    branch[branch_str == "P"] = -1
    branch[branch_str == "Q"] = 0

    return branch