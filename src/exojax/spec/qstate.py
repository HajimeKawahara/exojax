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

