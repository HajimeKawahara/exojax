"""API for Exomol molecular database

"""

import pandas as pd

def read_trans(transf):
    """Exomol IO for a transition file
    Note:
        i=Upper state counting number
        f=Lower state counting number
        Aif=Einstein coefficient in s-1
        nuif=transition wavenumber in cm-1
        See Table 12 in https://arxiv.org/pdf/1603.05890.pdf

    Args: 
        transf: transition file
    Returns:
        transition data in pandas DataFrame

    """
    try:
        dat = pd.read_csv(transf,sep="\s+",names=("i","f","Aif","nuif"))
    except:
        dat = pd.read_csv(transf,compression="bz2",sep="\s+",names=("i","f","Aif","nuif"))

    return dat 

def read_states(statesf):        
    """Exomol IO for a state file
    Note:
        i=state counting number
        E=state energy
        g=state degeneracy
        J=total angular momentum
        See Table 11 in https://arxiv.org/pdf/1603.05890.pdf            

    Args: 
        statesf: state file
    Returns:
        states data in pandas DataFrame

    """
    try:
        dat = pd.read_csv(statesf,sep="\s+",usecols=range(4),names=("i","E","g","J"))
    except:
        dat = pd.read_csv(statesf,compression="bz2",sep="\s+",usecols=range(4),names=("i","E","g","J"))
        
    return dat

def read_def(deff):
    """Exomol IO for a definition file

    Args: 
        deff: definition file
    Returns:
        temperature exponent n_Texp
        broadening parameter alpha_ref

    """

    dat = pd.read_csv(deff,sep="#",names=("VAL","COMMENT"))
    alpha_ref=None
    texp=None
    for i, com in enumerate(dat["COMMENT"]):
        if "Default value of Lorentzian half-width" in com:
            alpha_ref=float(dat["VAL"][i])
        elif "Default value of temperature exponent" in com:
            n_Texp=float(dat["VAL"][i])

    return n_Texp, alpha_ref

if __name__=="__main__":

#    datf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.def"
    statesf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2"
    d0=read_trans(statesf)
    transf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2"
    d1=read_trans(transf)
    print(len(d0))
    print(len(d1))
