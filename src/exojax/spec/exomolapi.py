"""API for Exomol molecular database

"""

import pandas as pd

def read_trans(transf):
    """Exomol IO for a transition file
    Note:
        i_upper=Upper state counting number
        i_lower=Lower state counting number
        A=Einstein coefficient in s-1
        nu_lines=transition wavenumber in cm-1
        See Table 12 in https://arxiv.org/pdf/1603.05890.pdf

    Args: 
        transf: transition file
    Returns:
        transition data in pandas DataFrame

    """
    try:
        dat = pd.read_csv(transf,sep="\s+",names=("i_upper","i_lower","A","nu_lines"))
    except:
        dat = pd.read_csv(transf,compression="bz2",sep="\s+",names=("i_upper","i_lower","A","nu_lines"))

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

def pick_gE(states,trans):
    """extract g_upper (gup) and E_lower (elower) from states DataFrame and insert them to transition DataFrame.

    Args:
       states: states pandas DataFrame
       trans: transition pandas DataFrame
    Returns:
       transition pandas DataFrame, inserted gup and elower

    """
    import tqdm
    E=states["E"].values
    g=states["g"].values
    # insert new columns in transition array
    trans["gup"]=0
    trans["elower"]=0.0
    for k,i in tqdm.tqdm(enumerate(states["i"])):
        #transition upper state
        mask_upper=(trans["i_upper"]==i) 
        trans["gup"][mask_upper]=g[k]
        #transition lower state
        mask_lower=(trans["i_lower"]==i) 
        trans["elower"][mask_lower]=E[k]
    return trans

if __name__=="__main__":
#    datf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.def"
    statesf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2"
    states=read_states(statesf)    
    transf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2"
    trans=read_trans(transf)
    trans=pick_gE(states,trans)
    print(trans)
