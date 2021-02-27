"""API for Exomol molecular database

"""
import numpy as np
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

def pickup_gE(states,trans):
    """extract g_upper (gup) and E_lower (elower) from states DataFrame and insert them to transition DataFrame.

    Args:
       states: states pandas DataFrame
       trans: transition pandas DataFrame

    Returns:
       A, nu_lines, elower, gup

    Note:
       We first convert pandas DataFrame to ndarray. The state counting numbers in states DataFrame is used as indices of the new array for states (newstates). We remove the state count numbers as the column of newstate, i.e. newstates[:,k] k=0: E, 1: g, 2: J. Then, we can directly use the state counting numbers as mask.


    """
    ndstates=states.to_numpy()
    ndtrans=trans.to_numpy()

    iorig=np.array(ndstates[:,0],dtype=int)
    maxii=int(np.max(iorig)+1) 
    newstates=np.zeros((maxii,np.shape(states)[1]-1),dtype=float)
    newstates[iorig,:]=ndstates[:,1:] 

    i_upper=np.array(ndtrans[:,0],dtype=int)
    i_lower=np.array(ndtrans[:,1],dtype=int)

    #use the state counting numbers (upper and lower) as masking.  
    elower=newstates[i_lower,0]
    gup=newstates[i_upper,1]
    A=ndtrans[:,2]
    nu_lines=ndtrans[:,3]
    
    return A, nu_lines, elower, gup


def pickup_gEslow(states,trans):
    """Slow version to extract g_upper (gup) and E_lower (elower) from states DataFrame and insert them to transition DataFrame.

    Note:
       This code is the (thousands times) slower version of pickup_gE. However, we did not remove this one just because this version is much easier to understand. 

    Args:
       states: states pandas DataFrame
       trans: transition pandas DataFrame

    Returns:
       A, nu_lines, elower, gup

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
        
    A=trans["A"].to_numpy()
    nu_lines=trans["nu_lines"].to_numpy()
    elower=trans["elower"].to_numpy()
    gpu=trans["gup"].to_numpy()
    return A, nu_lines, elower, gup



if __name__=="__main__":
    import time

    check=True
    statesf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2"
    states=read_states(statesf)    
    transf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2"
    trans=read_trans(transf)
    
    ts=time.time()
    A, nu_lines, elower, gup=pickup_gE(states,trans)
    te=time.time()
    
    tsx=time.time()
    if check:
        A_s, nu_lines_s, elower_s, gup_s=pickup_gEslow(states,trans)
    tex=time.time()
    print(te-ts,"sec")
    if check:
        print(tex-tsx,"sec for the slow version")
        print("CHECKING DIFFERENCES...")
        print(np.sum((A_s-A)**2))
        print(np.sum((nu_lines_s-nu_lines)**2))
        print(np.sum((elower_s-elower)**2))
        print(np.sum((gup_s-gup)**2))
    
