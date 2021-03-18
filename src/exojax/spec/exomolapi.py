"""API for Exomol molecular database

"""
import numpy as np
import pandas as pd

def read_def(deff):
    """Exomol IO for a definition file

    Args: 
        deff: definition file
    Returns:
        temperature exponent n_Texp
        broadening parameter alpha_ref
        molecular mass
        numinf: nu minimum for trans
        numtag: tag for wavelength range

    Note:
       For some molecules, ExoMol provides multiple trans files. numinf and numtag are the ranges and identifiers for the multiple trans files.


    """

    dat = pd.read_csv(deff,sep="#",names=("VAL","COMMENT"))
    alpha_ref=None
    texp=None
    molmasssw=False
    n_Texp=None
    ntransf=1
    maxnu=0.0
    for i, com in enumerate(dat["COMMENT"]):
        if "Default value of Lorentzian half-width" in com:
            alpha_ref=float(dat["VAL"][i])
            print("gamma width=",alpha_ref)
        elif "Default value of temperature exponent" in com:
            n_Texp=float(dat["VAL"][i])
            print("T exponent=",n_Texp)
        elif "Element symbol 2" in com:
            molmasssw=True
        elif "No. of transition files" in com:
            ntransf=int(dat["VAL"][i])
        elif "Maximum wavenumber (in cm-1)" in com:
            #maxnu=float(dat["VAL"][i])
            maxnu=20000.0
        elif molmasssw:
            c=np.unique(dat["VAL"][i].strip(" ").split(" "))
            c=np.array(c,dtype=np.float)
            molmass=(np.max(c))
            print("Mol mass=",molmass)
            molmasssw=False
    if ntransf>1:
        dnufile=maxnu/ntransf
        numinf=dnufile*np.array(range(ntransf+1))
        numtag=[]
        for i in range(len(numinf)-1):
            imin='{:05}'.format(int(numinf[i]))
            imax='{:05}'.format(int(numinf[i+1]))
            numtag.append(imin+"-"+imax)
    else:
        numinf=None
        numtag=""
        
    return n_Texp, alpha_ref, molmass, numinf, numtag

def read_pf(pff):
    """Exomol IO for partition file

    Note:
        T=temperature
        QT=partition function

    Args: 
        pff: partition file
    Returns:
        partition data in pandas DataFrame

    """
    dat = pd.read_csv(pff,sep="\s+",names=("T","QT"))
    return dat

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
        dat = pd.read_csv(transf,compression="bz2",sep="\s+",names=("i_upper","i_lower","A","nu_lines"))
    except:
        dat = pd.read_csv(transf,sep="\s+",names=("i_upper","i_lower","A","nu_lines"))

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
        dat = pd.read_csv(statesf,compression="bz2",sep="\s+",usecols=range(4),names=("i","E","g","J"))
    except:
        dat = pd.read_csv(statesf,sep="\s+",usecols=range(4),names=("i","E","g","J"))
        
    return dat


def pickup_gE(states,trans,trans_lines=False):
    """extract g_upper (gup) and E_lower (elower) from states DataFrame and insert them to transition DataFrame.

    Args:
       states: states pandas DataFrame
       trans: transition pandas DataFrame
       trans_lines: By default (False) we use nu_lines computed using the state file, i.e. E_upper - E_lower. If trans_nuline=True, we use the nu_lines in the transition file. Note that some trans files do not this info.


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
    eupper=newstates[i_upper,0]
    gup=newstates[i_upper,1]
    A=ndtrans[:,2]
    
    if trans_lines:
        nu_lines=ndtrans[:,3]
    else:
        nu_lines=eupper-elower

    #See Issue #16
    #import matplotlib.pyplot as plt
    #nu_lines_t=ndtrans[:,3]
    #plt.plot(nu_lines_t-nu_lines,".",alpha=0.03)
    #plt.ylabel("diff nu from trans and state (cm-1)")
    #plt.xlabel("wavenuber (cm-1)")
    #plt.savefig("nudiff.png", bbox_inches="tight", pad_inches=0.0)
    #plt.show()

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

    pff="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.pf"
    dat=read_pf(pff)
    
    print("Checking compution of Elower and gupper.")
    check=False
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
    
