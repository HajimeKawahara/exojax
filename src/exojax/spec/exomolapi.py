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
        elif "Default value of temperature exponent" in com:
            n_Texp=float(dat["VAL"][i])
        elif "Element symbol 2" in com:
            molmasssw=True
        elif "No. of transition files" in com:
            ntransf=int(dat["VAL"][i])
        elif "Maximum wavenumber (in cm-1)" in com:
            maxnu=float(dat["VAL"][i])
            #maxnu=20000.0
        elif molmasssw:
            c=np.unique(dat["VAL"][i].strip(" ").split(" "))
            c=np.array(c,dtype=np.float)
            molmass=(np.max(c))
            molmasssw=False

        #SOME DEF FILES CONTAINS ERRORS. THESE ARE THE EXCEPTIONS
        if deff.stem=="12C-16O2__UCL-4000":
            ntransf=20
        if deff.stem=="14N-1H3__CoYuTe":
            maxnu=20000.0
            
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
    """extract g_upper (gup), E_lower (elower), and J_lower and J_upper from states DataFrame and insert them to transition DataFrame.

    Args:
       states: states pandas DataFrame
       trans: transition pandas DataFrame
       trans_lines: By default (False) we use nu_lines computed using the state file, i.e. E_upper - E_lower. If trans_nuline=True, we use the nu_lines in the transition file. Note that some trans files do not this info.


    Returns:
       A, nu_lines, elower, gup, jlower, jupper

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
    jlower=np.array(newstates[i_lower,2],dtype=int)
    jupper=np.array(newstates[i_upper,2],dtype=int)
        
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

    return A, nu_lines, elower, gup, jlower, jupper


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


def read_broad(broadf):
    """ Reading braodening file (.broad)

    Args: 
       broadf: .broad file
    
    Return:
       broadening info in bdat form (pandas), defined by this instance.

    Note:
       See Table 16 in https://arxiv.org/pdf/1603.05890.pdf

    """
    bdat = pd.read_csv(broadf,sep="\s+",\
                names=("code","alpha_ref","n_Texp","jlower","jupper",\
               "kalower","kclower","kaupper","kcupper",\
               "v1lower","v2lower","v3lower",\
               "v1upper","v2upper","v3upper"))
    return bdat

def check_bdat(bdat):
    """ cheking codes in .broad

    Args: 
       bdat: exomol .broad data given by exomolapi.read_broad

    Returns:
       code level: None, a0, a1, other codes unavailable currently,

    """

    def checkcode(code):
        cmask=bdat["code"]==code
        if len(bdat["code"][cmask])>0:
            return True
        else:
            return False

    codelv=None            
    for code in ["a0","a1"]:
        if checkcode(code):
            codelv=code

    return codelv
        
def make_j2b(bdat,alpha_ref_default=0.07,n_Texp_default=0.5,jlower_max=None):
    """compute j2b (code a0, map from jlower to alpha_ref)
    
    Args: 
       bdat: exomol .broad data given by exomolapi.read_broad
       alpha_ref_default: default value      
       n_Texp_default: default value      
       jlower_max: maximum number of jlower    

    Returns:
       j2alpha_ref[jlower] provides alpha_ref for jlower
       j2n_Texp[jlower]  provides nT_exp for jlower

    """
    #a0
    cmask=bdat["code"]=="a0"
    jlower_arr=np.array(bdat["jlower"][cmask],dtype=int)
    alpha_ref_arr=np.array(bdat["alpha_ref"][cmask])
    n_Texp_arr=np.array(bdat["n_Texp"][cmask])

    if jlower_max==None:
        Nblower=np.max(jlower_arr)+1
    else:
        Nblower=np.max([jlower_max,np.max(jlower_arr)])+1        
    j2alpha_ref=np.ones(Nblower)*alpha_ref_default
    j2n_Texp=np.ones(Nblower)*n_Texp_default
    
    j2alpha_ref[jlower_arr]=alpha_ref_arr
    j2n_Texp[jlower_arr]=n_Texp_arr
    
    
    Ndef=Nblower-(np.max(jlower_arr)+1)
    if Ndef>0:
        print("default broadening parameters are used for ",Ndef," J lower states in ",Nblower," states")
    
    return j2alpha_ref, j2n_Texp

def make_jj2b(bdat,j2alpha_ref_def,j2n_Texp_def,jupper_max=None):
    """compute jj2b (code a1, map from (jlower, jupper) to alpha_ref and n_Texp)
    
    Args: 
       bdat: exomol .broad data given by exomolapi.read_broad
       j2alpha_ref_def: default value from a0
       j2n_Texp_def: default value from a0
       jupper_max: maximum number of jupper

    Returns:
       jj2alpha_ref[jlower,jupper] provides alpha_ref for (jlower, jupper)
       jj2n_Texp[jlower,jupper]  provides nT_exp for (jlower, jupper)

    Note:
       The pair of (jlower, jupper) for which broadening parameters are not given, jj2XXX contains None. 

    """
    #a1
    cmask=bdat["code"]=="a1"
    jlower_arr=np.array(bdat["jlower"][cmask],dtype=int)
    jupper_arr=np.array(bdat["jupper"][cmask],dtype=int)
    alpha_ref_arr=np.array(bdat["alpha_ref"][cmask])
    n_Texp_arr=np.array(bdat["n_Texp"][cmask])

    Nblower=len(j2alpha_ref_def)
    
    if jupper_max==None:
        Nbupper=np.max(jupper_arr)+1
    else:
        Nbupper=np.max([jupper_max,np.max(jupper_arr)])+1        

    jj2alpha_ref=j2alpha_ref_def[:,np.newaxis]*np.ones(Nbupper)
    jj2n_Texp=j2n_Texp_def[:,np.newaxis]*np.ones(Nbupper)

    jj2alpha_ref[jlower_arr,jupper_arr]=alpha_ref_arr
    jj2n_Texp[jlower_arr,jupper_arr]=n_Texp_arr
    
    return  jj2alpha_ref, jj2n_Texp

    
if __name__=="__main__":
    import time
    import sys
    import pathlib
    deff=pathlib.Path("/home/kawahara/exojax/examples/luhman16/.database/CO2/12C-16O2/UCL-4000/12C-16O2__UCL-4000.def")
    n_Texp, alpha_ref, molmass, numinf, numtag = read_def(deff)
    print(numtag)
    sys.exit()
    #various broad file
#    broadf="/home/kawahara/exojax/data/broad/12C-16O__H2.broad"
    broadf="/home/kawahara/exojax/data/broad/1H2-16O__H2.broad"
    bdat=read_broad(broadf)
    codelv=check_bdat(bdat)
    print(codelv)
    if codelv=="a0":
        j2alpha_ref, j2n_Texp=make_j2b(bdat,jlower_max=100)
    elif codelv=="a1":
        j2alpha_ref, j2n_Texp=make_j2b(bdat,jlower_max=100)
        jj2alpha_ref, jj2n_Texp=make_jj2b(bdat,j2alpha_ref,j2n_Texp,jupper_max=100)
        print(jj2alpha_ref[1,2])
        print(jj2alpha_ref[1,15])

    
    sys.exit()
    #broad file
    broadf="/home/kawahara/exojax/data/CO/12C-16O/12C-16O__H2.broad"
    bdat=read_broad(broadf)
    j2alpha_ref, j2n_Texp=make_j2b(bdat,jlower_max=100)
    
    #partition file
    pff="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.pf"
    dat=read_pf(pff)

    check=False
    if check:
        print("Checking compution of Elower and gupper.")
    statesf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2"
    states=read_states(statesf)    
    transf="/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2"
    trans=read_trans(transf)
    
    ts=time.time()
    A, nu_lines, elower, gup, jlower, jupper=pickup_gE(states,trans)
#    for i in range(0,len(A)):
#        print(jlower[i],"-",jupper[i])
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
    
    #computing alpha_ref, n_Texp
