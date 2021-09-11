"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix




def plg_exomol(cnu,indexnu,logsij0,elower,elower_grid=None,Nelower=10):
#(nu_grid,nu_lines,elower,alpha_ref,n_Texp,Nlimit=30):
    """PLG for exomol
    
    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       logsij0: log line strength
       elower: elower
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values when elower_grid is not given

    Returns:

    """
    
    kT0=10000.0
    expme=np.exp(-elower/kT0)
    if elower_grid is None:
        margin=1.0
        min_expme=np.min(expme)*np.exp(-margin/kT0)
        max_expme=np.max(expme)*np.exp(margin/kT0)
        expme_grid=np.linspace(min_expme,max_expme,Nelower)
        elower_grid=-np.log(expme_grid)*kT0
    else:
        expme_grid=np.exp(-elower_grid/kT0)
        
    qlogsij0,qcnu,zo=get_qlogsij0(cnu,indexnu,logsij0,expme,expme_grid,Nelower=10)

    print("# of unfrozen lines:",np.sum(1-zo))
    print("# of pseudo lines:",len(qlogsij0[qlogsij0>0.0]))
    
    return qlogsij0,qcnu,elower_grid

def get_qlogsij0(cnu,indexnu,logsij0,expme,expme_grid,Nelower=10,Ncrit=120):
    """gether (freeze) lines 

    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       logsij0: log line strength
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Nelower: # of division of elower between min to max values

    """
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    m=np.max(index)+2
    eindex=index+m*indexnu #extended index

    #frozen criterion
    num_unique=np.bincount(eindex) # number of the lines in a bin    
    frozen_mask=(num_unique>=Ncrit)
    frozen_mask=np.append(frozen_mask,False)
    Nql=len(frozen_mask)
    erange=range(0,np.max(eindex)+2)
    frozen_eindex=np.array(erange)[frozen_mask]
    zo=np.zeros_like(index,dtype=int) #one for frozen/zero for unfrozen
    for fei in frozen_eindex:
        zo[eindex==fei]=1
        
    Sij=np.exp(logsij0)
    #qlogsij0
    qlogsij0=np.append(np.bincount(eindex,weights=Sij*(1.0-cont)*zo),0.0)
    qlogsij0=qlogsij0+np.bincount(eindex+1,weights=Sij*cont*zo)    
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.append(np.bincount(eindex,weights=Sij*zo),0.0)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij*zo)
    qcnu_den[qcnu_den==0.0]=1.0
    
    qcnu_num=np.append(np.bincount(eindex,weights=Sij*cnu*zo),0.0)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu*zo)
    qcnu=qcnu_num/qcnu_den

    #print(np.append(num_unique,0.0)[frozen_mask])
    qlogsij0=qlogsij0.reshape((int(Nql/m),m))
    qcnu=qcnu.reshape((int(Nql/m),m))
    return qlogsij0,qcnu,zo


def plg_exomol_slow(cnu,indexnu,logsij0,elower,elower_grid=None,Nelower=10):
    """slow version of PLG for exomol, but easy to understand

    """
    
    kT0=10000.0
    expme=np.exp(-elower/kT0)
    if elower_grid is None:
        margin=1.0
        min_expme=np.min(expme)*np.exp(-margin/kT0)
        max_expme=np.max(expme)*np.exp(margin/kT0)
        expme_grid=np.linspace(min_expme,max_expme,Nelower)
        elower_grid=-np.log(expme_grid)*kT0
    else:
        expme_grid=np.exp(-elower_grid/kT0)

    qlogsij0=[]
    qcnu=[]
    for inu in np.unique(indexnu):
        mask=indexnu==inu
        cnu_=cnu[mask]
        logsij0_=logsij0[mask]
        expme_=expme[mask]
        each_qlogsij0,each_qcnu=get_qlogsij0_each(cnu_,logsij0_,expme_,expme_grid,Nelower=10)
        qlogsij0.append(each_qlogsij0)
        qcnu.append(each_qcnu)
    qlogsij0=np.array(qlogsij0)
    qcnu=np.array(qcnu)
    
    return qlogsij0,qcnu,elower_grid

def get_qlogsij0_each(cnu,logsij0,expme,expme_grid,Nelower=10):
    """gether (freeze) lines 

    Args:
       Nelower: # of division of elower between min to max values

    """    
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    num_unique=np.bincount(index)
    Sij=np.exp(logsij0)

    #qlogsij0
    qlogsij0=np.append(np.bincount(index,weights=Sij*(1.0-cont)),0.0)
    qlogsij0=(qlogsij0+np.bincount(index+1,weights=Sij*cont))
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.append(np.bincount(index,weights=Sij),0.0)
    qcnu_den=qcnu_den+np.bincount(index+1,weights=Sij)
    qcnu_num=np.append(np.bincount(index,weights=Sij*cnu),0.0)
    qcnu_num=qcnu_num+np.bincount(index+1,weights=Sij*cnu)
    qcnu=qcnu_num/qcnu_den    
    
    return qlogsij0,qcnu



if __name__ == "__main__":
    import numpy as np
    from exojax.spec.initspec import init_modit
    import time
    
    Nline=10001
    logsij0=np.random.rand(Nline)
    elower=np.linspace(2000.0,7000.0,Nline)
    nus=np.random.rand(Nline)*10.0

    #init modit
    Nnus=10
    nu_grid=np.linspace(0,10,Nnus)
    cnu,indexnu,R,pmarray=init_modit(nus,nu_grid)


    ts=time.time()
    qlogsij0,qcnu,elower_grid=plg_exomol(cnu,indexnu,logsij0,elower)
    te=time.time()
    print(te-ts,"sec")

    print("==========================")
    if Nline<20000:
        ts=time.time()
        qlogsij0_slow,qcnu_slow,elower_grid=plg_exomol_slow(cnu,indexnu,logsij0,elower)
        te=time.time()
        print(te-ts,"sec")
        print(np.sum((qlogsij0-qlogsij0_slow)**2))
        print(np.sum((qcnu-qcnu_slow)**2))
        
