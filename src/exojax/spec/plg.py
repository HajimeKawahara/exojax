"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix




def plg_exomol(cnu,indexnu,logsij0,elower,elower_grid=None,Nelower=10):
#(nu_grid,nu_lines,elower,alpha_ref,n_Texp,Nlimit=30):
    """PLG for exomol

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
    qlogsij0,qcnu,frozen_mask=get_qlogsij0(cnu,indexnu,logsij0,expme,expme_grid,Nelower=10)
    
    return qlogsij0,qcnu,elower_grid,frozen_mask

def get_qlogsij0(cnu,indexnu,logsij0,expme,expme_grid,Nelower=10,Ncrit=10):
    """gether (freeze) lines 

    Args:
       Nelower: # of division of elower between min to max values
       Ncrit: if # of lines in indices exceeds Ncrit

    """

    
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    m=np.max(index)+2
    eindex=index+m*indexnu #extended index
    num_unique=np.bincount(eindex)
    Sij=np.exp(logsij0)

    #qlogsij0
    qlogsij0=np.append(np.bincount(eindex,weights=Sij*(1.0-cont)),0.0)
    qlogsij0=(qlogsij0+np.bincount(eindex+1,weights=Sij*cont))
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.append(np.bincount(eindex,weights=Sij),0.0)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij)
    qcnu_num=np.append(np.bincount(eindex,weights=Sij*cnu),0.0)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu)
    qcnu=qcnu_num/qcnu_den
    

    frozen_mask=0.0
    qlogsij0=qlogsij0.reshape((int(len(qlogsij0)/m),m))
    return qlogsij0,qcnu,frozen_mask


def plg_exomol_slow(cnu,indexnu,logsij0,elower,elower_grid=None,Nelower=10):
#(nu_grid,nu_lines,elower,alpha_ref,n_Texp,Nlimit=30):
    """PLG for exomol

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
        each_qlogsij0,each_qcnu,frozen_mask=get_qlogsij0_slow(cnu_,logsij0_,expme_,expme_grid,Nelower=10)
        qlogsij0.append(each_qlogsij0)
        qcnu.append(each_qcnu)
    qlogsij0=np.array(qlogsij0)
    qcnu=np.array(qcnu)
    
    return qlogsij0,qcnu,elower_grid,frozen_mask

def get_qlogsij0_slow(cnu,logsij0,expme,expme_grid,Nelower=10,Ncrit=10):
    """gether (freeze) lines 

    Args:
       Nelower: # of division of elower between min to max values
       Ncrit: if # of lines in indices exceeds Ncrit

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
    

    frozen_mask=0.0
    
    return qlogsij0,qcnu,frozen_mask



if __name__ == "__main__":
    import numpy as np
    from exojax.spec.initspec import init_modit
    import time
    Nline=10000000
    logsij0=np.random.rand(Nline)
    elower=np.linspace(2000.0,7000.0,Nline)
    nus=np.random.rand(Nline)*10.0

    #init modit
    Nnus=10001
    nu_grid=np.linspace(0,10,Nnus)
    cnu,indexnu,R,pmarray=init_modit(nus,nu_grid)


    ts=time.time()
    qlogsij0,qcn,elower_grid,frozen_mask=plg_exomol(cnu,indexnu,logsij0,elower)
    te=time.time()
    #    print(qcn)
    print(te-ts,"sec")

    print("==========================")
    if Nline<1000000:
        ts=time.time()
        qlogsij0_slow,qcn_slow,elower_grid,frozen_mask=plg_exomol_slow(cnu,indexnu,logsij0,elower)
        te=time.time()
        #    print(qcn)
        print(te-ts,"sec")
        print(np.sum((qlogsij0-qlogsij0_slow)**2))
        
