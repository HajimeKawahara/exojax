"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix
import tqdm



def plg_exomol(cnu,indexnu,Nnugrid,logsij0,elower,elower_grid=None,Nelower=10,Ncrit=0,reshape=True):
    """PLG for exomol
    
    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of Ngrid
       logsij0: log line strength
       elower: elower
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values when elower_grid is not given
       Ncrit: frrozen line number per bin

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
        Nelower=len(expme_grid)
        
    qlogsij0,qcnu,num_unique,frozen_mask=get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=Ncrit)

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Nnugrid,Nelower)
        qcnu=qcnu.reshape(Nnugrid,Nelower)
        num_unique=num_unique.reshape(Nnugrid,Nelower)
    
    print("# of unfrozen lines:",np.sum(~frozen_mask))
    print("# of pseudo lines:",len(qlogsij0[qlogsij0>0.0]))
    
    return qlogsij0,qcnu,num_unique,elower_grid

def get_qlogsij0(cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=0):
    """gether (freeze) lines 

    Args:
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       logsij0: log line strength
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Nelower: # of division of elower between min to max values

    """
    m=len(expme_grid)
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    eindex=index+m*indexnu #extended index
    
    #frozen criterion
    Ng=m*Nnugrid
    num_unique=np.bincount(eindex,minlength=Ng) # number of the lines in a bin
    
    lmask=(num_unique>=Ncrit)
    erange=range(0,Ng)
    frozen_eindex=np.array(erange)[lmask]
    frozen_mask=np.isin(eindex,frozen_eindex)
        
    Sij=np.exp(logsij0)
    #qlogsij0
    qlogsij0=np.bincount(eindex,weights=Sij*(1.0-cont)*frozen_mask,minlength=Ng)
    qlogsij0=qlogsij0+np.bincount(eindex+1,weights=Sij*cont*frozen_mask,minlength=Ng)    
    qlogsij0=np.log(qlogsij0)

    #qcnu
    qcnu_den=np.bincount(eindex,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den=qcnu_den+np.bincount(eindex+1,weights=Sij*frozen_mask,minlength=Ng)
    qcnu_den[qcnu_den==0.0]=1.0
    
    qcnu_num=np.bincount(eindex,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu_num=qcnu_num+np.bincount(eindex+1,weights=Sij*cnu*frozen_mask,minlength=Ng)
    qcnu=qcnu_num/qcnu_den

    return qlogsij0,qcnu,num_unique,frozen_mask




if __name__ == "__main__":
    import numpy as np
    from exojax.spec.initspec import init_modit
    import time
    import matplotlib.pyplot as plt
    np.random.seed(1)
    Nline=10001
#    Nline=1001

    logsij0=np.random.rand(Nline)
    elower=(np.random.rand(Nline))**0.2*5000.+2000.0
    nus=np.random.rand(Nline)*10.0

    #init modit
    Nnus=101
    nu_grid=np.linspace(0,10,Nnus)
    cnu,indexnu,R,pmarray=init_modit(nus,nu_grid)
    
    Ncrit=10
    ts=time.time()
    qlogsij0,qcnu,num_unique,elower_grid=plg_exomol(cnu,indexnu,Nnus,logsij0,elower,Ncrit=Ncrit)
    te=time.time()
    print(te-ts,"sec")
    num_unique=np.array(num_unique,dtype=float)
    num_unique[num_unique<Ncrit]=None

    fig=plt.figure(figsize=(10,3))
    ax=fig.add_subplot(211)
    c=plt.imshow(num_unique.T)
    plt.colorbar(c,shrink=0.2)
    ax=fig.add_subplot(212)

    c=plt.imshow(qlogsij0.T)
    plt.colorbar(c,shrink=0.2)
    plt.show()
