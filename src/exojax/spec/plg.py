"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix

def plg_exomol(logsij0,elower,Nelower=10):
#(nu_grid,nu_lines,elower,alpha_ref,n_Texp,Nlimit=30):
    """PLG for exomol

    """
    get_qlogsij0(logsij0,elower,Nelower=10)
    return None



def get_qlogsij0(logsij0,elower,Nelower=10,Ncrit=10):
    """gether (freeze) lines 

    Args:
       Nelower: # of division of elower between min to max values
       Ncrit: if # of lines in indices exceeds Ncrit

    """
    
    kT0=10000.0    
    expme=np.exp(-elower/kT0)
    margin=1.0
    min_expme=np.min(expme)*np.exp(-margin/kT0)
    max_expme=np.max(expme)*np.exp(margin/kT0)
    expme_grid=np.linspace(min_expme,max_expme,Nelower)
    elower_grid=-np.log(expme_grid)*kT0
    cont,index=npgetix(expme,expme_grid)
    
    unindex,numunique=np.unique(index,return_counts=True)
    mask=numunique>=Ncrit
    
    frozen_index=unindex[mask] #frozen index
    plg_index=np.sort(np.unique(np.concatenate([frozen_index,frozen_index+1])))
    Ng=len(plg_index)
    range_index=np.array(range(0,Ng),dtype=int)
    qsij0=np.zeros(Ng,dtype=np.float64)

    frozen_mask=np.zeros(len(logsij0),dtype=bool)
    for fi in frozen_index:
        mask=(index==fi)
        frozen_mask[mask]=True
        j=range_index[plg_index==fi][0]
        qsij0[j]=qsij0[j]+np.sum((1.0-cont[mask])*np.exp(logsij0[mask]))
        mask=(index==fi+1)
        k=range_index[plg_index==fi+1][0]
        qsij0[k]=qsij0[k]+np.sum(cont[mask]*np.exp(logsij0[mask]))

    qlogsij0=np.log(qsij0)
        
    return qlogsij0,elower_grid,frozen_mask


if __name__ == "__main__":
    import numpy as np
    Nline=100
    logsij0=np.random.rand(Nline)
    elower=np.linspace(2000.0,7000.0,Nline)
    qlogsij0,elower_grid,frozen_mask=get_qlogsij0(logsij0,elower)
    print(qlogsij0)
    print(elower_grid)
    print(frozen_mask)
    print(elower[~frozen_mask])
