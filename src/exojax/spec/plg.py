"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix

def plg_exomol(elower,Nelower=10):
#(nu_grid,nu_lines,elower,alpha_ref,n_Texp,Nlimit=30):
    """PLG for exomol

    """
    kT0=10000.0    
    expme=np.exp(-elower/kT0)
    
    min_expme=np.min(expme)
    max_expme=np.max(expme)
    expme_grid=np.linspace(min_expme,max_expme,Nelower)
    elower_grid=-np.log(expme_grid)*kT0
    cont,index=npgetix(expme,expme_grid)
    for i in range(0,len(cont)):
        print(elower[i],cont[i],index[i],elower_grid[index[i]])
    
if __name__ == "__main__":
    import numpy as np
    
    elower=np.linspace(2000.0,7000.0,100)
    plg_exomol(elower)
