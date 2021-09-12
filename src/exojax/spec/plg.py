"""Pseudo line generator

"""
import numpy as np
from exojax.spec.dit import npgetix
import tqdm



def plg_elower_addcon(indexa,Na,cnu,indexnu,nu_grid,logsij0,elower,elower_grid=None,Nelower=10,Ncrit=0,reshape=False):
    """PLG for elower w/ an additional condition
    
    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       nugrid: nu grid
       logsij0: log line strength
       elower: elower
       elower_grid: elower_grid (optional)
       Nelower: # of division of elower between min to max values when elower_grid is not given
       Ncrit: frrozen line number per bin
       form: "reshape","gather" 

    Returns:
       qlogsij0: pseudo logsij0
       qcnu: pseudo cnu
       num_unique: number of lines in grids
       elower_grid: elower of pl
       frozen_mask: mask for frozen lines into pseudo lines 
       nonzeropl_mask: mask for pseudo-lines w/ non-zero

    """
    Nnugrid=len(nu_grid)
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
        
    qlogsij0,qcnu,num_unique,frozen_mask=get_qlogsij0_addcon(indexa,Na,cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=Ncrit)
    nonzeropl_mask=qlogsij0>-np.inf  #note that num_unique>=Ncrit is not nonzeropl_mask
    Nline=len(logsij0)
    Nunf=np.sum(~frozen_mask)
    Npl=len(qlogsij0[nonzeropl_mask])
    print("# of original lines:",Nline)        
    print("# of unfrozen lines:",Nunf)
    print("# of pseudo lines:",Npl)
    print("# compression:",(Npl+Nunf)/Nline)
    print("# of pseudo lines:",Npl)
    arrone=np.ones((Na,Nelower))
    qnu_grid=arrone[:,np.newaxis,:]*nu_grid[np.newaxis,:,np.newaxis]

    if reshape==True:
        qlogsij0=qlogsij0.reshape(Na,Nnugrid,Nelower)
        qcnu=qcnu.reshape(Na,Nnugrid,Nelower)
        num_unique=num_unique.reshape(Na,Nnugrid,Nelower)
    else:
        qnu_grid=qnu_grid.flatten

            
    return qlogsij0,qcnu,qnu_grid,num_unique,elower_grid,frozen_mask,nonzeropl_mask

def get_qlogsij0_addcon(indexa,Na,cnu,indexnu,Nnugrid,logsij0,expme,expme_grid,Nelower=10,Ncrit=0):
    """gether (freeze) lines w/ additional indexing

    Args:
       indexa: the indexing of the additional condition
       Na: the number of the additional condition grid
       cnu: contribution of wavenumber for LSD
       indexnu: index of wavenumber
       Nnugrid: number of nu grid
       logsij0: log line strength
       expme: exp(-elower/kT0)
       expme_grid: exp(-elower/kT0)_grid
       Nelower: # of division of elower between min to max values

    """
    m=len(expme_grid)
    n=m*Nnugrid
    Ng=n*Na
    
    cont,index=npgetix(expme,expme_grid) #elower contribution and elower index of lines
    eindex=index+m*indexnu+n*indexa #extended index elower,nu,a
    
    #frozen criterion
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

def plg_elower(cnu,indexnu,Nnugrid,logsij0,elower,elower_grid=None,Nelower=10,Ncrit=0,reshape=True):
    """PLG for elower
    
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
    print("tmp")
    from exojax.spec import moldb
    from exojax.spec.rtransfer import nugrid
    from exojax.spec import initspec
    import matplotlib.pyplot as plt
    import time
    import tqdm
    Nx=10000
    nus,wav,res=nugrid(16300.0,16600.0,Nx,unit="AA",xsmode="modit")
    print(res)
    mdb=moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/',nus,crit=1.e-40)
    print(len(mdb.A))

    cnu,indexnu,R,pmarray=initspec.init_modit(mdb.nu_lines,nus)

    #make index_gamma
    gammaL_set=mdb.alpha_ref+mdb.n_Texp*(1j) #complex value
    gammaL_set_unique=np.unique(gammaL_set)
    Ngamma=np.shape(gammaL_set_unique)[0]
    index_gamma=np.zeros_like(mdb.alpha_ref,dtype=int)
    alpha_ref_grid=gammaL_set_unique.real
    n_Texp_grid=gammaL_set_unique.imag
    print(alpha_ref_grid)
    for j,a in tqdm.tqdm(enumerate(gammaL_set_unique)):
        index_gamma=np.where(gammaL_set==a,j,index_gamma)        
    print("done.")
    #-------------------------------------------------------
    import sys
    sys.exit()
    Ncrit=10
    Nelower=7

    reshape=True
    ts=time.time()
    qlogsij0,qcnu,qnu_grid,num_unique,elower_grid,frozen_mask,nonzeropl_mask=plg_elower_addcon(index_gamma,Ngamma,cnu,indexnu,nus,mdb.logsij0,mdb.elower,Ncrit=Ncrit,Nelower=Nelower,reshape=reshape)    
    te=time.time()
    print(te-ts,"sec")
    print("elower_grid",elower_grid)

    if reshape:
        num_unique=np.array(num_unique,dtype=float)
        num_unique[num_unique<Ncrit]=None
        fig=plt.figure(figsize=(10,4.5))
        ax=fig.add_subplot(311)
        c=plt.imshow(num_unique[0,:,:].T)
        #    c=plt.imshow(np.sum(num_unique[:,:,:],axis=0).T)
        plt.colorbar(c,shrink=0.2)
        ax.set_aspect(0.1/ax.get_data_ratio())                
        ax=fig.add_subplot(312)
        c=plt.imshow(qlogsij0[0,:,:].T)
        plt.colorbar(c,shrink=0.2)
        ax.set_aspect(0.1/ax.get_data_ratio())
        ax=fig.add_subplot(313)
        c=plt.imshow(qnu_grid[0,:,:].T)
        plt.colorbar(c,shrink=0.2)
        ax.set_aspect(0.1/ax.get_data_ratio())        

        plt.show()
        import sys
        sys.exit()
    
    #gathering
    mdb.logsij0=np.hstack([qlogsij0[nonzeropl_mask],mdb.logsij0[~frozen_mask]])
    mdb.elower=np.hstack([elower_grid,mdb.elower[~frozen_mask]])
    cnu=np.hstack([qcnu[nonzeropl_mask],mdb.cnu[~frozen_mask]])
    indexnu=np.hstack([qcnu[nonzeropl_mask],mdb.cnu[~frozen_mask]])
    mdb.nu_lines=np.hstack([qnu_line[nonzeropl_mask],mdb.nu_line[~frozen_mask]])
    import jax.numpy as jnp
    mdb,dev_nu_lines=jnp.array(mdb.nu_lines)
    mdb.n_Texp
    mdb.alpha_ref
    mdb.A=jnp.zeros_like(mdb.A)

    # Precomputing gdm_ngammaL
    from exojax.spec.modit import setdgm_exomol
    from jax import jit, vmap
    
    fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
    T0_test=np.array([1100.0,1500.0,1100.0,1500.0])
    alpha_test=np.array([0.2,0.2,0.05,0.05])
    res=0.2
    dgm_ngammaL=setdgm_exomol(mdbCH4,fT,Parr,R,molmassCH4,res,T0_test,alpha_test)
    
    #check dgm
    if False:
        from exojax.plot.ditplot import plot_dgmn
        Tarr=1300.*(Parr/Pref)**0.1
        SijM_CH4,ngammaLM_CH4,nsigmaDl_CH4=modit.exomol(mdb,Tarr,Parr,R,molmassCH4)
        plot_dgmn(Parr,dgm_ngammaL,ngammaLM_CH4,0,6)
        plt.show()

    #a core driver
    def frun(Tarr,MMR_CH4,Mp,Rp,u1,u2,RV,vsini):        
        g=2478.57730044555*Mp/Rp**2
        SijM_CH4,ngammaLM_CH4,nsigmaDl_CH4=modit.exomol(mdbCH4,Tarr,Parr,R,molmassCH4)    
        xsm_CH4=modit.xsmatrix(cnu,indexnu,R,pmarray,nsigmaDl_CH4,ngammaLM_CH4,SijM_CH4,nus,dgm_ngammaL)
        #abs is used to remove negative values in xsv
        dtaumCH4=dtauM(dParr,jnp.abs(xsm_CH4),MMR_CH4*ONEARR,molmassCH4,g) 
        #CIA                                                                    
        dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
        dtau=dtaumCH4+dtaucH2H2
        sourcef = planck.piBarr(Tarr,nus)
        F0=rtrun(dtau,sourcef)/norm
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        mu=response.ipgauss_sampling(nusd,nus,Frot,beta_inst,RV)
        return mu

    #test
    if False:
        Tarr = 1200.0*(Parr/Pref)**0.1
        mu=frun(Tarr,MMR_CH4=0.0059,Mp=33.2,Rp=0.88,u1=0.0,u2=0.0,RV=10.0,vsini=20.0)
        plt.plot(wavd,mu)
        plt.show()
        
    
    
    
