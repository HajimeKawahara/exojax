if __name__ == "__main__":
    import numpy as np
    from exojax.spec import plg
    from exojax.spec import moldb, contdb
    from exojax.spec.rtransfer import nugrid
    from exojax.spec import initspec
    import matplotlib.pyplot as plt
    import time
    import tqdm

    import pandas as pd
#    dats=pd.read_csv("Gl229B_spectrum_CH4.dat",names=("wav","flux"),delimiter="\s")
#    wavmic=dats["wav"].values*1.e4
#    ccgs=29979245800.0
#    flux=dats["flux"].values*ccgs

    
    Nx=10000
    nus,wav,res=nugrid(16300.0,16600.0,Nx,unit="AA",xsmode="modit")
    print(res)
    mdb=moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/',nus,crit=1.e-40)
    print("Nline=",len(mdb.A))
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)

    cnu,indexnu,R,pmarray=initspec.init_modit(mdb.nu_lines,nus)

    #make index_gamma
    gammaL_set=mdb.alpha_ref+mdb.n_Texp*(1j) #complex value
    gammaL_set_unique=np.unique(gammaL_set)
    Ngamma=np.shape(gammaL_set_unique)[0]
    index_gamma=np.zeros_like(mdb.alpha_ref,dtype=int)
    alpha_ref_grid=gammaL_set_unique.real
    n_Texp_grid=gammaL_set_unique.imag
    for j,a in tqdm.tqdm(enumerate(gammaL_set_unique)):
        index_gamma=np.where(gammaL_set==a,j,index_gamma)        
    print("done.")
    #-------------------------------------------------------
    Ncrit=10
    Nelower=7

    reshape=False
    ts=time.time()
    qlogsij0,qcnu,num_unique,elower_grid,frozen_mask,nonzeropl_mask=plg.plg_elower_addcon(index_gamma,Ngamma,cnu,indexnu,nus,mdb.logsij0,mdb.elower,Ncrit=Ncrit,Nelower=Nelower,reshape=reshape)    
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
        plt.show()
        import sys
        sys.exit()
    Nnugrid=len(nus)
    
    mdb, cnu, indexnu=plg.gather_lines(mdb,Ngamma,Nnugrid,Nelower,nus,cnu,indexnu,qlogsij0,qcnu,elower_grid,alpha_ref_grid,n_Texp_grid,frozen_mask,nonzeropl_mask)
    
    ##
    from exojax.spec import rtransfer as rt
    from exojax.spec import dit, modit
    from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
    from exojax.spec import planck, response
    from exojax.spec import molinfo
    import jax.numpy as jnp

    
    #atm
    NP=100
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    molmassCH4=molinfo.molmass("CH4")
    mmw=2.33 #mean molecular weight
    mmrH2=0.74
    molmassH2=molinfo.molmass("H2")
    vmrH2=(mmrH2*mmw/molmassH2) #VMR
    
    Pref=1.0 #bar
    ONEARR=np.ones_like(Parr)

    
    # Precomputing gdm_ngammaL
    from exojax.spec.modit import setdgm_exomol
    from jax import jit, vmap
    
    
    fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
    T0_test=np.array([1100.0,1500.0,1100.0,1500.0])
    alpha_test=np.array([0.2,0.2,0.05,0.05])
    res=0.2
    dgm_ngammaL=setdgm_exomol(mdb,fT,Parr,R,molmassCH4,res,T0_test,alpha_test)

    u1=0.0
    u2=0.0
    vsini=5.0
    
    #a core driver
    def frun(Tarr,MMR_,Mp,Rp,u1,u2,RV,vsini):        
        g=2478.57730044555*Mp/Rp**2
        SijM_,ngammaLM_,nsigmaDl_=modit.exomol(mdb,Tarr,Parr,R,molmassCH4)    
        xsm_=modit.xsmatrix(cnu,indexnu,R,pmarray,nsigmaDl_,ngammaLM_,SijM_,nus,dgm_ngammaL)
        #abs is used to remove negative values in xsv
        dtaum=dtauM(dParr,jnp.abs(xsm_),MMR_*ONEARR,molmassCH4,g) 
        #CIA                                                                    
        dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
        dtau=dtaum+dtaucH2H2
        sourcef = planck.piBarr(Tarr,nus)
        F0=rtrun(dtau,sourcef)
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        #mu=response.ipgauss_sampling(nusd,nus,Frot,beta_inst,RV)
        mu=Frot
        return mu

    #test
    if True:
        Tarr = 1200.0*(Parr/Pref)**0.1
        mu=frun(Tarr,MMR_=0.0059,Mp=33.2,Rp=0.88,u1=0.0,u2=0.0,RV=10.0,vsini=20.0)
        plt.plot(wav[::-1],mu)
        #plt.plot(wavmic,flux,alpha=0.5,color="C2",label="petit?")
        plt.show()
        
    
    
