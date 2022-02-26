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
    mdb=moldb.MdbExomol('.database/H2O/1H2-16O/POKAZATEL/',nus,crit=1.e-40)
    print("Nline=",len(mdb.A))
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)

    cnu,indexnu,R,pmarray=initspec.init_modit(mdb.nu_lines,nus)

    
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
    molmassH2O=molinfo.molmass("H2O")
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
    dgm_ngammaL=setdgm_exomol(mdb,fT,Parr,R,molmassH2O,res,T0_test,alpha_test)

    u1=0.0
    u2=0.0
    vsini=5.0
    
    #a core driver
    def frun(Tarr,MMR_,Mp,Rp,u1,u2,RV,vsini):        
        g=2478.57730044555*Mp/Rp**2
        SijM_,ngammaLM_,nsigmaDl_=modit.exomol(mdb,Tarr,Parr,R,molmassH2O)    
        xsm_=modit.xsmatrix(cnu,indexnu,R,pmarray,nsigmaDl_,ngammaLM_,SijM_,nus,dgm_ngammaL)
        #abs is used to remove negative values in xsv
        dtaum=dtauM(dParr,jnp.abs(xsm_),MMR_*ONEARR,molmassH2O,g) 
        #CIA                                                                    
        dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
        dtau=dtaum+dtaucH2H2
        sourcef = planck.piBarr(Tarr,nus)
        F0=rtrun(dtau,sourcef)
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        #mu=response.ipgauss_sampling(nusd,nus,Frot,beta_inst,RV)
        mu=Frot
        return mu

    #save
    
    if True:
        Tarr = 1200.0*(Parr/Pref)**0.1
        mu=frun(Tarr,MMR_=0.0059,Mp=33.2,Rp=0.88,u1=0.0,u2=0.0,RV=10.0,vsini=20.0)
        np.savez("h2onoplg.npz",[wav[::-1],mu])
        plt.plot(wav[::-1],mu)
        #plt.plot(wavmic,flux,alpha=0.5,color="C2",label="petit?")
        plt.savefig("h2onoplg.png")
        
    
    
