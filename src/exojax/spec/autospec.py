"""Automatic Opacity and Spectrum Generator
   
"""
import time
from exojax.spec import defmol
from exojax.spec import defcia
from exojax.spec import moldb
from exojax.spec import contdb 
from exojax.spec.opacity import xsection
from exojax.spec.hitran import SijT, doppler_sigma,  gamma_natural, gamma_hitran
from exojax.spec import planck
from exojax.spec.exomol import gamma_exomol
from exojax.spec import molinfo
from exojax.spec.rtransfer import rtrun, pressure_layer, dtauM, dtauCIA, check_nugrid
from exojax.spec.make_numatrix import make_numatrix0
from exojax.spec.lpf import xsmatrix
from exojax.spec import response
import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import pathlib
import tqdm
__all__ = ['AutoXS','AutoRT']


class AutoXS(object):
    """exojax auto cross section generator
    
    """
    def __init__(self,nus,database,molecules,databasedir=".database",memory_size=30):
        """
        Args:
           nus: wavenumber bin (cm-1)
           database: database= HITRAN, HITEMP, ExoMol
           molecules: molecule name

        """
        self.molecules=molecules
        self.database=database
        self.nus=nus
        self.databasedir=databasedir
        self.memory_size=memory_size
        
        self.identifier=defmol.search_molfile(database,molecules)
        print(self.identifier)
        if self.identifier is None:
            print("ERROR: "+molecules+" is an undefined molecule. Add your molecule in defmol.py and do pull-request!")
        else:
            self.init_database()
        
    def init_database(self):
        if self.database=="HITRAN" or self.database=="HITEMP":
            molpath=pathlib.Path(self.databasedir)/pathlib.Path(self.identifier)
            self.mdb=moldb.MdbHit(molpath,nurange=[self.nus[0],self.nus[-1]])
        elif self.database=="ExoMol":
            molpath=pathlib.Path(self.databasedir)/pathlib.Path(self.identifier)
            molpath=str(molpath)
            self.mdb=moldb.MdbExomol(molpath,nurange=[self.nus[0],self.nus[-1]])
        else:
            print("Select database from HITRAN, HITEMP, ExoMol.")

    def linest(self,T,P):
        """line strength 

        Args: 
           T: temperature (K)
           P: pressure (bar)

        Returns:
           line strength (cm)

        """
        mdb=self.mdb
        if self.database == "ExoMol":
            qt=mdb.qr_interp(T)
        elif self.database == "HITRAN" or self.database == "HITEMP":
            qt=mdb.Qr_line(T)
            
        Sij=SijT(T,mdb.logsij0,mdb.nu_lines,mdb.elower,qt)
        return Sij
        
    def xsection(self,T,P):
        """cross section

        Args: 
           T: temperature (K)
           P: pressure (bar)

        Returns:
           cross section (cm2)

        """

        mdb=self.mdb
        if self.database == "ExoMol":
            gammaL = gamma_exomol(P,T,mdb.n_Texp,mdb.alpha_ref)\
                     + gamma_natural(mdb.A)
            molmass=mdb.molmass
        elif self.database == "HITRAN" or self.database == "HITEMP":
            gammaL = gamma_hitran(P,T, P, mdb.n_air, \
                      mdb.gamma_air, mdb.gamma_self) \
                      + gamma_natural(mdb.A)
            molmass=molinfo.molmass(self.molecules)
        
        Sij=self.linest(T,P)
        sigmaD=doppler_sigma(mdb.nu_lines,T,molmass)
        nu0=mdb.nu_lines
        xsv=xsection(self.nus,nu0,sigmaD,gammaL,Sij,memory_size=self.memory_size)
        return xsv

    def xsmatrix(self,Tarr,Parr):
        """cross section matrix
        Args: 
           Tarr: temperature layer (K)
           Parr: pressure layer (bar)
        Returns:
           cross section (cm2)
        """
        mdb=self.mdb
        if self.database == "ExoMol":
            qt=vmap(mdb.qr_interp)(Tarr)
            gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
                              (Parr,Tarr,mdb.n_Texp,mdb.alpha_ref)
            gammaLMN=gamma_natural(mdb.A)
            #gammaLM=gammaLMP[:,None]+gammaLMN[None,:]
            gammaLM=gammaLMP+gammaLMN[None,:]
            self.molmass=mdb.molmass
            SijM=jit(vmap(SijT,(0,None,None,None,0)))\
                  (Tarr,mdb.logsij0,mdb.nu_lines,mdb.elower,qt)

        elif self.database == "HITRAN" or self.database == "HITEMP":
            qt=mdb.Qr_layer(Tarr)
            gammaLM = jit(vmap(gamma_hitran,(0,0,0,None,None,None)))\
                      (Parr,Tarr,Parr, mdb.n_air, mdb.gamma_air, mdb.gamma_self)\
                      + gamma_natural(mdb.A)
            self.molmass=molinfo.molmass(self.molecules)
            SijM=jit(vmap(SijT,(0,None,None,None,0)))\
                  (Tarr,mdb.logsij0,mdb.nu_lines,mdb.elower,qt)
            
        sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
                 (mdb.nu_lines,Tarr,self.molmass)
        nu0=mdb.nu_lines

        #####
        #numatrix=make_numatrix0(nus,nu0)
        #xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM)
        ####

        memory_size=15.0
        d=int(memory_size/(len(nu0)*4/1024./1024.))
        Ni=int(len(self.nus)/d)        
        d2=100
        Nlayer=np.shape(SijM)[0]
        Nline=np.shape(SijM)[1]
        Nj=int(Nline/d2)
        xsm=[]
        for i in tqdm.tqdm(range(0,Ni+1)):
            s=int(i*d);e=int((i+1)*d);e=min(e,len(self.nus))
            xsmtmp=np.zeros((Nlayer,e-s))
            #line 
            for j in range(0,Nj+1):
                s2=int(j*d2);e2=int((j+1)*d2);e2=min(e2,Nline)
                numatrix=make_numatrix0(self.nus[s:e],nu0[s2:e2])
                xsmtmp=xsmtmp+\
                        xsmatrix(numatrix,sigmaDM[:,s2:e2],gammaLM[:,s2:e2],SijM[:,s2:e2])
            if i==0:
                xsm=np.copy(xsmtmp.T)
            else:
                xsm = np.concatenate([xsm,xsmtmp.T])
        xsm=xsm.T
        
        return xsm
        
class AutoRT(object):
    
    """exojax auto radiative transfer
    
    """
    def __init__(self,nus,gravity,mmw,Tarr,Parr,dParr=None,databasedir=".database"):
        """
        Args:
           nus: wavenumber bin (cm-1)
           gravity: gravity (cm/s2)
           mmw: mean molecular weight of the atmosphere
           Tarr: temperature layer (K)
           Parr: pressure layer (bar)
           dParr: delta pressure (bar) optional
        """
        self.nus=nus
        self.gravity=gravity
        self.mmw=mmw
        self.nlayer=len(Tarr)        
        self.Tarr=Tarr
        self.Parr=Parr
        
        if check_nugrid(nus):
            print("nu grid is evenly spaced in log space (ESLOG).")
        else:
            print("**************************************************")
            print("WARNING!")
            print("nu grid is NOT evenly spaced in log space (ESLOG).")
            print("astro/inst responses won't work properly.")
            print("Consider to use rtransfer.nugrid instead.")
            print("**************************************************")
            
        if dParr is None:
            from exojax.utils.chopstacks import buildwall 
            wParr=buildwall(Parr)
            self.dParr=wParr[1:]-wParr[0:-1]
        else:
            self.dParr=dParr
        self.databasedir=databasedir
        
        self.sourcef=planck.piBarr(self.Tarr,self.nus)
        self.dtau=np.zeros((self.nlayer,len(nus)))

    def addmol(self,database,molecules,mmr):
        """
        Args:
           database: database= HITRAN, HITEMP, ExoMol
           molecules: molecule name
           mmr: mass mixing ratio (float or ndarray for the layer)
        """
        mmr=mmr*np.ones_like(self.Tarr)
        axs=AutoXS(self.nus,database,molecules)
        xsm=axs.xsmatrix(self.Tarr,self.Parr) 
        dtauMx=dtauM(self.dParr,xsm,mmr,axs.molmass,self.gravity)
        self.dtau=self.dtau+dtauMx

    def addcia(self,interaction,mmr1,mmr2):
        """
        Args:
           interaction: e.g. H2-H2, H2-He
           mmr1: mass mixing ratio for molecule 1
           mmr2: mass mixing ratio for molecule 2

        """
        mol1,mol2=defcia.interaction2mols(interaction)
        molmass1=molinfo.molmass(mol1)
        molmass2=molinfo.molmass(mol2)
        vmr1=(mmr1*self.mmw/molmass1)
        vmr2=(mmr2*self.mmw/molmass2)
        ciapath=pathlib.Path(self.databasedir)/pathlib.Path(defcia.ciafile(interaction))
        cdb=contdb.CdbCIA(str(ciapath),[self.nus[0],self.nus[-1]])
        dtauc=dtauCIA(self.nus,self.Tarr,self.Parr,self.dParr,vmr2,vmr2,\
                      self.mmw,self.gravity,cdb.nucia,cdb.tcia,cdb.logac)
        self.dtau=self.dtau+dtauc
        
    def rtrun(self):
        """running radiative transfer
        
        Returns:
           spectrum (F0)

        """
        self.F0=rtrun(self.dtau,self.sourcef)
        return self.F0

    def spectrum(self,nuobs,R,vsini,RV,u1=0.0,u2=0.0,zeta=0.,betamic=0.,direct=True):
        """generating spectrum
        
        Args:
           nuobs: observation wavenumber array
           R: resolving power
           vsini: vsini for a stellar/planet rotation
           RV: radial velocity (km/s)
           u1: Limb-darkening coefficient 1
           u2: Limb-darkening coefficient 2
           zeta: macroturbulence distrubunce (km/s) in the radial-tangential model (Gray2005)
           betamic: microturbulence beta (STD, km/s)
           direct: True=use rigidrot/ipgauss_sampling, False=use rigidrot2, ipgauss2, sampling

        Returns:
           spectrum (F)

        """

        self.nuobs=nuobs
        self.R=R
        self.vsini=vsini
        self.u1=u1
        self.u2=u2
        self.zeta=zeta
        self.betamic=betamic
        self.RV=RV
        
        c=299792.458
        self.betaIP=c/(2.0*np.sqrt(2.0*np.log(2.0))*self.R)
        beta=np.sqrt((self.betaIP)**2+(self.betamic)**2)
        ts=time.time()
        F0=self.rtrun()
        te=time.time()
        print("radiative transfer",te-ts,"s")
        if len(self.nus)<50000 and direct==True:
            ts=time.time()
            Frot=response.rigidrot(self.nus,F0,self.vsini,u1=self.u1,u2=self.u2)
            te=time.time()
            print("rotation",te-ts,"s")
            ts=time.time()
            self.F=response.ipgauss_sampling(self.nuobs,self.nus,Frot,beta,self.RV)
            te=time.time()
            print("IP",te-ts,"s")
        else:
            ts=time.time()
            c=299792.458
            dv=c*(np.log(self.nus[1])-np.log(self.nus[0]))
            Nv=int(self.vsini/dv)+1
            vlim=Nv*dv
            Nkernel=2*Nv+1
            varr_kernel=jnp.linspace(-vlim,vlim,Nkernel)
            Frot=response.rigidrot2(self.nus,F0,varr_kernel,self.vsini,u1=self.u1,u2=self.u2)
            te=time.time()
            print("rotation(2)",te-ts,"s")
            ts=time.time()
            maxp=5.0 #5sigma
            Nv=int(maxp*beta/dv)+1
            vlim=Nv*dv
            Nkernel=2*Nv+1
            varr_kernel=jnp.linspace(-vlim,vlim,Nkernel)
            Fgrot=response.ipgauss2(self.nus,Frot,varr_kernel,beta)                      
            self.F=response.sampling(self.nuobs,self.nus,Fgrot,self.RV)
            te=time.time()
            print("IP(2)",te-ts,"s")
            
        return self.F
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exojax.spec.rtransfer import nugrid

    #nus=np.linspace(6101.0,6115.0,3000,dtype=np.float64)
    #nus=np.linspace(6101.0,6115.0,3000,dtype=np.float64)
    #XS
    #autoxs=AutoXS(nus,"HITRAN","CO")
    #xsv=autoxs.xsection(1000.0,1.0) #1000K, 1bar
    #Tarr=np.array([1000.0,1500.0])
    #Parr=np.array([1.0,0.1])
    #xsm=autoxs.xsmatrix(Tarr,Parr) 

    #RT
    nus,wav,res=nugrid(1900.0,2300.0,40000,"cm-1")
    #nus=np.linspace(1900.0,2300.0,40000,dtype=np.float64)
    #nus=np.linspace(1900.0,1910.0,1000,dtype=np.float64)
    Parr=np.logspace(-8,2,100)
    Tarr = 500.*(Parr/Parr[-1])**0.02    
    autort=AutoRT(nus,1.e5,2.33,Tarr,Parr) #g=1.e5 cm/s2, mmw=2.3
    autort.addcia("H2-H2",0.74,0.74) #CIA mmr(H)=0.74
    autort.addcia("H2-He",0.74,0.25) #CIA mmr(He)=0.25
    autort.addmol("ExoMol","CO",0.01) #mmr=0.01
    
    F0=autort.rtrun()
    fig=plt.figure(figsize=(10,3))
    plt.plot(nus,autort.F0,alpha=0.5)
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("flux (erg/cm2/s/cm-1)")
    plt.savefig("spec0.png", bbox_inches="tight", pad_inches=0.0)

    nusobs=np.linspace(1900.0,2300.0,10000,dtype=np.float64) #observation bin
    F=autort.spectrum(nusobs,100000.0,20.0,0.0) #R=100000,vsini=10. km/s, RV=0. km/s

    fig=plt.figure(figsize=(10,3))
    plt.plot(nus,autort.F0,alpha=0.5,label="raw")
    plt.plot(nusobs,F,lw=2,label="obs")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("flux (erg/cm2/s/cm-1)")
    plt.legend()
    plt.savefig("spec.png", bbox_inches="tight", pad_inches=0.0)

