from exojax.spec import defmol
from exojax.spec import moldb
from exojax.spec.opacity import xsection
from exojax.spec.hitran import SijT, doppler_sigma,  gamma_natural, gamma_hitran
from exojax.spec import planck
from exojax.spec.exomol import gamma_exomol
from exojax.spec import molinfo
from exojax.spec.rtransfer import rtrun, pressure_layer, dtaux
from exojax.spec import make_numatrix0
from exojax.spec.lpf import xsmatrix

from jax import jit, vmap
import pathlib
import tqdm
__all__ = ['AutoXS']


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
            qt=mdb.qr_interp(T)
            gammaL = gamma_exomol(P,T,mdb.n_Texp,mdb.alpha_ref)\
                     + gamma_natural(mdb.A)
            molmass=mdb.molmass
        elif self.database == "HITRAN" or self.database == "HITEMP":
            qt=mdb.Qr_line(T)
            gammaL = gamma_hitran(P,T, P, mdb.n_air, \
                      mdb.gamma_air, mdb.gamma_self) \
                      + gamma_natural(mdb.A)
            molmass=molinfo.molmass(self.molecules)
            
        Sij=SijT(T,mdb.logsij0,mdb.nu_lines,mdb.elower,qt)
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
            qt=mdb.qr_interp(T)
            gammaLM = jit(vmap(gamma_exomol,(0,0,None,None)))\
                              (P,T,mdb.n_Texp,mdb.alpha_ref)\
                     + gamma_natural(mdb.A)
            self.molmass=mdb.molmass
            SijM=jit(vmap(SijT,(0,None,None,None,None)))\
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
        Ni=int(len(nus)/d)        
        d2=100
        Nlayer=np.shape(SijM)[0]
        Nline=np.shape(SijM)[1]
        Nj=int(Nline/d2)
        xsm=[]
        for i in tqdm.tqdm(range(0,Ni+1)):
            s=int(i*d);e=int((i+1)*d);e=min(e,len(nus))
            xsmtmp=np.zeros((Nlayer,e-s))
            #line 
            for j in range(0,Nj+1):
                s2=int(j*d2);e2=int((j+1)*d2);e2=min(e2,Nline)
                numatrix=make_numatrix0(nus[s:e],nu0[s2:e2])
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
    def __init__(self,nus,gravity,Tarr,Parr,dParr=None):
        """
        Args:
           nus: wavenumber bin (cm-1)
           gravity: gravity (cm/s2)
           Tarr: temperature layer (K)
           Parr: pressure layer (bar)
           dParr: delta pressure (bar) optional
        """
        self.nus=nus
        self.gravity=gravity
        self.nlayer=len(Tarr)        
        self.Tarr=Tarr
        self.Parr=Parr
        if dParr is None:
            from exojax.utils.chopstacks import buildwall 
            wParr=buildwall(Parr)
            self.dParr=wParr[1:]-wParr[0:-1]
        else:
            self.dParr=dParr
            
        self.sourcef=planck.piBarr(self.Tarr,self.nus)
        self.dtauM=np.zeros((self.nlayer,len(nus)))

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
        dtauMx=dtaux(dParr,xsm,mmr,axs.molmass,self.gravity)
        self.dtauM=self.dtauM+dtauMx

    def rtrun(self):
        Fx0=rtrun(self.dtauM,self.sourcef)
        return Fx0

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    #nus=np.linspace(6101.0,6115.0,3000,dtype=np.float64)
#    nus=np.linspace(6101.0,6115.0,3000,dtype=np.float64)
    #XS
    #autoxs=AutoXS(nus,"HITRAN","CO")
    #xsv=autoxs.xsection(1000.0,1.0) #1000K, 1bar
    #Tarr=np.array([1000.0,1500.0])
    #Parr=np.array([1.0,0.1])
    #xsm=autoxs.xsmatrix(Tarr,Parr) 

    #RT
    nus=np.linspace(1900.0,2300.0,40000,dtype=np.float64)
    Parr, dParr, k=pressure_layer(NP=100)
    Tarr = 1500.*(Parr/Parr[-1])**0.02    
    autort=AutoRT(nus,1.e5,Tarr,Parr) #g=1.e5 cm/s2
    autort.addmol("HITRAN","CO",0.01) #mmr=0.01
    F=autort.rtrun()

    plt.plot(nus,F)
    plt.show()

