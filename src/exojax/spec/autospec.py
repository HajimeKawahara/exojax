from exojax.spec import defmol
from exojax.spec import moldb
from exojax.spec import xsection
from exojax.spec import SijT, doppler_sigma,  gamma_natural
from exojax.spec.exomol import gamma_exomol
import pathlib
__all__ = ['AutoXS']


class AutoXS(object):
    """exojax auto cross section generator
    
    """
    def __init__(self,nus,database,molecules,databasedir=".database",memory_size=30):
        self.molecules=molecules
        self.database=database
        self.nus=nus
        self.databasedir=databasedir
        self.memory_size=memory_size
        
        self.identifier=defmol.search_molfile(database,molecules)
        self.init_database()
        
    def init_database(self):
        if self.database=="HITRAN" or self.database=="HITEMP":
            try:
                self.mdb=moldb.MdbHit(self.molfile,nurange=[self.nus[0],self.nus[-1]])
            except:
                print("Not supported yet. Define by yourself.")
        elif self.database=="ExoMol":
            try:
                molpath=pathlib.Path(self.databasedir)/pathlib.Path(self.identifier)
                molpath=str(molpath)
                self.mdb=moldb.MdbExomol(molpath,nurange=[self.nus[0],self.nus[-1]])
            except:
                print("Not supported yet. Define by yourself.")

        else:
            print("Select database from HITRAN, HITEMP, ExoMol.")

    def xsec(self,T,P):
        """cross section
        Args: 
           T: temperature (K)
           P: pressure (bar)
        Returns:
           cross section (cm2)
        """
        mdb=self.mdb
        qt=mdb.qr_interp(T)
        Sij=SijT(T,mdb.logsij0,mdb.nu_lines,mdb.elower,qt)
        gammaL = gamma_exomol(P,T,mdb.n_Texp,mdb.alpha_ref)\
                 + gamma_natural(mdb.A) 
        sigmaD=doppler_sigma(mdb.nu_lines,T,mdb.molmass)
        nu0=mdb.nu_lines
        xsv=xsection(nus,nu0,sigmaD,gammaL,Sij,memory_size=self.memory_size)
        return xsv
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    nus=np.linspace(1900.0,2300.0,4000,dtype=np.float64)
    autoxs=AutoXS(nus,"ExoMol","CO")
    xsv=autoxs.xsec(1000.0,1.0) #1000K, 1bar
    plt.plot(nus,xsv)
    plt.show()
