from exojax.spec import defmol
from exojax.spec import moldb
from exojax.spec.opacity import xsection
from exojax.spec.hitran import SijT, doppler_sigma,  gamma_natural, gamma_hitran
from exojax.spec.exomol import gamma_exomol
from exojax.spec import molinfo

import pathlib
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
        
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    nus=np.linspace(1900.0,2300.0,40000,dtype=np.float64)
    autoxs=AutoXS(nus,"HITRAN","CO")
    #autoxs=AutoXS(nus,"ExoMol","CO")
    #    autoxs=AutoXS(nus,"HITEMP","CO")
    xsv=autoxs.xsection(1000.0,1.0) #1000K, 1bar
    plt.plot(nus,xsv)
    plt.show()
