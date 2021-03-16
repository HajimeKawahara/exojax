from exojax.spec.make_numatrix import make_numatrix0
from exojax.spec.lpf import xsvector
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
import hapi
import numpy as np

__all__ = ['AutoXS']



class AutoXS(object):
    """exojax auto cross section generator
    
    """
    def __init__(self,nus,database,molecules,databasedir=".database"):
        self.molecules=molecules
        self.database=database
        self.nus=nus
        self.databasedir=databasedir
        
        defmol.search_molfile(database,molecules)
        
    def initialization_database(self):
        if self.database=="HITRAN" or self.database=="HITEMP":
            try:
                self.mdb=moldb.MdbHit(self.molfile,self.nus)
            except:
                print("Define databasedir.")
        elif self.database=="ExoMol":
            
        else:
            print("Select database from HITRAN, HITEMP, ExoMol.")
                    
    def load_hitran(self):
        self.A_all = hapi.getColumn(self.molec, 'a')
        self.n_air_all = hapi.getColumn(self.molec, 'n_air')
        self.isoid_all = hapi.getColumn(self.molec,'local_iso_id')
        self.gamma_air_all = hapi.getColumn(self.molec, 'gamma_air')
        self.gamma_self_all = hapi.getColumn(self.molec, 'gamma_self')
        self.nu_lines_all = hapi.getColumn(self.molec, 'nu')
        self.delta_air_all = hapi.getColumn(self.molec, 'delta_air')
        self.Sij0_all = hapi.getColumn(self.molec, 'sw')
        self.elower_all = hapi.getColumn(self.molec, 'elower')
        self.gpp_all = hapi.getColumn(self.molec, 'gpp')

    def partition_function_hitran(self,Tfix):
        Tref=296.0 # HITRAN reference temeprature is 296 K
        Qr=[]
        for iso in self.uniqiso:
            Qr.append(hapi.partitionSum(self.molecid,iso, [Tref,Tfix]))
            Qr=np.array(Qr)
        qr=Qr[:,0]/Qr[:,1] #Q(Tref)/Q(T)
        qt=np.zeros(len(self.isoid))
        for idx,iso in enumerate(self.uniqiso):
            mask=isoid==iso
        qt[mask]=qr[idx]        
        return qt

    def compute_line_strength(self,Tfix):
        if self.database == "HITRAN" or self.database == "HITEMP": 
            logsij0=np.log(self.Sij0)
            self.Sij=SijT(Tfix,logsij0,self.nu_lines,self.gpp,self.elower,self.qt)
            
    def compute_gamma(self,Tfix,Pfix,Ppart):
        if self.database == "HITRAN" or self.database == "HITEMP": 
            self.gammaL = gamma_hitran(Pfix,Tfix,Ppart, self.n_air, self.gamma_air, self.gamma_self) \
                     + gamma_natural(self.A) 

def hitran_molec():
    molec_dict={"CO":'05_hit12'}
    molecid_dict={"CO":5}
    
def default_molec_weight():
    mweight_dict={"CO":28.010446441149536}
