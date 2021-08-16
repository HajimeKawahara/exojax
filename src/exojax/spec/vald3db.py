"""Atomic database (ADB) class
    created by mimicking "moldb.py"
"""
import numpy as np
import jax.numpy as jnp
import pathlib
from exojax.spec import vald3api, vald3
import pandas as pd

__all__ = ['AdbVald',]

class AdbVald(object):
    """ molecular database of ExoMol
    
    AdbVald is a class for VALD3.
    
    Attributes:
        ...
        elower (jnp array): the lower state energy (cm-1)
        ...
        #\\\\
    
    """
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=1.0,crit=-np.inf, bkgdatm="H2", broadf=True, pathdat=pathlib.Path("~/ghR/exojax/src/exojax/metaldata")): #tako210721
    
        """Atomic database for VALD3 "Long format"

        Args:
           #\\\\
           
        Note:
           #\\\\

        """
        #explanation="
        
        self.path = pathlib.Path(path)
        #t0=self.path.parents[0].stem
        #molec=t0+"__"+str(self.path.stem)
        self.bkgdatm=bkgdatm
        print("Background atmosphere: ",self.bkgdatm)
        #molecbroad=t0+"__"+self.bkgdatm
        
        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]
        self.broadf=broadf
        #VALD3 output
        self.vald3_file = self.path #tako210721
        #Where exomol files are
        #self.states_file = self.path/pathlib.Path(molec+".states.bz2")
        #self.pf_file = self.path/pathlib.Path(molec+".pf")
        #self.def_file = self.path/pathlib.Path(molec+".def")
        #self.broad_file = self.path/pathlib.Path(molecbroad+".broad")




        #load partition function (for 284 atomic species) from Barklem et al. (2016)
        pff = pathdat/"J_A+A_588_A96/table8.dat"
        pfTf = pathdat/"J_A+A_588_A96/table8_T.dat"
        #"/home/tako/work/.database/Barklem_2016/J_A+A_588_A96/...

        pfTdat = pd.read_csv(pfTf, sep="\s+")
        self.T_gQT = jnp.array(pfTdat.columns[1:].to_numpy(dtype=float)) #T for grid QT    #self.
        self.pfdat = pd.read_csv(pff, sep="\s+", comment="#", names=pfTdat.columns)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(dtype=float)) #grid Q vs T vs Species

        


        #load vald file ("Extract Stellar" request)
        print("Reading VALD file")
        valdd=vald3api.read_ExAll(self.vald3_file)
        #valdd=vald3api.read_ExStellar(self.vald3_file)
        #better to use .feather format? #\\\\
        #valdd.to_feather(self.vald3_file.with_suffix(".feather"))
        
        #compute additional transition parameters
        self._A, self.nu_lines, self._elower, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._vdWdamp, self._gamRad = vald3api.pickup_param(valdd)
        
        
        
        self.Tref=296.0 #\\\\
        self.QTref_284 = np.array(self.QT_interp_284(self.Tref))
        self.QTmask = self.make_QTmask(valdd) #identify species for each line



        ##Line strength: input shoud be ndarray not jnp array
        self.Sij0 = vald3.Sij0(self._A, self._gupper, self.nu_lines, self._elower, self.QTref_284, self.QTmask)



        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)
        
        self.masking(mask)
        


        #Compile atomic-specific data for each line
        ipccf = pathdat/"ipcc_Asplund2009_pre.dat"
        ipccc = ('ielem', 'ionizationE1', 'dam1', 'dam2', 'solarA', 'mass', 'ionizationE2')
        self.ipccd = pd.read_csv(ipccf, sep="\s+", skiprows=1, usecols=[1,2,3,4,5,6,7], names=ipccc)
        
        ionE = jnp.array(list(map(lambda x: self.ipccd[self.ipccd['ielem']==x].iat[0, 1], self.ielem)))
        ionE2 = jnp.array(list(map(lambda x: self.ipccd[self.ipccd['ielem']==x].iat[0, 6], self.ielem)))
        self.ionE = ionE * np.where(self.iion==1, 1, 0) + ionE2 * np.where(self.iion==2, 1, 0)
        self.solarA = jnp.array(list(map(lambda x: self.ipccd[self.ipccd['ielem']==x].iat[0, 4], self.ielem)))
        self.atomicmass = jnp.array(list(map(lambda x: self.ipccd[self.ipccd['ielem']==x].iat[0, 5], self.ielem)))
        
            
    #End of the CONSTRUCTOR definition ↑



    #Defining METHODS ↓
    
    
    def masking(self,mask):
        """applying mask and (re)generate jnp.arrays
        
        Args:
           mask: mask to be applied. self.mask is updated.

        Note:
           We have nd arrays and jnp arrays. We apply the mask to nd arrays and generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]
        self._A=self._A[mask]
        self._elower=self._elower[mask]
        self._gupper=self._gupper[mask]
        self._jlower=self._jlower[mask]
        self._jupper=self._jupper[mask]
        self.QTmask=self.QTmask[mask]
        self._ielem=self._ielem[mask]
        self._iion=self._iion[mask]
        self._vdWdamp=self._vdWdamp[mask]
        self._gamRad=self._gamRad[mask]
        
        #jnp arrays
        self.dev_nu_lines=jnp.array(self.nu_lines)
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.A=jnp.array(self._A)
        self.gamma_natural=vald3.gamma_natural(self.A) #gamma_natural [cm-1] #natural broadeningの原理はmoleculeと同様のはず(tako210802)
        self.elower=jnp.array(self._elower)
        self.gupper=jnp.array(self._gupper)
        self.jlower=jnp.array(self._jlower,dtype=int)
        self.jupper=jnp.array(self._jupper,dtype=int)
        
        self.QTmask=jnp.array(self.QTmask,dtype=int)
        self.ielem=jnp.array(self._ielem,dtype=int)
        self.iion=jnp.array(self._iion,dtype=int)
        self.vdWdamp=jnp.array(self._vdWdamp)
        self.gamRad=jnp.array(self._gamRad)

    def Atomic_gQT(self, atomspecies):
        """Select grid of partition function especially for the species of interest
        
        Args:
            atomspecies: species e.g., "Fe 1"
            
        Returns:
            gQT: grid Q(T) for the species
        
        """
        if len(atomspecies.split(' '))==2:
            atomspecies_Roman = atomspecies.split(' ')[0] + '_' + 'I'*int(atomspecies.split(' ')[-1])
        gQT = self.gQT_284species[ np.where(self.pfdat['T[K]']==atomspecies_Roman) ][0]
        return gQT
    
    def QT_interp(self, atomspecies, T):
        """interpolated partition function

        Args:
           T: temperature
           atomspecies: species e.g., "Fe 1"

        Returns:
           Q(T): interpolated in jnp.array for the Atomic Species

        """
        gQT = self.Atomic_gQT(atomspecies)
        return jnp.interp(T, self.T_gQT, gQT)

    def qr_interp(self, atomspecies, T):
        """interpolated partition function ratio

        Args:
           T: temperature
           atomspecies: species e.g., "Fe 1"

        Returns:
           qr(T)=Q(T)/Q(Tref): interpolated in jnp.array

        """
        return self.QT_interp(atomspecies,T)/self.QT_interp(atomspecies,self.Tref)

    def QT_interp_284(self, T):
        """interpolated partition function of all 284 species

        Args:
           T: temperature

        Returns:
           Q(T)*284: interpolated in jnp.array for all 284 Atomic Species

        """
        #self.T_gQT.shape -> (42,)
        #self.gQT_284species.shape -> (284, 42)
        list_gQT_eachspecies = self.gQT_284species.tolist()
        listofDA_gQT_eachspecies = list(map(lambda x: jnp.array(x), list_gQT_eachspecies))
        listofQT = list(map(lambda x: jnp.interp(T, self.T_gQT, x), listofDA_gQT_eachspecies))
        QT_284 = jnp.array(listofQT)
        return QT_284

    def make_QTmask(self, ExAll): #identify species for each line
        """Make array for assigning identifier of Q(Tref) to each line
        
        Input: ExAll: Dataframe outputted from read_ExAll
        """
        QTmask_sp = np.zeros(len(ExAll))
        for i, sp in enumerate(ExAll['species']):
            sp_Roman = sp.strip("'").split(' ')[0] + '_' + 'I'*int(sp.strip("'").split(' ')[-1])
            QTmask_sp[i] = np.where(self.pfdat['T[K]']==sp_Roman)[0][0]

        QTmask_sp = QTmask_sp.astype('int')
        return(QTmask_sp)
