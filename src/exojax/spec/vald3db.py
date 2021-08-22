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
    """ atomic database of VALD3
    
    AdbVald is a class for VALD3.
    
    Attributes:
        nu_lines (nd array):      line center (cm-1) (#NOT frequency in (s-1))
        dev_nu_lines (jnp array): line center (cm-1) in device
        Sij0 (nd array): line strength at T=Tref (cm)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient in (s-1)
        gamma_natural (jnp array): gamma factor of the natural broadening
        elower (jnp array): the lower state energy (cm-1)
        gupper: (jnp array): upper statistical weight
        jlower (jnp array): lower J (rotational quantum number, total angular momentum)
        jupper (jnp array): upper J
        QTmask (jnp array): identifier of species for Q(T)
        ielem (jnp array):  atomic number (e.g., Fe=26)
        iion (jnp array):  ionized level (e.g., neutral=1, singly)
        vdWdamp (jnp array):  van der Waals damping parameters
        gamRad (jnp array): gamma(HWHM of Lorentzian) of radiation damping
            
    """
    def __init__(self, path, nurange=[-np.inf,np.inf], margin=1.0, crit=-np.inf): #tako210721
    
        """Atomic database for VALD3 "Long format"

        Args:
           path: path for linelists downloaded from VALD3 with a query of "Long format" in the format of "Extract All" and "Extract Element" (NOT "Extract Stellar")
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           
        Note:
           (written with reference to moldb.py, but without using feather format)

        """
        #explanation="
        
        #load args
        self.vald3_file = pathlib.Path(path) #VALD3 output
        #self.path = pathlib.Path(path) #molec=t0+"__"+str(self.path.stem) #t0=self.path.parents[0].stem
        self.nurange = [np.min(nurange),np.max(nurange)]
        self.margin = margin
        self.crit = crit
        #self.bkgdatm=bkgdatm
        #self.broadf=broadf
        
        

        #load vald file ("Extract Stellar" request)
        print("Reading VALD file")
        valdd=vald3api.read_ExAll(self.vald3_file)
        #better to use .feather format? #\\\\
        #valdd.to_feather(self.vald3_file.with_suffix(".feather"))
        
        #compute additional transition parameters
        self._A, self.nu_lines, self._elower, self._gupper, self._jlower, self._jupper, self._ielem, self._iion, self._vdWdamp, self._gamRad = vald3api.pickup_param(valdd)
        
        
        
        #load the partition functions (for 284 atomic species)
        pfTdat, self.pfdat = vald3api.load_pf_Barklem2016() #Barklem & Collet (2016)
        self.T_gQT = jnp.array(pfTdat.columns[1:], dtype=float)
        self.gQT_284species = jnp.array(self.pfdat.iloc[:, 1:].to_numpy(dtype=float)) #grid Q vs T vs Species
        self.Tref=296.0 #\\\\
        self.QTref_284 = np.array(self.QT_interp_284(self.Tref))
        self._QTmask = self.make_QTmask(valdd) #identify species for each line



        ##Line strength: input shoud be ndarray not jnp array
        self.Sij0 = vald3.Sij0(self._A, self._gupper, self.nu_lines, self._elower, self.QTref_284, self._QTmask)



        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)
        
        self.masking(mask)
        


        #Compile atomic-specific data for each absorption line of interest
        self.ipccd = vald3api.load_atomicdata()
        #print(self.ipccd)#test
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
        self._QTmask=self._QTmask[mask]
        self._ielem=self._ielem[mask]
        self._iion=self._iion[mask]
        self._vdWdamp=self._vdWdamp[mask]
        self._gamRad=self._gamRad[mask]
        
        #jnp arrays
        self.dev_nu_lines=jnp.array(self.nu_lines)
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.A=jnp.array(self._A)
        self.gamma_natural=vald3.gamma_natural(self.A) #gamma_natural [cm-1]
        self.elower=jnp.array(self._elower)
        self.gupper=jnp.array(self._gupper)
        self.jlower=jnp.array(self._jlower,dtype=int)
        self.jupper=jnp.array(self._jupper,dtype=int)
        
        self.QTmask=jnp.array(self._QTmask,dtype=int)
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
           atomspecies: species e.g., "Fe 1"
           T: temperature

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
