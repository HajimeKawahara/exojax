"""Molecular database (MDB) class

   * MdbExomol is the MDB for ExoMol
   * MdbHit is the MDB for HITRAN or HITEMP  
   
"""
import numpy as np
import jax.numpy as jnp
import pathlib
from exojax.spec import hapi, exomolapi, exomol
from exojax.spec.hitran import gamma_natural as gn
import pandas as pd

__all__ = ['MdbExomol','MdbHit']

class MdbExomol(object):
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=1.0,crit=-np.inf):
        """Molecular database for Exomol form

        Args: 
           path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction

        Note:
           The trans/states files can be very large. For the first time to read it, we convert it to the feather-format. After the second-time, we use the feather format instead.

        """
        explanation="Note: Couldn't find the feather format. We convert data to the feather format. After the second time, it will become much faster."
        
        self.path = pathlib.Path(path)
        t0=self.path.parents[0].stem        
        molec=t0+"__"+str(self.path.stem)
        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]
            
        self.states_file = self.path/pathlib.Path(molec+".states.bz2")
        self.pf_file = self.path/pathlib.Path(molec+".pf")
        self.def_file = self.path/pathlib.Path(molec+".def")
        if not self.def_file.exists():
                self.download(molec,extension=[".def",".pf",".states.bz2"])

        #load def 
        self.n_Texp, self.alpha_ref, self.molmass, numinf, numtag=exomolapi.read_def(self.def_file)
        #  default n_Texp value if not given
        if self.n_Texp is None:
            self.n_Texp=0.5
        #  default alpha_ref value if not given
        if self.alpha_ref is None:
            self.alpha_ref=0.07

        #load states
        if self.states_file.with_suffix(".feather").exists():
            states=pd.read_feather(self.states_file.with_suffix(".feather"))
        else:
            print(explanation)
            states=exomolapi.read_states(self.states_file)
            states.to_feather(self.states_file.with_suffix(".feather"))
        #load pf
        pf=exomolapi.read_pf(self.pf_file)
        self.gQT=jnp.array(pf["QT"].to_numpy()) #grid QT
        self.T_gQT=jnp.array(pf["T"].to_numpy()) #T forgrid QT
                
        #trans file(s)
        print("Reading transition file")
        if numinf is None:
            self.trans_file = self.path/pathlib.Path(molec+".trans.bz2")
            if not self.trans_file.exists():
                self.download(molec,[".trans.bz2"])

            if self.trans_file.with_suffix(".feather").exists():
                trans=pd.read_feather(self.trans_file.with_suffix(".feather"))
            else:
                print(explanation)
                trans=exomolapi.read_trans(self.trans_file)
                trans.to_feather(self.trans_file.with_suffix(".feather"))
            #compute gup and elower
            self._A, self.nu_lines, self._elower, self._gpp=exomolapi.pickup_gE(states,trans)        
        else:
            imin=np.searchsorted(numinf,nurange[0],side="right")-1 #left side
            imax=np.searchsorted(numinf,nurange[1],side="right")-1 #left side
            self.trans_file=[]
            for k,i in enumerate(range(imin,imax+1)):
                trans_file = self.path/pathlib.Path(molec+"__"+numtag[i]+".trans.bz2")
                if not trans_file.exists():
                    self.download(molec,extension=[".trans.bz2"],numtag=numtag[i])
                if trans_file.with_suffix(".feather").exists():
                    trans=pd.read_feather(trans_file.with_suffix(".feather"))
                else:
                    print(explanation)
                    trans=exomolapi.read_trans(trans_file)
                    trans.to_feather(trans_file.with_suffix(".feather"))
                self.trans_file.append(trans_file)
                #compute gup and elower                
                if k==0:
                    self._A, self.nu_lines, self._elower, self._gpp=exomolapi.pickup_gE(states,trans)
                else:
                    Ax, nulx, elowerx, gppx=exomolapi.pickup_gE(states,trans)
                    self._A=np.hstack([self._A,Ax])
                    self.nu_lines=np.hstack([self.nu_lines,nulx])
                    self._elower=np.hstack([self._elower,elowerx])
                    self._gpp=np.hstack([self._gpp,gppx])

        self.Tref=296.0        
        self.QTref=np.array(self.QT_interp(self.Tref))
        
        ##input should be ndarray not jnp array
        self.Sij0=exomol.Sij0(self._A,self._gpp,self.nu_lines,self._elower,self.QTref)
        
        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)
        
        self.masking(mask)
        
    def masking(self,mask):
        """applying mask and (re)generate jnp.arrays
        
        Args:
           mask: mask to be applied

        """
        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]
        self._A=self._A[mask]
        self._elower=self._elower[mask]
        self._gpp=self._gpp[mask]
        
        #jnp arrays
        self.dev_nu_lines=jnp.array(self.nu_lines)
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.A=jnp.array(self._A)
        self.gamma_natural=gn(self.A)
        self.elower=jnp.array(self._elower)
        self.gpp=jnp.array(self._gpp)

        
    def QT_interp(self,T):
        """interpolated partition function

        Args:
           T: temperature

        Returns:
           Q(T) interpolated in jnp.array

        """
        return jnp.interp(T,self.T_gQT,self.gQT)
    
    def qr_interp(self,T):
        """interpolated partition function ratio

        Args:
           T: temperature

        Returns:
           qr(T)=Q(T)/Q(Tref) interpolated in jnp.array

        """
        return self.QT_interp(T)/self.QT_interp(self.Tref)
    
        
    def download(self,molec,extension,numtag=None):
        """Downloading Exomol files

        Args: 
           molec: like "12C-16O__Li2015"
           extension: extension list e.g. [".pf",".def",".trans.bz2",".states.bz2"]
           numtag: number tag of transition file if exists. e.g. "11100-11200"

        Note:
           The download URL is written in exojax.utils.url.

        """
        import urllib.request
        from exojax.utils.molname import e2s
        import os
        from exojax.utils.url import url_ExoMol

        tag=molec.split("__")
        molname_simple=e2s(tag[0])        
        url = url_ExoMol()+molname_simple+"/"+tag[0]+"/"+tag[1]+"/"

        
        for ext in extension:
            if ext==".trans.bz2" and numtag is not None:
                ext="__"+numtag+ext
            pfname=molec+ext
            pfpath=url+pfname
            os.makedirs(str(self.path), exist_ok=True)
            print("Downloading "+pfpath)
            try:
                urllib.request.urlretrieve(pfpath,str(self.path/pfname))
            except:
                print("Error: Couldn't download "+ext+" file and save.")




class MdbHit(object):
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=250.0,crit=-np.inf):
        """Molecular database for HITRAN/HITEMP form

        Args: 
           path: path for HITRAN/HITEMP par file
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction

        """        
        #downloading
        self.path = pathlib.Path(path)
        molec=str(self.path.stem)
        if not self.path.exists():
            self.download()

        #bunzip2 if suffix is .bz2
        if self.path.suffix==".bz2":
            import bz2,shutil
            print("bunziping")
            with bz2.BZ2File(str(self.path)) as fr:
                with open(str(self.path.with_suffix('')),"wb") as fw:
                    shutil.copyfileobj(fr,fw)
            self.path=self.path.with_suffix('')
            
        molec=str(self.path.stem)
            
        hapi.db_begin(str(self.path.parent))            
        self.Tref=296.0        
        self.molecid = search_molecid(molec)
        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]
        self.nu_lines = hapi.getColumn(molec, 'nu')
        self.Sij0 = hapi.getColumn(molec, 'sw')

        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)

        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]        
        self.delta_air = hapi.getColumn(molec, 'delta_air')[mask]

        #jnp array
        A=hapi.getColumn(molec, 'a')[mask]
        self.A = jnp.array(A)
        self.gamma_natural=gn(A)

        self.n_air = jnp.array(hapi.getColumn(molec, 'n_air')[mask])
        self.gamma_air = jnp.array(hapi.getColumn(molec, 'gamma_air')[mask])
        self.gamma_self = jnp.array(hapi.getColumn(molec, 'gamma_self')[mask])        
        self.elower = jnp.array(hapi.getColumn(molec, 'elower')[mask])
        self.gpp = jnp.array(hapi.getColumn(molec, 'gpp')[mask]) 
        self.logsij0=jnp.array(np.log(self.Sij0)) 
        self.dev_nu_lines=jnp.array(self.nu_lines)

        #int
        self.isoid = hapi.getColumn(molec,'local_iso_id')[mask]
        self.uniqiso=np.unique(self.isoid)

        

        
    def download(self):
        """Downloading HITRAN/HITEMP par file

        Note:
           The download URL is written in exojax.utils.url.

        """
        import urllib.request
        from exojax.utils.url import url_HITRAN12
        from exojax.utils.url import url_HITEMP

        try:
            url = url_HITRAN12()+self.path.name
            urllib.request.urlretrieve(url,str(self.path))
        except:
            print(url)
            print("HITRAN download failed")
        try:
            url = url_HITEMP()+self.path.name
            print(url)
            urllib.request.urlretrieve(url,str(self.path))
        except:
            print("HITEMP download failed")

            
    def Qr(self,Tarr):
        """Partition Function ratio using HAPI partition sum

        Args:
           Tarr: temperature array (K)
        
        Returns:
           Qr = partition function ratio array [N_Tarr x N_iso]

        Note: 
           N_Tarr = len(Tarr), N_iso = len(self.uniqiso)

        """
        allT=list(np.concatenate([[self.Tref],Tarr]))
        Qrx=[]
        for iso in self.uniqiso:
            Qrx.append(hapi.partitionSum(self.molecid,iso, allT))
        Qrx=np.array(Qrx)
        qr=Qrx[:,1:].T/Qrx[:,0] #Q(T)/Q(Tref)
        return qr

    def Qr_line(self,T):
        """Partition Function ratio using HAPI partition sum

        Args:
           T: temperature (K)

        Returns:
           Qr_line, partition function ratio array for lines [Nlines]

        Note: 
           Nlines=len(self.nu_lines)

        """
        qr_line=np.ones_like(self.isoid,dtype=np.float64)
        qrx=self.Qr([T])
        for idx,iso in enumerate(self.uniqiso):
            mask=self.isoid==iso
            qr_line[mask]=qrx[0,idx]
        return qr_line

    def Qr_layer(self,Tarr):
        """Partition Function ratio using HAPI partition sum

        Args:
           Tarr: temperature array (K)

        Returns:
           Qr_layer, partition function ratio array for lines [N_Tarr x Nlines]

        Note: 
           Nlines=len(self.nu_lines)
           N_Tarr=len(Tarr)

        """
        NP=len(Tarr)
        qt=np.zeros((NP,len(self.isoid)))
        qr=self.Qr(Tarr)
        for idx,iso in enumerate(self.uniqiso):
            mask=self.isoid==iso
            for ilayer in range(NP):
                qt[ilayer,mask]=qr[ilayer,idx]
        return qt
    
def search_molecid(molec):
    """molec id from molec (source table name) of HITRAN/HITEMP

    Args:
       molec: source table name

    Return:
       int: molecid (HITRAN molecular id)

    """
    try:
        hitf=molec.split("_")
        molecid=int(hitf[0])
        return molecid

    except:
        print("Warning: Define molecid by yourself.")
        return None

if __name__ == "__main__":
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/CO/12C-16O/Li2015/")
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/NO/14N-16O/NOname/14N-16O__NOname")
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/NO/14N-16O/NOname/14N-16O__NOname")
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/CH4/12C-1H4/YT34to10/",nurange=[6050.0,6150.0])
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/NH3/14N-1H3/CoYuTe/",nurange=[6050.0,6150.0])
    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/H2S/1H2-32S/AYT2/",nurange=[6050.0,6150.0])
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/FeH/56Fe-1H/MoLLIST/",nurange=[6050.0,6150.0])
