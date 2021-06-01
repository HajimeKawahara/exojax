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
    """ molecular database of ExoMol

    MdbExomol is a class for ExoMol.

    Attributes:
        nu_lines (nd array): line center (cm-1)
        Sij0 (nd array): line strength at T=Tref (cm)
        dev_nu_lines (jnp array): line center in device (cm-1)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient
        gamma_natural (jnp array): gamma factor of the natural broadening
        elower (jnp array): the lower state energy (cm-1)
        gpp (jnp array): statistical weight
        jlower (jnp array): J_lower
        jupper (jnp array): J_upper
        n_Tref (jnp array): temperature exponent
        alpha_ref (jnp array): alpha_ref (gamma0)
        n_Tref_def: default temperature exponent in .def file, used for jlower not given in .broad
        alpha_ref_def: default alpha_ref (gamma0) in .def file, used for jlower not given in .broad

    """
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=1.0,crit=-np.inf, bkgdatm="H2"):
        """Molecular database for Exomol form

        Args: 
           path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)
           crit: line strength lower limit for extraction
           bkgdatm: background atmosphere for broadening. e.g. H2, He, 

        Note:
           The trans/states files can be very large. For the first time to read it, we convert it to the feather-format. After the second-time, we use the feather format instead.

        """
        explanation="Note: Couldn't find the feather format. We convert data to the feather format. After the second time, it will become much faster."
        
        self.path = pathlib.Path(path)
        t0=self.path.parents[0].stem        
        molec=t0+"__"+str(self.path.stem)
        self.bkgdatm=bkgdatm
        print("Background atmosphere: ",self.bkgdatm)
        molecbroad=t0+"__"+self.bkgdatm

        self.crit = crit
        self.margin = margin
        self.nurange=[np.min(nurange),np.max(nurange)]

        #Where exomol files are
        self.states_file = self.path/pathlib.Path(molec+".states.bz2")
        self.pf_file = self.path/pathlib.Path(molec+".pf")
        self.def_file = self.path/pathlib.Path(molec+".def")
        self.broad_file = self.path/pathlib.Path(molecbroad+".broad")

        if not self.def_file.exists():
                self.download(molec,extension=[".def"])
        if not self.pf_file.exists():
                self.download(molec,extension=[".pf"])
        if not self.states_file.exists():
                self.download(molec,extension=[".states.bz2"])
        if not self.broad_file.exists():
                self.download(molec,extension=[".broad"])
        
        #load def 
        self.n_Texp_def, self.alpha_ref_def, self.molmass, numinf, numtag=exomolapi.read_def(self.def_file)
        #  default n_Texp value if not given
        if self.n_Texp_def is None:
            self.n_Texp_def=0.5
        #  default alpha_ref value if not given
        if self.alpha_ref_def is None:
            self.alpha_ref_def=0.07

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
            self._A, self.nu_lines, self._elower, self._gpp, self._jlower, self._jupper=exomolapi.pickup_gE(states,trans)        
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
                    self._A, self.nu_lines, self._elower, self._gpp, self._jlower, self._jupper=exomolapi.pickup_gE(states,trans)
                else:
                    Ax, nulx, elowerx, gppx, jlowerx, jupperx=exomolapi.pickup_gE(states,trans)
                    self._A=np.hstack([self._A,Ax])
                    self.nu_lines=np.hstack([self.nu_lines,nulx])
                    self._elower=np.hstack([self._elower,elowerx])
                    self._gpp=np.hstack([self._gpp,gppx])
                    self._jlower=np.hstack([self._jlower,jlowerx])
                    self._jupper=np.hstack([self._jupper,jupperx])
                    

        self.Tref=296.0        
        self.QTref=np.array(self.QT_interp(self.Tref))
        
        ##Line strength: input should be ndarray not jnp array
        self.Sij0=exomol.Sij0(self._A,self._gpp,self.nu_lines,self._elower,self.QTref)
        
        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)
        
        self.masking(mask)
        
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
        self._gpp=self._gpp[mask]
        self._jlower=self._jlower[mask]
        self._jupper=self._jupper[mask]
        
        #jnp arrays
        self.dev_nu_lines=jnp.array(self.nu_lines)
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.A=jnp.array(self._A)
        self.gamma_natural=gn(self.A)
        self.elower=jnp.array(self._elower)
        self.gpp=jnp.array(self._gpp)
        self.jlower=jnp.array(self._jlower,dtype=int)
        self.jupper=jnp.array(self._jupper,dtype=int)

        ##Broadening parameters 
        self.set_broadening()
        

    def set_broadening(self,broadf=True,alpha_ref_def=None,n_Texp_def=None):
        """setting broadening parameters
        
        Args:
           broadf: True=use .broad file for available jlower.
           alpha_ref: set default alpha_ref and apply it. None=use self.alpha_ref_def
           n_Texp_def: set default n_Texp and apply it. None=use self.n_Texp_def
        """
        if alpha_ref_def:
            self.alpha_ref_def = alpha_ref_def
        if n_Texp_def:
            self.n_Texp_def = n_Texp_def
            
        if broadf:
            bdat=exomolapi.read_broad(self.broad_file)
            codelv=exomolapi.check_bdat(bdat)
            print("Broadening code level=",codelv)
            if codelv=="a0":
                j2alpha_ref, j2n_Texp = exomolapi.make_j2b(bdat,\
                    alpha_ref_default=self.alpha_ref_def,\
                    n_Texp_default=self.n_Texp_def,\
                        jlower_max=np.max(self._jlower))
                self.alpha_ref=jnp.array(j2alpha_ref[self._jlower])
                self.n_Texp=jnp.array(j2n_Texp[self._jlower])                
            elif codelv=="a1":
                j2alpha_ref, j2n_Texp = exomolapi.make_j2b(bdat,\
                    alpha_ref_default=self.alpha_ref_def,\
                    n_Texp_default=self.n_Texp_def,\
                        jlower_max=np.max(self._jlower))                
                jj2alpha_ref, jj2n_Texp=exomolapi.make_jj2b(bdat,\
                    j2alpha_ref_def=j2alpha_ref,j2n_Texp_def=j2n_Texp,\
                        jupper_max=np.max(self._jupper))
                self.alpha_ref=jnp.array(jj2alpha_ref[self._jlower,self._jupper])
                self.n_Texp=jnp.array(jj2n_Texp[self._jlower,self._jupper])            
        else:
            print("No .broad file is given.")
            self.alpha_ref=jnp.array(self.alpha_ref_def*np.ones_like(self._jlower))
            self.n_Texp=jnp.array(self.n_Texp_def*np.ones_like(self._jlower))


            
        
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
           extension: extension list e.g. [".pf",".def",".trans.bz2",".states.bz2",".broad"]
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
        
        for ext in extension:
            if ext==".trans.bz2" and numtag is not None:
                ext="__"+numtag+ext
                
            if ext==".broad":
                pfname_arr=[tag[0]+"__H2"+ext,tag[0]+"__He"+ext,tag[0]+"__air"+ext]
                url = url_ExoMol()+molname_simple+"/"+tag[0]+"/"
            else:
                pfname_arr=[molec+ext]
                url = url_ExoMol()+molname_simple+"/"+tag[0]+"/"+tag[1]+"/"
                
            for pfname in pfname_arr:
                pfpath=url+pfname
                os.makedirs(str(self.path), exist_ok=True)
                print("Downloading "+pfpath)
                try:
                    urllib.request.urlretrieve(pfpath,str(self.path/pfname))
                except:
                    print("Error: Couldn't download "+ext+" file and save.")




class MdbHit(object):
    """ molecular database of ExoMol

    MdbExomol is a class for ExoMol.

    Attributes:
        nu_lines (nd array): line center (cm-1)
        Sij0 (nd array): line strength at T=Tref (cm)
        dev_nu_lines (jnp array): line center in device (cm-1)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient
        gamma_natural (jnp array): gamma factor of the natural broadening
        gamma_air (jnp array): gamma factor of air pressure broadening
        gamma_self (jnp array): gamma factor of self pressure broadening
        elower (jnp array): the lower state energy (cm-1)
        gpp (jnp array): statistical weight
        n_air (jnp array): air temperature exponent

    """

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
            if self.path.with_suffix('').exists():
                import os
                os.remove(self.path.with_suffix(''))
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

        #nd arrays using DRAM (not jnp, not in GPU)
        self.nu_lines = hapi.getColumn(molec, 'nu')
        self.Sij0 = hapi.getColumn(molec, 'sw')
        self.delta_air = hapi.getColumn(molec, 'delta_air')
        self.isoid = hapi.getColumn(molec,'local_iso_id')
        self.uniqiso=np.unique(self.isoid)

        self._A=hapi.getColumn(molec, 'a')
        self._n_air = hapi.getColumn(molec, 'n_air')
        self._gamma_air = hapi.getColumn(molec, 'gamma_air')
        self._gamma_self =hapi.getColumn(molec, 'gamma_self')
        self._elower = hapi.getColumn(molec, 'elower')
        self._gpp = hapi.getColumn(molec, 'gpp')

        ### MASKING ###
        mask=(self.nu_lines>self.nurange[0]-self.margin)\
        *(self.nu_lines<self.nurange[1]+self.margin)\
        *(self.Sij0>self.crit)
        
        self.masking(mask)
        
    def masking(self,mask):
        """applying mask and (re)generate jnp.arrays
        
        Args:
           mask: mask to be applied

        Note:
           We have nd arrays and jnp arrays. We apply the mask to nd arrays and generate jnp array from the corresponding nd array. For instance, self._A is nd array and self.A is jnp array.

        """
        
        #numpy float 64 Do not convert them jnp array
        self.nu_lines = self.nu_lines[mask]
        self.Sij0 = self.Sij0[mask]
        self.delta_air=self.delta_air[mask]
        self.isoid = self.isoid[mask]
        self.uniqiso=np.unique(self.isoid)

        ##numpy float 64 copy source for jnp
        self._A=self._A[mask]
        self._n_air = self._n_air[mask]
        self._gamma_air = self._gamma_air[mask]
        self._gamma_self = self._gamma_self[mask]
        self._elower = self._elower[mask]
        self._gpp = self._gpp[mask]

        #jnp.array copy from the copy sources
        self.dev_nu_lines=jnp.array(self.nu_lines)
        self.logsij0=jnp.array(np.log(self.Sij0))
        self.A=jnp.array(self._A)
        self.n_air=jnp.array(self._n_air)
        self.gamma_air = jnp.array(self._gamma_air)
        self.gamma_self = jnp.array(self._gamma_self)
        self.elower=jnp.array(self._elower)
        self.gpp=jnp.array(self._gpp)
        self.gamma_natural=gn(self.A)

        
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

    ####################################

    def ExomolQT(self,path):
        """use a partition function from ExoMol

        Args:
           path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"

        """
        #load pf

        self.empath = pathlib.Path(path)
        t0=self.empath.parents[0].stem        
        molec=t0+"__"+str(self.empath.stem)
        self.pf_file = self.empath/pathlib.Path(molec+".pf")
        if not self.pf_file.exists():
                self.exomol_pf_download(molec)

        pf=exomolapi.read_pf(self.pf_file)
        self.gQT=jnp.array(pf["QT"].to_numpy()) #grid QT
        self.T_gQT=jnp.array(pf["T"].to_numpy()) #T forgrid QT

    def exomol_pf_download(self,molec):
        """Downloading Exomol pf files

        Args: 
           molec: like "12C-16O__Li2015"

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

        ext=".pf"
        pfname=molec+ext
        pfpath=url+pfname
        os.makedirs(str(self.empath), exist_ok=True)
        print("Downloading "+pfpath)
        try:
            urllib.request.urlretrieve(pfpath,str(self.empath/pfname))
        except:
            print("Error: Couldn't download "+ext+" file and save.")

        
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
    

    def Qr_HAPI(self,Tarr):
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

    def Qr_line_HAPI(self,T):
        """Partition Function ratio using HAPI partition sum

        Args:
           T: temperature (K)

        Returns:
           Qr_line, partition function ratio array for lines [Nlines]

        Note: 
           Nlines=len(self.nu_lines)

        """
        qr_line=np.ones_like(self.isoid,dtype=np.float64)
        qrx=self.Qr_HAPI([T])
        for idx,iso in enumerate(self.uniqiso):
            mask=self.isoid==iso
            qr_line[mask]=qrx[0,idx]
        return qr_line

    def Qr_layer_HAPI(self,Tarr):
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
        qr=self.Qr_HAPI(Tarr)
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
    #mdb=MdbExomol("/home/kawahara/exojax/data/CO/12C-16O/Li2015/")    
    #mdb=MdbExomol("/home/kawahara/exojax/data/CH4/12C-1H4/YT34to10/",nurange=[6050.0,6150.0])
    mdb=MdbExomol('.database/H2O/1H2-16O/POKAZATEL',[4310.0,4320.0],crit=1.e-45) 

#    mask=mdb.A>1.e-42
#    mdb.masking(mask)
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/NH3/14N-1H3/CoYuTe/",nurange=[6050.0,6150.0])
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/H2S/1H2-32S/AYT2/",nurange=[6050.0,6150.0])
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/FeH/56Fe-1H/MoLLIST/",nurange=[6050.0,6150.0])
#    mdb=MdbExomol("/home/kawahara/exojax/data/exomol/NO/14N-16O/NOname/14N-16O__NOname")
