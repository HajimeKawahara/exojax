[1mdiff --git a/src/exojax/spec/moldb.py b/src/exojax/spec/moldb.py[m
[1mindex 60175c6..c2c3623 100644[m
[1m--- a/src/exojax/spec/moldb.py[m
[1m+++ b/src/exojax/spec/moldb.py[m
[36m@@ -418,8 +418,24 @@[m [mclass MdbHit(object):[m
         # downloading[m
         self.path = pathlib.Path(path)[m
         numinf, numtag = hitranapi.read_path(self.path)[m
[31m-        if not self.path.exists():[m
[31m-            self.download()[m
[32m+[m
[32m+[m[32m        if numinf is None:[m
[32m+[m[32m            if not self.path.exists():[m
[32m+[m[32m                self.download()[m
[32m+[m[32m        else:[m
[32m+[m[32m            molec = str(self.path.name)[0:2][m
[32m+[m[32m            self.nurange = [np.min(nurange), np.max(nurange)][m
[32m+[m[41m            [m
[32m+[m[32m            imin = np.searchsorted([m
[32m+[m[32m                numinf, self.nurange[0], side='right')-1  # left side[m
[32m+[m[32m            imax = np.searchsorted([m
[32m+[m[32m                numinf, self.nurange[1], side='right')-1  # left side[m
[32m+[m[32m            for k, i in enumerate(range(imin, imax+1)):[m
[32m+[m[32m                sub_file = self.path.stem / \[m
[32m+[m[32m                    pathlib.Path(molec+'_'+numtag[i]+'_HITEMP2010.par')[m
[32m+[m[32m                print(sub_file)[m
[32m+[m[32m                if not sub_file.exists():[m
[32m+[m[32m                    self.download(numtag=numtag[i])[m
 [m
         # extract?[m
         if extract:[m
[36m@@ -510,7 +526,7 @@[m [mclass MdbHit(object):[m
         self.gpp = jnp.array(self._gpp)[m
         self.gamma_natural = gn(self.A)[m
 [m
[31m-    def download(self):[m
[32m+[m[32m    def download(self, numtag=None):[m
         """Downloading HITRAN/HITEMP par file.[m
 [m
         Note:[m
[36m@@ -519,6 +535,8 @@[m [mclass MdbHit(object):[m
         import urllib.request[m
         from exojax.utils.url import url_HITRAN12[m
         from exojax.utils.url import url_HITEMP[m
[32m+[m[32m        from exojax.utils.url import url_HITEMP10[m
[32m+[m[32m        import os[m
 [m
         try:[m
             url = url_HITRAN12()+self.path.name[m
[36m@@ -528,11 +546,29 @@[m [mclass MdbHit(object):[m
             print('HITRAN download failed')[m
         try:[m
             url = url_HITEMP()+self.path.name[m
[31m-            print(url)[m
             urllib.request.urlretrieve(url, str(self.path))[m
         except:[m
[32m+[m[32m            print(url)[m
             print('HITEMP download failed')[m
 [m
[32m+[m[32m        molec = str(self.path.name)[0:2][m
[32m+[m[32m        if molec == '01':[m
[32m+[m[32m            os.makedirs(str(self.path), exist_ok=True)[m
[32m+[m[32m            dldir = 'H2O_line_list/'[m
[32m+[m[32m        if molec == '02':[m
[32m+[m[32m            os.makedirs(str(self.path), exist_ok=True)[m
[32m+[m[32m            dldir = 'CO2_line_list/'[m
[32m+[m[32m        flname = molec+'_'+numtag+'_HITEMP2010.zip'[m
[32m+[m[32m        url = url_HITEMP10()+dldir+flname[m
[32m+[m[32m        urllib.request.urlretrieve(url, str(self.path/flname))[m
[32m+[m[32m        exit()[m
[32m+[m[32m        try:[m
[32m+[m[32m            url = url_HITEMP10()+dlpath[m
[32m+[m[32m            print(url)[m
[32m+[m[32m            urllib.request.urlretrieve(url, str(self.path/dlpath))[m
[32m+[m[32m        except:[m
[32m+[m[32m            print('HITEMP2010 download failed')[m
[32m+[m[32m        exit()[m
     ####################################[m
 [m
     def ExomolQT(self, path):[m
[1mdiff --git a/src/exojax/utils/url.py b/src/exojax/utils/url.py[m
[1mindex 97817a9..35d8940 100644[m
[1m--- a/src/exojax/utils/url.py[m
[1m+++ b/src/exojax/utils/url.py[m
[36m@@ -34,6 +34,16 @@[m [mdef url_HITEMP():[m
     return url[m
 [m
 [m
[32m+[m[32mdef url_HITEMP10():[m
[32m+[m[32m    """return URL for HITEMP2010.[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m       URL for HITEMP2010 db[m
[32m+[m[32m    """[m
[32m+[m[32m    url = u'https://hitran.org/hitemp/data/HITEMP-2010/'[m
[32m+[m[32m    return url[m
[32m+[m
[32m+[m
 def url_ExoMol():[m
     """return URL for ExoMol.[m
 [m
