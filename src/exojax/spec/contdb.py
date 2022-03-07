"""Continuum database (CDB) class.

* CdbCIA is the CDB for CIA
"""
import numpy as np
import jax.numpy as jnp
from jax import lax
import pathlib
from exojax.spec.hitrancia import read_cia

__all__ = ['CdbCIA']


class CdbCIA(object):
    def __init__(self, path, nurange=[-np.inf, np.inf], margin=10.0):
        """Continuum database for HITRAN CIA.

        Args:
           path: path for HITRAN cia file
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)
        """
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.path = pathlib.Path(path)
        if not self.path.exists():
            self.download()
        self.nucia, self.tcia, ac = read_cia(
            path, self.nurange[0]-self.margin, self.nurange[1]+self.margin)
        self.logac = jnp.array(np.log10(ac))
        self.tcia = jnp.array(self.tcia)
        self.nucia = jnp.array(self.nucia)

    def download(self):
        """Downloading HITRAN cia file.

        Note:
           The download URL is written in exojax.utils.url.
        """
        import urllib.request
        import os
        from exojax.utils.url import url_HITRANCIA
        try:
            os.makedirs(str(self.path.parent), exist_ok=True)
            url = url_HITRANCIA()+self.path.name
            urllib.request.urlretrieve(url, str(self.path))
        except:
            print(url)
            print('HITRAN download failed')


if __name__ == '__main__':
    ciaH2H2 = CdbCIA(
        '/home/kawahara/exojax/data/CIA/H2-H2_2011.cia', nurange=[4050.0, 4150.0])
    print(ciaH2H2.tcia)
