"""Continuum database (CDB) class.

* CdbCIA is the CDB for CIA
"""
import pathlib

import jax.numpy as jnp
import numpy as np

from exojax.database.cia.io import read_cia

__all__ = ["CdbCIA"]


class CdbCIA:
    def __init__(self, path, nurange=[-np.inf, np.inf], margin=10.0):
        """Continuum database for hitrancia.

        Args:
            path: path for hitrancia file
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
        """
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.path = pathlib.Path(path)
        if not self.path.exists():
            self.download()
        self.nucia, self.tcia, ac = read_cia(
            path, self.nurange[0] - self.margin, self.nurange[1] + self.margin
        )
        self.logac = jnp.array(np.log10(ac))
        self.tcia = jnp.array(self.tcia)
        self.nucia = jnp.array(self.nucia)

    def download(self):
        """Downloading hitrancia file.

        Note:
            The download URL is written in exojax.utils.url.
        """
        import os
        import urllib.request

        from exojax.utils.url import url_HITRANCIA

        try:
            os.makedirs(str(self.path.parent), exist_ok=True)
            url = url_HITRANCIA()+self.path.name
            data = urllib.request.urlopen(url).read()
            with open(str(self.path), mode="wb") as f:
                f.write(data)   
            #urllib.request.urlretrieve(url, str(self.path))
        except:
            print("HITRAN download failed")


if __name__ == "__main__":
    ciaH2H2 = CdbCIA("~/exojax/data/CIA/H2-H2_2011.cia", nurange=[4050.0, 4150.0])
    print(ciaH2H2.tcia)
