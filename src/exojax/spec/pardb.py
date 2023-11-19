"""Particulates Database

- Cloud
- Haze (in future)

"""

from tkinter import Y
import numpy as np
import jax.numpy as jnp
from jax import vmap
import pathlib
from exojax.atm.viscosity import calc_vfactor, eta_Rosner
from exojax.atm.vterm import terminal_velocity

__all__ = ["PdbCloud"]


class PdbCloud(object):
    def __init__(self, condensate, bkgatm, nurange=[-np.inf, np.inf], margin=10.0, path="./.database/particulates"):
        """Particulates Database for clouds

        Args:
            condensate: condensate, such as NH3, H2O, MgSiO3 etc
            bkgatm: background atmosphere, such as H2, air
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
        """
        self.path = pathlib.Path(path)
        self.download()
        self.condensate = condensate
        self.bkgatm = bkgatm
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.vfactor, self.trange_vfactor = calc_vfactor(atm="H2")

    def download(self):
        """Downloading virga refractive index data

        Note:
           The download URL is written in exojax.utils.url.
        """
        import urllib.request
        import os
        from exojax.utils.url import url_virga
        try:
            os.makedirs(str(self.path), exist_ok=True)
            filepath=self.path/"virga.zip"
            if (filepath).exists():
                print(str(filepath), " exists.")
            else:
                print("Downloading ", url_virga())
                data = urllib.request.urlopen(url_virga()).read()
                with open(str(filepath), mode="wb") as f:
                    f.write(data)
                #urllib.request.urlretrieve(url_virga(), str(filepath))
                
        except:
            print('VIRGA download failed')


    def set_saturation_pressure_list(self):
        from exojax.atm.psat import psat_ammonia_AM01, psat_water_AM01
        self.saturation_pressure_list = {"NH3": psat_ammonia_AM01, "H2O": psat_water_AM01}

    def set_condensate_density(self):
        from exojax.atm.condensate import condensate_density
        self.rhoc = condensate_density[self.condensate]

    def saturation_pressure(self):
        return self.saturation_pressure_list[self.condensate]

if __name__ == "__main__":
    pdb = PdbCloud("NH3","H2")