"""Particulates Database

- Cloud
- Haze (in future)

"""

import numpy as np
import jax.numpy as jnp
from jax import vmap
import pathlib
from exojax.atm.psat import psat_Fe_AM01
from exojax.atm.viscosity import calc_vfactor, eta_Rosner
from exojax.atm.vterm import terminal_velocity

__all__ = ["PdbCloud"]


class PdbCloud(object):
    def __init__(
        self,
        condensate,
        nurange=[-np.inf, np.inf],
        margin=10.0,
        path="./.database/particulates/virga",
    ):
        """Particulates Database for clouds

        Args:
            condensate: condensate, such as NH3, H2O, MgSiO3 etc
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)
        """
        self.path = pathlib.Path(path)
        self.condensate = condensate
        self.download_and_unzip()
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin

        self.set_saturation_pressure_list()
        self.set_condensate_density()

    def download_and_unzip(self):
        """Downloading virga refractive index data

        Note:
            The download URL is written in exojax.utils.url.
        """
        import urllib.request
        import os
        import shutil
        from exojax.utils.url import url_virga
        from exojax.utils.files import find_files_by_extension
        from exojax.utils.files import get_file_names_without_extension

        try:
            os.makedirs(str(self.path), exist_ok=True)
            filepath = self.path / "virga.zip"
            if (filepath).exists():
                print(
                    str(filepath),
                    " exists. Remove it if you wanna re-download and unzip.",
                )
            else:
                print("Downloading ", url_virga())
                # urllib.request.urlretrieve(url_virga(), str(filepath))
                data = urllib.request.urlopen(url_virga()).read()
                with open(str(filepath), mode="wb") as f:
                    f.write(data)
                shutil.unpack_archive(str(filepath), str(self.path))
            self.virga_condensates = get_file_names_without_extension(
                find_files_by_extension(str(self.path), ".refrind")
            )
            if self.condensate in self.virga_condensates:
                self.refrind_path = self.path / pathlib.Path(
                    self.condensate + ".refrind"
                )
                print("Refractive index file found: ", self.refrind_path)
            else:
                print(
                    "No refrind file found. Refractive indices of ",
                    self.virga_condensates,
                    "are available.",
                )
        except:
            print("VIRGA refractive index download failed")

    def load_virga(self):
        from exojax.spec.unitconvert import wav2nu

        _, wave, nn, kk = np.loadtxt(
            open(self.refrind_path, "rt").readlines(), unpack=True, usecols=[0, 1, 2, 3]
        )
        self.refraction_index_wavenumber = wav2nu(wave, "um")  # wave in micron
        self.refraction_index = nn + kk * (1j)

    def set_saturation_pressure_list(self):
        from exojax.atm.psat import (
            psat_ammonia_AM01,
            psat_water_AM01,
            psat_Fe_AM01,
            psat_enstatite_AM01,
        )

        self.saturation_pressure_solid_list = {
            "NH3": psat_ammonia_AM01,
            "H2O": psat_water_AM01,
            "MgSiO3": psat_enstatite_AM01,
            "Fe": psat_Fe_AM01,
        }

    def set_condensate_density(self):
        from exojax.atm.condensate import condensate_density

        self.rhoc = condensate_density[self.condensate]

    def saturation_pressure(self, temperatures):
        return self.saturation_pressure_solid_list[self.condensate](temperatures)


if __name__ == "__main__":
    pdb = PdbCloud("NH3")
