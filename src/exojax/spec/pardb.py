""" Particulates Database

- Cloud
- Haze (in future)

"""

import numpy as np
import pathlib
from exojax.spec.mie import evaluate_miegrid
from exojax.spec.mie import compute_mieparams_cgs_from_miegrid


__all__ = ["PdbCloud"]


class PdbCloud(object):
    def __init__(
        self,
        condensate,
        nurange=None,
        margin=10.0,
        path="./.database/particulates/virga",
        download=True,
        refrind_path=None,
    ):
        """Particulates Database for clouds

        Args:
            condensate: condensate, such as NH3, H2O, MgSiO3 etc
            nurange: wavenumber range list (cm-1) or wavenumber array, default to None, then set to the range of the refraction index file
            margin: margin for nurange (cm-1)
            path: database path
            download: allow download from virga. default to True
            refrinf_path: manual setting of path to refraction index file. default to None
        """
        self.path = pathlib.Path(path)
        self.condensate = condensate
        if download:
            self.download_and_unzip()
        if refrind_path is not None:
            self.refrind_path = pathlib.Path(refrind_path)

        self.load_virga()

        if nurange is None:
            tip = 1.0e-11
            nurange = [
                self.refraction_index_wavenumber[0] + tip,
                self.refraction_index_wavenumber[-1] - tip,
            ]
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin
        self.set_saturation_pressure_list()
        self.set_condensate_substance_density()

        # Mie scattering
        self.ready_mie = False
        self.N0 = 1.0  # reference number density (cm^-3) to compute beta_0 (mie coefficient as an input of PyMieScatt)
        self.set_miegrid_filename()
        self.set_miegrid_path()

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
        """loads VIRGA refraction index
        
        Notes:
            self.refraction_index_wavenumber is in ascending and self.refraction_index_wavelength_nm is in descending.
            Each component of both corresponds to that of self.refraction_index. (no need to put [::-1])   
        
        """
        from exojax.spec.unitconvert import wav2nu

        _, wave, nn, kk = np.loadtxt(
            open(self.refrind_path, "rt").readlines(), unpack=True, usecols=[0, 1, 2, 3]
        )
        self.refraction_index_wavenumber = wav2nu(wave, "um")  # wave in micron ascending
        self.refraction_index_wavelength_nm = wave * 1.0e3  #descending
        self.refraction_index = nn + kk * (
            1j
        )  # m = n + ik because PyMieScatt uses this form (not m = n - ik).



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

    def set_condensate_substance_density(self):
        """sets condensate density

        Note:
            "condensate substance density" means the mass substance density of the condensate matter itself.
            For instance, in the room temperature, for liquid water clouds, condensate_substance_density ~ 1 g/cm3
            Notice that the mass density of the condensates in the atmosphere is the different quantity.

        """
        from exojax.atm.condensate import condensate_substance_density

        self.condensate_substance_density = condensate_substance_density[
            self.condensate
        ]

    def saturation_pressure(self, temperatures):
        return self.saturation_pressure_solid_list[self.condensate](temperatures)

    def set_miegrid_filename(self, miegrid_filename=None):
        if miegrid_filename is None:
            self.miegrid_filename = "miegrid_lognorm_" + self.condensate + ".mg"
        elif miegrid_filename == "auto":
            raise ValueError("not implemented yet")
        else:
            self.miegrid_filename = miegrid_filename

    def set_miegrid_path(self, miegrid_path=None):
        if miegrid_path is None:
            self.miegrid_path = self.path / pathlib.Path(self.miegrid_filename + ".npz")
        else:
            self.miegrid_path = pathlib.Path(miegrid_path + ".npz")

        if self.miegrid_path.exists():
            print("Miegrid file exists:", str(self.miegrid_path))
        else:
            print("Miegrid file does not exist at ", str(self.miegrid_path))
            print(
                "Generate miegrid file using pdb.generate_miegrid if you use Mie scattering"
            )

    def load_miegrid(self):
        """loads Miegrid

        Raises:
            ValueError: _description_
        """
        from exojax.spec.mie import read_miegrid

        if self.miegrid_path.exists():
            self.miegrid, self.rg_arr, self.sigmag_arr = read_miegrid(self.miegrid_path)
            self.ready_mie = True
            print(
                "pdb.miegrid, pdb.rg_arr, pdb.sigmag_arr are now available. The Mie scattering computation is ready."
            )
        else:
            raise ValueError("Miegrid file Not Found.")

        self.reset_miegrid_for_nurange()

    def reset_miegrid_for_nurange(self):
        """Resets wavenumber indices of miegrid, refraction_index_wavenumber, refraction_index_wavelength_nm, refraction_index
        Raises:
            ValueError: _description_
        """
        ist, ien = np.searchsorted(self.refraction_index_wavenumber, self.nurange)
        if ist == 0 or ien == len(self.refraction_index_wavenumber):
            print("pdb.nurange:", self.nurange, "cm-1")
            print(
                "Miegrid wavenumber range:[",
                self.refraction_index_wavenumber[0],
                ",",
                self.refraction_index_wavenumber[-1],
                "] cm-1",
            )
            raise ValueError(
                "Miegrid wavenumber is out of the input range (pdb.nurange)."
            )
        self._redefine_wavenumber_indices(ist - 1, ien + 1)

    def _redefine_wavenumber_indices(self, i, j):
        self.refraction_index_wavenumber = self.refraction_index_wavenumber[i:j]
        self.miegrid = self.miegrid[:, :, i:j, :]
        self.refraction_index_wavelength_nm = self.refraction_index_wavelength_nm[i:j]
        self.refraction_index = self.refraction_index[i:j]

    def generate_miegrid(
        self,
        sigmagmin=1.0001,
        sigmagmax=4.0,
        Nsigmag=10,
        log_rg_min=-7.0,
        log_rg_max=-3.0,
        Nrg=40,
    ):
        """generates miegrid assuming lognormal size distribution

        Args:
            sigmagmin (float, optional): sigma_g minimum. Defaults to 1.0001.
            sigmagmax (float, optional): sigma_g maximum. Defaults to 4.0.
            Nsigmag (int, optional): the number of the sigmag grid. Defaults to 10.
            log_rg_min (float, optional): log r_g (cm) minimum . Defaults to -7.0.
            log_rg_max (float, optional): log r_g (cm) minimum. Defaults to -3.0.
            Nrg (int, optional): the number of the rg grid. Defaults to 40.

        Note:
            it will take a bit long time. See src/exojax/tests/generate_pdb.py as a sample code.

        """

        from exojax.spec.mie import make_miegrid_lognormal

        make_miegrid_lognormal(
            self.refraction_index,
            self.refraction_index_wavelength_nm,
            str(self.path / pathlib.Path(self.miegrid_filename)),
            sigmagmin,
            sigmagmax,
            Nsigmag,
            log_rg_min,
            log_rg_max,
            Nrg,
            self.N0,
        )
        print(str(self.miegrid_filename), " was generated.")

    def miegrid_interpolated_values(self, rg, sigmag):
        """evaluates the value at rg and sigmag by interpolating miegrid

        Args:
            rg (float): rg parameter in lognormal distribution
            sigmag (float): sigma_g parameter in lognormal distribution

        Note:
            beta derived here is in the unit of 1/Mm (Mega meter) for diameter
            multiply 1.e-8 to convert to 1/cm for radius.

        Returns:
            _type_: evaluated values of miegrid, output of MieQ_lognormal Bext (1/Mm), Bsca, Babs, G, Bpr, Bback, Bratio (wavenumber, number of mieparams)
        """

        return evaluate_miegrid(rg, sigmag, self.miegrid, self.rg_arr, self.sigmag_arr)

    def mieparams_cgs_at_refraction_index_wavenumber_from_miegrid(self, rg, sigmag):
        """computes Mie parameters in the original refraction index wavenumber, i.e. extinction coeff, sinigle scattering albedo, asymmetric factor from miegrid

        Args:
            rg (float): rg parameter in lognormal distribution
            sigmag (float): sigma_g parameter in lognormal distribution

        Note:
            Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction
            The output returns are computed at self.refraction_index_wavenumber

        Returns:
            sigma_extinction, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            sigma_scattering, scattering cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            g, asymmetric factor (mean g)
        """

        return compute_mieparams_cgs_from_miegrid(
            rg, sigmag, self.miegrid, self.rg_arr, self.sigmag_arr, self.N0
        )


if __name__ == "__main__":
    pdb = PdbCloud("NH3")
    pdb.load_miegrid()

