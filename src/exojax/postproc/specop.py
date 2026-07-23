"""Spectral Operators (Sop)

    The role of SOP is to apply various operators (essentially convolution) to a single spectrum, such as spin rotation, gaussian IP, RV shift etc.
    There are several convolution methods:
    - "exojax.signal.convolve": regular FFT-based convolution
    - "exojax.signal.ola": Overlap-and-Add based convolution

"""

import pathlib
import warnings

import numpy as np
import pandas as pd

from exojax.postproc.response import ipgauss, ipgauss_ola, sampling
from exojax.postproc.spin_rotation import (
    convolve_rigid_rotation,
    convolve_rigid_rotation_ola,
    convolve_rigid_rotation_trans,
    convolve_rigid_rotation_ola_trans
)
from exojax.utils.grids import grid_resolution, velocity_grid, delta_velocity_from_resolution
from exojax.utils.photometry import apparent_magnitude


class SopPhoto:
    """
    Spectral Operator for photometry
    """

    def __init__(
        self,
        filter_id,
        filter_bank="svo",
        path=".database/filter",
        download=True,
        up_resolution_factor=32.0,
        factor=1.0e20,
    ):
        """initialization

        Args:
            filter_id (str): filter id (e.g. "2MASS/2MASS.Ks")
            filter_bank (str, optional) filter bank. Defaults to "svo".
            path (str, optional): path to store the data. Defaults to ".database/filter".
            download (bool, optional): Download or uses the saved data. Defaults to True.
            up_resolution_factor (float, optional): _description_. Defaults to 32.0.
            factor (float): factor to normalize the flux to prevent underflow
        """
        self.up_resolution_factor = up_resolution_factor
        self.factor = factor

        self.filter_id = filter_id
        self.facility, self.filter_tag = self.filter_id.split("/")
        self.filter_bank = filter_bank

        self.path = pathlib.Path(path) / filter_bank / self.facility
        self.filepath = self.path / f"{self.filter_tag}.csv"
        self.infopath = self.path / f"{self.filter_tag}.info.csv"

        self.download = download

        if download:
            self.download_filter()
            self.save_filter()
        else:
            self.load_filter()

        self.computes_interpolated_transmission_filter()

    def load_filter(self):
        """loads the filter from the saved dataset

        Raises:
            ValueError: datasets not found
        """
        if not self.filepath.exists():
            raise ValueError(f"filter file {self.filepath} does not exist.")

        print("load ", self.filepath)
        df = pd.read_csv(self.filepath)
        self.nu_ref = df["nu"].values
        self.transmission_ref = df["transmission"].values
        print("load ", self.infopath)
        dg = pd.read_csv(self.infopath)
        self.nu_center = dg["nu_center"].values[0]
        self.f0_nu_cgs = dg["f0_nu_cgs"].values[0]
        self.resolution_photo = dg["resolution_photo"].values[0]

    def download_filter(self):
        """downloads the filter

        Raises:
            ValueError: No filter bank
        """
        if self.filter_bank == "svo":
            self.download_filter_svo()
        else:
            raise ValueError("No filter bank (e.g. svo).")

    def download_filter_svo(self):
        """downloads the filter from SVO"""
        from exojax.utils.photometry import (
            average_resolution,
            download_filter_from_svo,
            download_zero_magnitude_flux_from_svo,
        )

        self.nu_ref, self.transmission_ref = download_filter_from_svo(self.filter_id)
        self.nu_center, self.f0_nu_cgs = download_zero_magnitude_flux_from_svo(
            self.filter_id, unit="cm-1"
        )
        self.resolution_photo = (
            average_resolution(self.nu_ref) * self.up_resolution_factor
        )
        print("resolution_photo=", self.resolution_photo)

    def save_filter(self):
        """save the filter dataset"""
        import os

        os.makedirs(str(self.path), exist_ok=True)
        pd.DataFrame(
            {
                "nu": self.nu_ref,
                "transmission": self.transmission_ref,
            }
        ).to_csv(self.filepath)
        print("save ", self.filepath)
        pd.DataFrame(
            {
                "nu_center": [self.nu_center],
                "f0_nu_cgs": [self.f0_nu_cgs],
                "resolution_photo": [self.resolution_photo],
            }
        ).to_csv(self.infopath)
        print("save ", self.infopath)

    def computes_interpolated_transmission_filter(self, xsmode="premodit"):
        """computes

        Args:
            xsmode (str, optional): xsmode for wavenumber_grid. Defaults to "premodit".
        """
        from exojax.utils.grids import wavenumber_grid
        from exojax.utils.instfunc import nx_even_from_resolution_eslog

        Nx = nx_even_from_resolution_eslog(
            np.min(self.nu_ref), np.max(self.nu_ref), self.resolution_photo
        )
        self.nu_grid_filter, self.wav_filter, self.res_filter = wavenumber_grid(
            self.nu_ref[0], self.nu_ref[-1], Nx, unit="cm-1", xsmode=xsmode
        )
        self.transmission_filter = np.interp(
            self.nu_grid_filter, self.nu_ref, self.transmission_ref
        )

    def apparent_magnitude(self, flux):
        """computes apparent magnitude

        Args:
            flux (array): flux in the unit of erg/s/cm2/cm-1, the same dimension as self.transmission_filter

        Returns:
            float: apparent magnitude
        """
        return apparent_magnitude(
            flux,
            self.nu_grid_filter,
            self.transmission_filter,
            self.f0_nu_cgs,
            self.factor,
        )


class SopCommonConv:
    """Common Spectral Operator for convloution type operators"""

    def __init__(self, nu_grid, vrmax, convolution_method):
        """initialization of Sop

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
            resolution (float): wavenumber grid resolution, defined by nu/delta nu
            vrmax (float): velocity maximum to be applied in km/s
        """
        self.convolution_method_list = ["exojax.signal.convolve", "exojax.signal.ola"]
        self.convolution_method = convolution_method
        self.nu_grid = nu_grid
        self.vrmax = vrmax
        self.resolution = grid_resolution("ESLOG", self.nu_grid)
        self.generate_vrarray()
        self.ola_ndiv = 4

    def generate_vrarray(self):
        self.vrarray = velocity_grid(self.resolution, self.vrmax)

    def check_ola_reducible(self, spectrum):
        div_length = int(float(len(spectrum)) / float(self.ola_ndiv))
        if len(spectrum) != self.ola_ndiv * div_length:
            raise ValueError(
                "len(spectrum) can be reduced by self.ola_ndiv =" + str(self.ola_ndiv)
            )
        return div_length


class SopRotation(SopCommonConv):
    """Spectral operator on rotation"""

    def __init__(
        self, nu_grid, vsini_max=100.0, convolution_method="exojax.signal.convolve"
    ):
        super().__init__(nu_grid, vsini_max, convolution_method)

    def rigid_rotation(self, spectrum, vsini, u1, u2):
        """apply a rigid rotation

        Args:
            spectrum (nd array): 1D spectrum
            vsini (float): V sini in km/s
            u1 (float): Limb darkening parameter u1
            u2 (float): Limb darkening parameter u2

        Raises:
            ValueError: _description_

        Returns:
            nd array: rotationally broaden spectrum
        """
        if (
            self.convolution_method == self.convolution_method_list[0]
        ):  # "exojax.signal.convolve"
            return convolve_rigid_rotation(spectrum, self.vrarray, vsini, u1, u2)
        elif (
            self.convolution_method == self.convolution_method_list[1]
        ):  # "exojax.signal.olaconv"
            div_length = self.check_ola_reducible(spectrum)
            folded_spectrum = spectrum.reshape((self.ola_ndiv, div_length))
            return convolve_rigid_rotation_ola(
                folded_spectrum, self.vrarray, vsini, u1, u2
            )
        else:
            raise ValueError("No convolution_method.")

    def rigid_rotation_trans(self, spectrum, vsini):
        """apply a rigid rotation for transmission geometry

        Args:
            spectrum (nd array): 1D spectrum
            vsini (float): V sini in km/s

        Returns:
            nd array: rotationally broaden spectrum
        """
        dv = delta_velocity_from_resolution(self.resolution)
        if (
            self.convolution_method == self.convolution_method_list[0]
        ):  # "exojax.signal.convolve"
            return convolve_rigid_rotation_trans(spectrum, self.vrarray, dv, vsini)
        elif (
            self.convolution_method == self.convolution_method_list[1]
        ):  # "exojax.signal.olaconv"
            div_length = self.check_ola_reducible(spectrum)
            folded_spectrum = spectrum.reshape((self.ola_ndiv, div_length))
            return convolve_rigid_rotation_ola_trans(folded_spectrum, self.vrarray, dv, vsini)


class SopInstProfile(SopCommonConv):
    """Spectral operator on Instrumental profile and sampling"""

    def __init__(
        self, nu_grid, vrmax=100.0, convolution_method="exojax.signal.convolve"
    ):
        super().__init__(nu_grid, vrmax, convolution_method)

    def check_vrmax(self, standard_deviation):
        """Check vrmax for a Gaussian instrumental profile.

        The instrumental resolution is represented by standard_deviation;
        the model wavenumber grid is intentionally not used in this check.
        This method is intended for an explicit check before JIT compilation
        and is also called automatically by ipgauss.

        Args:
            standard_deviation (float): standard deviation of the Gaussian
                instrumental profile in km/s

        Warns:
            UserWarning: if vrmax covers fewer than five standard deviations
        """
        try:
            standard_deviation = float(standard_deviation)
        except (TypeError, ValueError):
            # A traced value cannot be checked without a runtime callback.
            return

        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            return

        required_nsigma = 5.0
        required_velocity_span = required_nsigma * standard_deviation
        if self.vrmax < required_velocity_span:
            covered_nsigma = self.vrmax / standard_deviation
            warnings.warn(
                "`vrmax` for the Gaussian instrumental profile is too small: "
                f"vrmax={self.vrmax:.3g} km/s covers "
                f"{covered_nsigma:.3g} Gaussian standard deviations for "
                f"standard_deviation={standard_deviation:.3g} km/s. "
                f"Use `vrmax >= {required_velocity_span:.3g} km/s` when "
                "constructing `SopInstProfile` to cover at least "
                f"{required_nsigma:g} standard deviations. "
                "For a variable standard deviation, set `vrmax` "
                "from its maximum expected value before JIT compilation.",
                UserWarning,
                stacklevel=2,
            )

    def ipgauss(self, spectrum, standard_deviation):
        """Gaussian Instrumental Profile

        Args:
            spectrum (nd array): 1D spectrum
            standard_deviation (float): standard deviation of Gaussian in km/s

        Raises:
            ValueError: _description_

        Returns:
            array: IP applied spectrum
        """
        self.check_vrmax(standard_deviation)
        if (
            self.convolution_method == self.convolution_method_list[0]
        ):  # "exojax.signal.convolve"
            return ipgauss(spectrum, self.vrarray, standard_deviation)
        elif (
            self.convolution_method == self.convolution_method_list[1]
        ):  # "exojax.signal.olaconv"
            div_length = self.check_ola_reducible(spectrum)
            folded_spectrum = spectrum.reshape((self.ola_ndiv, div_length))
            return ipgauss_ola(folded_spectrum, self.vrarray, standard_deviation)
        else:
            raise ValueError("No convolution_method.")

    def sampling(self, spectrum, radial_velocity, nu_grid_sampling):
        """sampling to instrumental wavenumber grid (not necessary ESLOG nor ESLIN)

        Args:
            spectrum (nd array): 1D spectrum
            radial_velocity (float): radial velocity in km/s
            nu_grid_sampling (array): instrumental wavenumber grid

        Returns:
            array: inst sampled spectrum
        """
        return sampling(nu_grid_sampling, self.nu_grid, spectrum, radial_velocity)
