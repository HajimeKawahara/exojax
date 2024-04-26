"""Spectral Operators (Sop)

    The role of SOP is to apply various operators (essentially convolution) to a single spectrum, such as spin rotation, gaussian IP, RV shift etc.
    There are several convolution methods:
    - "exojax.signal.convolve": regular FFT-based convolution
    - "exojax.signal.ola": Overlap-and-Add based convolution

"""
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.spec.spin_rotation import convolve_rigid_rotation_ola
from exojax.spec.response import ipgauss, sampling
from exojax.spec.response import ipgauss_ola, sampling
from exojax.utils.grids import grid_resolution


class SopCommon():
    """Common Spectral Operator
    """

    def __init__(self, nu_grid, vrmax, convolution_method):
        """initialization of Sop

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
            resolution (float): wavenumber grid resolution, defined by nu/delta nu
            vrmax (float): velocity maximum to be applied in km/s
        """
        self.convolution_method_list = [
            "exojax.signal.convolve", "exojax.signal.ola"]
        self.convolution_method = convolution_method
        self.nu_grid = nu_grid
        self.vrmax = vrmax
        self.resolution = grid_resolution('ESLOG', self.nu_grid)
        self.generate_vrarray()
        self.ola_ndiv = 4

    def generate_vrarray(self):
        self.vrarray = velocity_grid(self.resolution, self.vrmax)

    def check_ola_reducible(self, spectrum):
        div_length = int(float(len(spectrum))/float(self.ola_ndiv))
        if len(spectrum) != self.ola_ndiv*div_length:
            raise ValueError("len(spectrum) can be reduced by self.ola_ndiv ="+str(self.ola_ndiv))
        return div_length


class SopRotation(SopCommon):
    """Spectral operator on rotation
    """

    def __init__(self,
                 nu_grid,
                 vsini_max=100.0,
                 convolution_method="exojax.signal.convolve"
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
        if self.convolution_method == self.convolution_method_list[0]:  # "exojax.signal.convolve"
            return convolve_rigid_rotation(spectrum, self.vrarray, vsini, u1, u2)
        elif self.convolution_method == self.convolution_method_list[1]:  # "exojax.signal.olaconv"
            div_length = self.check_ola_reducible(spectrum)
            folded_spectrum = spectrum.reshape((self.ola_ndiv, div_length))
            return convolve_rigid_rotation_ola(folded_spectrum, self.vrarray, vsini, u1, u2)
        else:
            raise ValueError("No convolution_method.")

    

class SopInstProfile(SopCommon):
    """Spectral operator on Instrumental profile and sampling
    """

    def __init__(self,
                 nu_grid,
                 vrmax=100.0,
                 convolution_method="exojax.signal.convolve"
                 ):
        super().__init__(nu_grid, vrmax, convolution_method)

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
        if self.convolution_method == self.convolution_method_list[0]:  # "exojax.signal.convolve"
            return ipgauss(spectrum, self.vrarray, standard_deviation)
        elif self.convolution_method == self.convolution_method_list[1]:  # "exojax.signal.olaconv"
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
