"""Spectral Operators (Sop)
"""
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.spec.response import ipgauss, sampling
from exojax.utils.grids import grid_resolution

class SopCommon():
    """Common Spectral Operator
    """
    def __init__(self, nu_grid, resolution, vrmax):
        """initialization of Sop

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
            resolution (float): wavenumber grid resolution, defined by nu/delta nu
            vrmax (float): velocity maximum to be applied in km/s
        """
        self.convolution_method = "exojax.signal.convolve"
        self.nu_grid = nu_grid
        self.vrmax = vrmax
        self.resolution = resolution
        self.generate_vrarray()   
        self.resolution = grid_resolution('ESLOG', self.nu_grid)

    def generate_vrarray(self):
        self.vrarray = velocity_grid(self.resolution, self.vrmax)


class SopRotation(SopCommon):
    """Spectral operator on rotation
    """
    def __init__(self,
                 nu_grid,
                 resolution,
                 vsini_max=100.0,
                 ):
        super().__init__(nu_grid, resolution, vsini_max)
        
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
            nd array: rotatinoal broaden spectrum
        """
        if self.convolution_method == "exojax.signal.convolve":
            return convolve_rigid_rotation(spectrum, self.vrarray, vsini, u1, u2)
        else:
            raise ValueError("No convolution_method")


class SopInstProfile(SopCommon):
    """Spectral operator on Instrumental profile and sampling
    """
    def __init__(self,
                 nu_grid,
                 resolution,
                 vrmax=100.0,
                 ):
        super().__init__(nu_grid, resolution, vrmax)
    
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
        if self.convolution_method == "exojax.signal.convolve":
            return ipgauss(spectrum, self.vrarray, standard_deviation)
        else:
            raise ValueError("No convolution_method")

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


