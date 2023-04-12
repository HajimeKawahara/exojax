"""Operators on Spectra
"""
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.spec.response import ipgauss, sampling
from exojax.utils.grids import grid_resolution

class SosCommon():
    """Common Single Operator on Spectra
    """
    def __init__(self, nu_grid, resolution, vrmax):
        """initialization of sos

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
        """
        self.convolution_method = "exojax.signal.convolve"
        self.nu_grid = nu_grid
        self.vrmax = vrmax
        self.resolution = resolution
        self.generate_vrarray()   
        self.resolution = grid_resolution('ESLOG', self.nu_grid)

    def generate_vrarray(self):
        self.vrarray = velocity_grid(self.resolution, self.vrmax)


class SosRotaion(SosCommon):
    
    def __init__(self,
                 nu_grid,
                 resolution,
                 vsini_max=100.0,
                 ):
        super().__init__(nu_grid, resolution, vsini_max)
        
    def rigid_rotation(self, spectrum, vsini, u1, u2):
        if self.convolution_method == "exojax.signal.convolve":
            return convolve_rigid_rotation(spectrum, self.vrarray, vsini, u1, u2)
        else:
            raise ValueError("No convolution_method")


class SosInstProfile(SosCommon):
    def __init__(self,
                 nu_grid,
                 resolution,
                 vrmax=100.0,
                 ):
        super().__init__(nu_grid, resolution, vrmax)
    
    def ipgauss(self, spectrum, standard_deviation):
        if self.convolution_method == "exojax.signal.convolve":
            return ipgauss(spectrum, self.vrarray, standard_deviation)
        else:
            raise ValueError("No convolution_method")

    def sampling(self, spectrum, radial_velocity, nu_grid_sampling):
        return sampling(nu_grid_sampling, self.nu_grid, spectrum, radial_velocity)


