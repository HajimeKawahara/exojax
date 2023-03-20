"""Operators on Spectra
"""
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
from exojax.spec.response import ipgauss_sampling
from exojax.utils.grids import grid_resolution

class SosCommon():
    """Common Single Operator on Spectra
    """
    def __init__(self, nu_grid, vrmax, resolution):
        """initialization of sos

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
        """
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
                 vsini_max=100.0,
                 ):
        super().__init__(nu_grid, vsini_max)
        self.method = ""

    def rigid_rotation(self, spectrum, vsini, u1, u2):
        convolve_rigid_rotation(spectrum, self.vrarray, vsini, u1, u2)




