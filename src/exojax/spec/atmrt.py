"""Atmospheric Radiative Transfer (art) class
"""
import numpy as np
from exojax.spec.planck import piBarr
from exojax.spec.rtransfer import rtrun as rtrun_emis_pure_absorption
from exojax.spec.rtransfer import dtauM
import jax.numpy as jnp


class ArtCommon():
    """Common Atmospheric Radiative Transfer
    """
    __slots__ = [
        "artinfo",
    ]

    def __init__(self, nu_grid, pressure_layer_params):
        """initialization of art

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
            pressure_layer_params (list[3]): (bottom pressure in bar, top pressure in bar, # of layers)
        """
        self.artinfo = None
        self.method = None  # which art is used
        self.ready = False  # ready for art computation

        self.nu_grid = nu_grid
        self.log_pressure_bottom = np.log10(pressure_layer_params[0])
        self.log_pressure_top = np.log10(pressure_layer_params[1])
        self.n_pressure_layer = pressure_layer_params[2]
        self.init_pressure_profile()

    def init_pressure_profile(self):
        from exojax.spec.rtransfer import pressure_layer
        self.pressure, self.dParr, self.k = pressure_layer(
            logPtop=self.log_pressure_top,
            logPbtm=self.log_pressure_bottom,
            NP=self.n_pressure_layer,
            mode='ascending',
            numpy=False)

    def constant_mmr_profile(self, MMR):
        return MMR * np.ones_like(self.pressure)


class ArtEmisPure(ArtCommon):
    """Atmospheric RT for emission w/ pure absorption

    Attributes:
        pressure_layer: pressure profile in bar
        
    """
    def __init__(self, nu_grid, pressure_layer_params=[1.e-8, 1.e2, 100]):
        """initialization of ArtEmisPure

        
        """
        super().__init__(nu_grid, pressure_layer_params)

        #default setting
        self.method = "emission_with_pure_absorption"

    def dtau_lines(self, xsmatrix, mmr_profile, molmass, gravity):
        return dtauM(self.dParr, jnp.abs(xsmatrix), mmr_profile, molmass, gravity)

    def run(self, dtau, temperature):
        sourcef = piBarr(temperature, self.nu_grid)
        return rtrun_emis_pure_absorption(dtau, sourcef)
