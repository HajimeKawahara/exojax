"""Atmospheric Radiative Transfer (art) class

    Notes:
        opacity is computed in art because it uses planet physical quantities 
        such as gravity, mmr.

"""
import numpy as np
from exojax.spec.planck import piBarr
from exojax.spec.rtransfer import rtrun as rtrun_emis_pure_absorption
from exojax.spec.rtransfer import dtauM
#from exojax.spec.rtransfer import dtauCIA
from exojax.atm.atmprof import atmprof_gray, atmprof_Guillot, atmprof_powerlow
import jax.numpy as jnp
from exojax.atm.idealgas import number_density
from exojax.utils.constants import logkB, logm_ucgs


class ArtCommon():
    """Common Atmospheric Radiative Transfer
    """
    def __init__(self, nu_grid, pressure_top, pressure_btm, nlayer):
        """initialization of art

        Args:
            nu_grid (nd.array): wavenumber grid in cm-1
            pressure_top (float):top pressure in bar
            pressure_bottom (float): bottom pressure in bar
            nlayer (int): # of atmospheric layers
        """
        self.artinfo = None
        self.method = None  # which art is used
        self.ready = False  # ready for art computation
        self.Tlow = 0.0
        self.Thigh = jnp.inf

        self.nu_grid = nu_grid
        self.pressure_top = pressure_top
        self.pressure_btm = pressure_btm
        self.nlayer = nlayer
        self.check_pressure()
        self.log_pressure_btm = np.log10(self.pressure_btm)
        self.log_pressure_top = np.log10(self.pressure_top)
        self.init_pressure_profile()

        self.fguillot = 0.25

    def opacity_profile_lines(self, xsmatrix, mmr_profile, molmass, gravity):
        """opacity profile (delta tau) for lines

        Args:
            xsmatrix (2D array): cross section matrix (Nlayer, N_wavenumber)
            mmr_profile (1D array): mass mixing ratio, Nlayer, (or volume mixing ratio profile)
            molmass (float): molecular mass (or mean molecular weight)
            gravity (_type_): constant or 1d profile of gravity in cgs

        Returns:
            dtau: opacity profile, whose element is optical depth in each layer. 
        """
        return dtauM(self.dParr, jnp.abs(xsmatrix), mmr_profile, molmass,
                     gravity)

    def opacity_profile_cia(self, logacia_matrix, temperature, vmr1, vmr2, mmw,
                            gravity):

        narr = number_density(self.pressure, temperature)
        lognarr1 = jnp.log10(vmr1 * narr)  # log number density
        lognarr2 = jnp.log10(vmr2 * narr)  # log number density
        logg = jnp.log10(gravity)
        ddParr = self.dParr / self.pressure
        return 10**(logacia_matrix + lognarr1[:, None] + lognarr2[:, None] +
                    logkB - logg -
                    logm_ucgs) * temperature[:, None] / mmw * ddParr[:, None]

    def check_pressure(self):
        if self.pressure_btm < self.pressure_top:
            raise ValueError(
                "Pressure at bottom should be higher than that at top atmosphere."
            )
        if type(self.nlayer) is not int:
            raise ValueError("Number of the layer should be integer")

    def init_pressure_profile(self):
        from exojax.spec.rtransfer import pressure_layer
        self.pressure, self.dParr, self.k = pressure_layer(
            logPtop=self.log_pressure_top,
            logPbtm=self.log_pressure_btm,
            NP=self.nlayer,
            mode='ascending',
            reference_point=0.5,
            numpy=True)

    def constant_mmr_profile(self, value):
        return value * np.ones_like(self.pressure)

    def constant_gravity_profile(self, value):
        return value * np.array([np.ones_like(self.pressure)]).T

    def change_temperature_range(self, Tlow, Thigh):
        """temperature range to be assumed.

        Note:
            The default temperature range is self.Tlow = 0 K, self.Thigh = jnp.inf.

        Args:
            Tlow (float): lower temperature
            Thigh (float): higher temperature
        """
        self.Tlow = Tlow
        self.Thigh = Thigh

    def clip_temperature(self, temperature):
        """temperature clipping

        Args:
            temperature (array): temperature profile

        Returns:
            array: temperature profile clipped in the range of (self.Tlow-self.Thigh)
        """
        return jnp.clip(temperature, self.Tlow, self.Thigh)

    def powerlaw_temperature(self, T0, alpha):
        """powerlaw temperature profile

        Args:
            T0 (float): T at P=1 bar in K
            alpha (float): powerlaw index

        Returns:
            array: temperature profile
        """
        return self.clip_temperature(atmprof_powerlow(self.pressure, T0,
                                                      alpha))

    def gray_temperature(self, gravity, kappa, Tint):
        """ gray temperature profile

        Args:
            gravity: gravity (cm/s2)
            kappa: infrared opacity 
            Tint: temperature equivalence of the intrinsic energy flow in K

        Returns:
            array: temperature profile

        """
        return self.clip_temperature(
            atmprof_gray(self.pressure, gravity, kappa, Tint))

    def guillot_temeprature(self, gravity, kappa, gamma, Tint, Tirr):
        """ Guillot tempearture profile

        Notes:  
            Set self.fguillot (default 0.25) to change the assumption of irradiation.
            self.fguillot = 1. at the substellar point, self.fguillot = 0.5 for a day-side average 
            and self.fguillot = 0.25 for an averaging over the whole planetary surface
            See Guillot (2010) Equation (29) for details.

        Args:
            gravity: gravity (cm/s2)
            kappa: thermal/IR opacity (kappa_th in Guillot 2010)
            gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
            Tint: temperature equivalence of the intrinsic energy flow in K
            Tirr: temperature equivalence of the irradiation in K
            
        Returns:
            array: temperature profile

        """
        return self.clip_temperature(
            atmprof_Guillot(self.pressure, gravity, kappa, gamma, Tint, Tirr,
                            self.fguillot))


class ArtEmisPure(ArtCommon):
    """Atmospheric RT for emission w/ pure absorption

    Attributes:
        pressure_layer: pressure profile in bar
        
    """
    def __init__(self,
                 nu_grid,
                 pressure_top=1.e-8,
                 pressure_btm=1.e2,
                 nlayer=100):
        """initialization of ArtEmisPure

        
        """
        super().__init__(nu_grid, pressure_top, pressure_btm, nlayer)
        self.method = "emission_with_pure_absorption"

    def run(self, dtau, temperature):
        sourcef = piBarr(temperature, self.nu_grid)
        return rtrun_emis_pure_absorption(dtau, sourcef)
