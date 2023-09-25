"""Atmospheric Radiative Transfer (art) class

    Notes:
        opacity is computed in art because it uses planet physical quantities 
        such as gravity, mmr.

"""
import numpy as np
import jax.numpy as jnp
from exojax.spec.planck import piBarr
from exojax.spec.rtransfer import rtrun_emis_pureabs_ibased_linsap
from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st
from exojax.spec.rtransfer import rtrun_emis_pureabs_ibased
from exojax.spec.rtransfer import rtrun_emis_scat_lart_toonhm
from exojax.spec.rtransfer import rtrun_trans_pureabs
from exojax.spec.layeropacity import layer_optical_depth
from exojax.atm.atmprof import atmprof_gray, atmprof_Guillot, atmprof_powerlow
from exojax.atm.idealgas import number_density
from exojax.atm.atmprof import normalized_layer_height
from exojax.spec.opachord import chord_geometric_matrix
from exojax.spec.opachord import chord_optical_depth
from exojax.utils.constants import logkB, logm_ucgs
import warnings


class ArtCommon():
    """Common Atmospheric Radiative Transfer
    """
    def __init__(self, pressure_top, pressure_btm, nlayer, nu_grid=None):
        """initialization of art

        Args:
            pressure_top (float):top pressure in bar
            pressure_bottom (float): bottom pressure in bar
            nlayer (int): # of atmospheric layers
            nu_grid (nd.array, optional): wavenumber grid in cm-1
        """
        self.artinfo = None
        self.method = None  # which art is used
        self.ready = False  # ready for art computation
        self.Tlow = 0.0
        self.Thigh = jnp.inf

        if nu_grid is None:
            warnings.warn(
                "nu_grid is not given. specify nu_grid when using 'run' ",
                UserWarning)
        self.nu_grid = nu_grid

        self.pressure_top = pressure_top
        self.pressure_btm = pressure_btm
        self.nlayer = nlayer
        self.check_pressure()
        self.log_pressure_btm = np.log10(self.pressure_btm)
        self.log_pressure_top = np.log10(self.pressure_top)
        self.init_pressure_profile()

        self.fguillot = 0.25

    def atmosphere_height(self, temperature, mean_molecular_weight, radius_btm,
                          gravity_btm):
        """atmosphere height and radius

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array): mean molecular weight profile (float/Nlayer)
            radius_btm (float): the bottom radius of the atmospheric layer
            gravity_btm (float): the bottom gravity cm2/s at radius_btm, i.e. G M_p/radius_btm

        Returns:
            1D array: height normalized by radius_btm (Nlayer)
            1D array: layer radius r_n normalized by radius_btm (Nlayer)
            1D array: radius at lower boundary normalized by radius_btm (Nlayer)

        Notes:
            Our definitions of the radius_lower, radius_layer, and height are as follows:
            n=0,1,...,N-1
            radius_lower[N-1] = radius_btm (i.e. R0)
            radius_lower[n-1] = radius_lower[n] + height[n]
            radius_layer[n] =  radius_lower[n] + height[n]/2
            "normalized" means physical length divided by radius_btm


        """
        print("k=", self.k)
        normalized_height, normalized_radius_lower = normalized_layer_height(
            temperature, self.k, mean_molecular_weight, radius_btm,
            gravity_btm)
        normalized_radius_layer = normalized_radius_lower + 0.5 * normalized_height
        return normalized_height, normalized_radius_layer, normalized_radius_lower

    def constant_gravity_profile(self, value):
        return value * np.array([np.ones_like(self.pressure)]).T

    def gravity_profile(self, temperature, mean_molecular_weight, radius_btm,
                        gravity_btm):
        """gravity layer profile assuming hydrostatic equilibrium

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array): mean molecular weight profile (float/Nlayer)
            radius_btm (float): the bottom radius of the atmospheric layer
            gravity_btm (float): the bottom gravity cm2/s at radius_btm, i.e. G M_p/radius_btm

        Returns:
            2D array: gravity in cm2/s (Nlayer, 1), suitable for the input of opacity_profile_lines
        """
        _, normalized_radius_layer, _ = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm)
        return jnp.array([gravity_btm / normalized_radius_layer]).T

    def constant_mmr_profile(self, value):
        return value * np.ones_like(self.pressure)

    def opacity_profile_lines(self, xsmatrix, mixing_ratio, molmass, gravity):
        """opacity profile (delta tau) for lines

        Args:
            xsmatrix (2D array): cross section matrix (Nlayer, N_wavenumber)
            mixing_ratio (1D array): mass mixing ratio, Nlayer, (or volume mixing ratio profile)
            molmass (float): molecular mass (or mean molecular weight)
            gravity (float/1D profile): constant or 1d profile of gravity in cgs

        Returns:
            dtau: opacity profile, whose element is optical depth in each layer. 
        """
        return layer_optical_depth(self.dParr, jnp.abs(xsmatrix), mixing_ratio,
                                   molmass, gravity)

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
        from exojax.atm.atmprof import pressure_layer_logspace
        from exojax.atm.atmprof import pressure_upper_logspace

        self.pressure, self.dParr, self.k = pressure_layer_logspace(
            log_pressure_top=self.log_pressure_top,
            log_pressure_btm=self.log_pressure_btm,
            nlayer=self.nlayer,
            mode='ascending',
            reference_point=0.5,
            numpy=True)
        self.pressure_upper =  pressure_upper_logspace(self.pressure,self.k,reference_point=0.5)

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

    def guillot_temperature(self, gravity, kappa, gamma, Tint, Tirr):
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

    def powerlaw_temperature_upper(self, T0, alpha):
        """powerlaw temperature at the upper point (overline{T}) profile

        Args:
            T0 (float): T at P=1 bar in K
            alpha (float): powerlaw index

        Returns:
            array: temperature profile
        """
        return self.clip_temperature(atmprof_powerlow(self.pressure_upper, T0,
                                                      alpha))

    def gray_temperature_upper(self, gravity, kappa, Tint):
        """ gray temperature at the upper point (overline{T}) profile 

        Args:
            gravity: gravity (cm/s2)
            kappa: infrared opacity 
            Tint: temperature equivalence of the intrinsic energy flow in K

        Returns:
            array: temperature profile

        """
        return self.clip_temperature(
            atmprof_gray(self.pressure_upper, gravity, kappa, Tint))

    def guillot_temperature_upper(self, gravity, kappa, gamma, Tint, Tirr):
        """ Guillot tempearture at the upper point (overline{T}) profile

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
            atmprof_Guillot(self.pressure_upper, gravity, kappa, gamma, Tint, Tirr,
                            self.fguillot))



class ArtEmisScat(ArtCommon):
    """Atmospheric RT for emission w/ scattering

    Attributes:
        pressure_layer: pressure profile in bar
        
    """
    def __init__(self,
                 pressure_top=1.e-8,
                 pressure_btm=1.e2,
                 nlayer=100,
                 nu_grid=None,
                 rtsolver="toon_hemispheric_mean"):
        """initialization of ArtEmisPure

        Args:
            rtsolver (str): Radiative Transfer Solver, toon_hemispheric_mean, SH1, SH3 


        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.rtsolver = rtsolver
        self.method = "emission_with_scattering_using_" + self.rtsolver

    def run(self,
            dtau,
            single_scattering_albedo,
            asymmetric_parameter,
            temperature,
            nu_grid=None,
            show=False):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            nu_grid (1D array): if nu_grid is not initialized, provide it. 
            show: plot intermediate results

        Returns:
            _type_: _description_
        """
        if self.nu_grid is not None:
            sourcef = piBarr(temperature, self.nu_grid)
        elif nu_grid is not None:
            sourcef = piBarr(temperature, nu_grid)
        else:
            raise ValueError("the wavenumber grid is not given.")

        if self.rtsolver == "toon_hemispheric_mean":

            spectrum, cumTtilde, Qtilde, trans_coeff, scat_coeff, piB = rtrun_emis_scat_lart_toonhm(
                dtau, single_scattering_albedo, asymmetric_parameter, sourcef)
            if show:
                from exojax.plot.rtplot import comparison_with_pure_absorption
                comparison_with_pure_absorption(cumTtilde, Qtilde, spectrum,
                                                trans_coeff, scat_coeff, piB)

            return spectrum
        else:
            raise ValueError("Unknown radiative transfer solver (rtsolver).")


class ArtEmisPure(ArtCommon):
    """Atmospheric RT for emission w/ pure absorption

    Attributes:
        pressure_layer: pressure profile in bar
        
    """
    def __init__(
        self,
        pressure_top=1.e-8,
        pressure_btm=1.e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="fbased2st",
        nstream=2,
    ):
        """initialization of ArtEmisPure

        
        """

        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.method = "emission_with_pure_absorption"
        self.set_capable_rtsolvers()
        self.validate_rtsolver(rtsolver, nstream)

        
    def set_capable_rtsolvers(self):
        self.rtsolver_dict = {
            "fbased2st": rtrun_emis_pureabs_fbased2st,
            "ibased": rtrun_emis_pureabs_ibased,
            "ibased_linsap": rtrun_emis_pureabs_ibased_linsap
        }

        self.valid_rtsolvers = list(self.rtsolver_dict.keys())

        # source function to be used in rtsolver
        self.source_position_dict = {
            "fbased2st": "representative",
            "ibased": "representative",
            "ibased_linsap": "upper_boundary"
        }
        
        self.rtsolver_explanation = {
            "fbased2st":
            "Flux-based two-stream solver, isothermal layer (ExoJAX1, HELIOS-R1 like)",
            "ibased":
            "Intensity-based n-stream solver, isothermal layer (e.g. NEMESIS, pRT like)",
            "ibased_linsap":
            "Intensity-based n-stream solver w/ linear source approximation (linsap), see Olson and Kunasz (e.g. HELIOS-R2 like)"
        }
        
    def validate_rtsolver(self, rtsolver, nstream):
        """validates rtsolver

        Args:
            rtsolver (str): rtsolver
            nstream (int): the number of streams

        """

        if rtsolver in self.valid_rtsolvers:
            self.rtsolver = rtsolver
            self.source_position = self.source_position_dict[self.rtsolver]
            print("rtsolver: ", self.rtsolver)
            print(self.rtsolver_explanation[self.rtsolver]) 
        else:
            str_valid_rtsolvers = ", ".join(
                self.valid_rtsolvers[:-1]) + f", or {self.valid_rtsolvers[-1]}"
            raise ValueError("Unknown rtsolver. Use " + str_valid_rtsolvers)
        if rtsolver == "fbased2st" and nstream != 2:
            raise ValueError(
                "fbased2st (flux-based two-stream) rtsolver requires nstream = 2."
            )
        self.nstream = nstream

    def run(self, dtau, temperature, nu_grid=None):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            nu_grid (1D array): if nu_grid is not initialized, provide it. 

        Returns:
            _type_: _description_
        """
        if self.nu_grid is not None:
            nu_grid = self.nu_grid

        sourcef = piBarr(temperature, nu_grid)
        rtfunc = self.rtsolver_dict[self.rtsolver]
        
        if self.rtsolver == "fbased2st":
            return rtfunc(dtau, sourcef)
        elif self.rtsolver == "ibased" or self.rtsolver == "ibased_linsap":
            from exojax.spec.rtransfer import initialize_gaussian_quadrature
            mus, weights = initialize_gaussian_quadrature(self.nstream)
            return rtfunc(dtau, sourcef, mus, weights)
        
class ArtTransPure(ArtCommon):
    def __init__(self, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100):
        """initialization of ArtTransPure

        
        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid=None)
        self.method = "transmission_with_pure_absorption"

    def run(self, dtau, temperature, mean_molecular_weight, radius_btm,
            gravity_btm):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            mean_molecular_weight (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom) 
            radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
            gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

        Returns:
            1D array: transit squared radius normalized by radius_btm**2, i.e. it returns (radius/radius_btm)**2

        Notes:
            This function gives the sqaure of the transit radius.
            If you would like to obtain the transit radius, take sqaure root of the output and multiply radius_btm.
            If you would like to compute the transit depth, divide the output by (stellar radius/radius_btm)**2

        """

        normalized_height, _, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm)
        cgm = chord_geometric_matrix(normalized_height,
                                     normalized_radius_lower)
        tauchord = chord_optical_depth(cgm, dtau)
        return rtrun_trans_pureabs(tauchord, normalized_radius_lower)
