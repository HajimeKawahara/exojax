"""Atmospheric Radiative Transfer (art) class

    Notes:
        The opacity is computed in art because it uses planet physical quantities 
        such as gravity, mmr. "run" method computes a spectrum. 

"""

import numpy as np
import jax.numpy as jnp
from exojax.spec.planck import piBarr
from exojax.spec.rtransfer import rtrun_emis_pureabs_ibased_linsap
from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st
from exojax.spec.rtransfer import rtrun_emis_pureabs_ibased
from exojax.spec.rtransfer import rtrun_emis_scat_lart_toonhm
from exojax.spec.rtransfer import rtrun_emis_scat_fluxadding_toonhm
from exojax.spec.rtransfer import rtrun_reflect_fluxadding_toonhm
from exojax.spec.rtransfer import rtrun_trans_pureabs_trapezoid
from exojax.spec.rtransfer import rtrun_trans_pureabs_simpson
from exojax.spec.layeropacity import layer_optical_depth
from exojax.spec.layeropacity import layer_optical_depth_clouds_lognormal
from exojax.atm.atmprof import atmprof_gray, atmprof_Guillot, atmprof_powerlow
from exojax.atm.idealgas import number_density
from exojax.atm.atmprof import normalized_layer_height
from exojax.spec.opachord import chord_geometric_matrix_lower
from exojax.spec.opachord import chord_geometric_matrix
from exojax.spec.opachord import chord_optical_depth
from exojax.utils.constants import logkB, logm_ucgs
from exojax.utils.indexing import get_smooth_index
import warnings


class ArtCommon:
    """Common Atmospheric Radiative Transfer"""

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
        self.reference_point = 0.5  # ref point (r) for pressure layers

        if nu_grid is None:
            warnings.warn(
                "nu_grid is not given. specify nu_grid when using 'run' ", UserWarning
            )
        self.nu_grid = nu_grid

        self.pressure_top = pressure_top
        self.pressure_btm = pressure_btm
        self.nlayer = nlayer
        self.check_pressure()
        self.log_pressure_btm = np.log10(self.pressure_btm)
        self.log_pressure_top = np.log10(self.pressure_top)
        self.init_pressure_profile()

        self.fguillot = 0.25

    def atmosphere_height(
        self, temperature, mean_molecular_weight, radius_btm, gravity_btm
    ):
        """atmosphere height and radius

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array):
                mean molecular weight profile (float/Nlayer)
            radius_btm (float):
                the bottom radius of the atmospheric layer
            gravity_btm (float): the bottom gravity cm2/s at radius_btm, i.e. G M_p/radius_btm

        Returns:
            1D array: height normalized by radius_btm (Nlayer)
            1D array: layer radius r_n normalized by radius_btm (Nlayer)

        Notes:
            Our definitions of the radius_lower, radius_layer, and height are as follows:
            n=0,1,...,N-1
            radius_lower[N-1] = radius_btm (i.e. R0)
            radius_lower[n-1] = radius_lower[n] + height[n]
            "normalized" means physical length divided by radius_btm


        """
        normalized_height, normalized_radius_lower = normalized_layer_height(
            temperature,
            self.pressure_decrease_rate,
            mean_molecular_weight,
            radius_btm,
            gravity_btm,
        )
        return normalized_height, normalized_radius_lower

    def constant_gravity_profile(self, value):
        return value * np.array([np.ones_like(self.pressure)]).T

    def gravity_profile(
        self, temperature, mean_molecular_weight, radius_btm, gravity_btm
    ):
        """gravity layer profile assuming hydrostatic equilibrium

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array):
                mean molecular weight profile (float/Nlayer)
            radius_btm (float): the bottom radius of the atmospheric layer
            gravity_btm (float):
                the bottom gravity cm2/s at radius_btm, i.e. G M_p/radius_btm

        Returns:
            2D array:
                gravity in cm2/s (Nlayer, 1), suitable for the input of opacity_profile_lines
        """
        normalized_height, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm
        )
        normalized_radius_layer = normalized_radius_lower + 0.5 * normalized_height
        return jnp.array([gravity_btm / normalized_radius_layer]).T

    def constant_mmr_profile(self, value):
        return value * np.ones_like(self.pressure)

    def opacity_profile_lines(self, xs, mixing_ratio, molmass, gravity):
        raise ValueError(
            "opacity_profile_lines was removed. Use opacity_profile_xs instead"
        )

    def opacity_profile_xs(self, xs, mixing_ratio, molmass, gravity):
        """opacity profile (delta tau) from cross section matrix or vector, molecular line/Rayleigh scattering

        Args:
            xs (2D array/1D array): cross section matrix i.e. xsmatrix (Nlayer, N_wavenumber) or vector i.e. xsvector (N_wavenumber)
            mixing_ratio (1D array): mass mixing ratio, Nlayer, (or volume mixing ratio profile)
            molmass (float): molecular mass (or mean molecular weight)
            gravity (float/1D profile): constant or 1d profile of gravity in cgs

        Returns:
            dtau: opacity profile, whose element is optical depth in each layer.
        """
        return layer_optical_depth(
            self.dParr, jnp.abs(xs), mixing_ratio, molmass, gravity
        )

    def opacity_profile_cloud_lognormal(
        self,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
    ):
        """
        opacity profile (delta tau) from extinction coefficient assuming the AM cloud model with a lognormal cloud distribution
        Args:
            extinction coefficient: extinction coefficient  in cgs (cm-1) [N_layer, N_nus]
            condensate_substance_density: condensate substance density (g/cm3)
            mmr_condensate: Mass mixing ratio (array) of condensate [Nlayer]
            rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag:sigmag parameter (geometric standard deviation) in the lognormal distribution of condensate size, defined by (9) in AM01, must be sigmag > 1
            gravity: gravity (cm/s2)

        Returns:
            2D array: optical depth matrix, dtau  [N_layer, N_nus]

        """

        return layer_optical_depth_clouds_lognormal(
            self.dParr,
            extinction_coefficient,
            condensate_substance_density,
            mmr_condensate,
            rg,
            sigmag,
            gravity,
        )

    def opacity_profile_cia(
        self, logacia_matrix, temperature, vmr1, vmr2, mmw, gravity
    ):
        """opacity profile (delta tau) from collision-induced absorption

        Args:
            logacia_matrix (_type_): _description_
            temperature (_type_): _description_
            vmr1 (_type_): _description_
            vmr2 (_type_): _description_
            mmw (_type_): _description_
            gravity (_type_): _description_

        Returns:
            _type_: _description_
        """
        narr = number_density(self.pressure, temperature)
        lognarr1 = jnp.log10(vmr1 * narr)  # log number density
        lognarr2 = jnp.log10(vmr2 * narr)  # log number density
        logg = jnp.log10(gravity)
        ddParr = self.dParr / self.pressure
        return (
            10
            ** (
                logacia_matrix
                + lognarr1[:, None]
                + lognarr2[:, None]
                + logkB
                - logg
                - logm_ucgs
            )
            * temperature[:, None]
            / mmw
            * ddParr[:, None]
        )

    def check_pressure(self):
        if self.pressure_btm < self.pressure_top:
            raise ValueError(
                "Pressure at bottom should be higher than that at top atmosphere."
            )
        if type(self.nlayer) is not int:
            raise ValueError("Number of the layer should be integer")

    def init_pressure_profile(self):
        from exojax.atm.atmprof import pressure_layer_logspace
        from exojax.atm.atmprof import pressure_boundary_logspace

        (
            self.pressure,
            self.dParr,
            self.pressure_decrease_rate,
        ) = pressure_layer_logspace(
            log_pressure_top=self.log_pressure_top,
            log_pressure_btm=self.log_pressure_btm,
            nlayer=self.nlayer,
            mode="ascending",
            reference_point=self.reference_point,
            numpy=True,
        )
        self.pressure_boundary = pressure_boundary_logspace(
            self.pressure,
            self.pressure_decrease_rate,
            reference_point=self.reference_point,
        )

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
        return self.clip_temperature(atmprof_powerlow(self.pressure, T0, alpha))

    def gray_temperature(self, gravity, kappa, Tint):
        """gray temperature profile

        Args:
            gravity: gravity (cm/s2)
            kappa: infrared opacity
            Tint: temperature equivalence of the intrinsic energy flow in K

        Returns:
            array: temperature profile

        """
        return self.clip_temperature(atmprof_gray(self.pressure, gravity, kappa, Tint))

    def guillot_temperature(self, gravity, kappa, gamma, Tint, Tirr):
        """Guillot tempearture profile

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
            atmprof_Guillot(
                self.pressure, gravity, kappa, gamma, Tint, Tirr, self.fguillot
            )
        )

    def custom_temperature(self, np_temperature):
        """custom temperature profile from numpy ndarray

        Notes: this function is equivalen to jnp.array(np_temperature), but it is necessary for the compatibility.

        Args:
            np_temperature (numpy nd array): temperature profile

        Returns:
            array: jnp.array temperature profile
        """
        return jnp.array(np_temperature)

    def powerlaw_temperature_boundary(self, T0, alpha):
        """powerlaw temperature at the upper point (overline{T}) + TB profile

        Args:
            T0 (float): T at P=1 bar in K
            alpha (float): powerlaw index

        Returns:
            array: layer boundary temperature profile (Nlayer + 1)
        """
        return self.clip_temperature(
            atmprof_powerlow(self.pressure_boundary, T0, alpha)
        )


class ArtAbsPure(ArtCommon):
    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
    ):
        """initialization of ArtAbsPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)

    def run(self, dtau, pressure_surface, incoming_flux, mu_in, mu_out):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            pressure_surface: pressure at the surface (bar)
            incoming flux: incoming flux (x reflectivity) (N_nus)
            mu_in (float>0) : cosine of the viewing angle for incoming ray (>0.0)
            mu_out (float>0 or None): cosine of the viewing angle for outgoing ray, (>0.0)
                    when mu_out is given (is not None), the observer is located at the top of the atmosphere
                    if None, the observe is located at the ground.

        Notes:
            We include the reflectivity by surface in "incoming flux" for the simplicity
            when the obserber is located at the top of atmosphere (mu_out is not None).

        Returns:
            1D array: spectrum
        """
        factor = 1.0 / mu_in
        if mu_out is not None:
            factor = factor + 1.0 / mu_out

        logk = jnp.log10(self.pressure_decrease_rate)
        logp_btm = jnp.log10(self.pressure) + (self.reference_point - 1.0) * logk
        logp_surface = jnp.log10(pressure_surface)
        smooth_index = get_smooth_index(logp_btm, logp_surface)
        ind = smooth_index.astype(int)
        res = smooth_index - jnp.floor(smooth_index)
        stepfunc = jnp.heaviside(logp_surface - logp_btm, 0.5)
        tau_opaque = (
            jnp.sum(dtau * stepfunc[:, jnp.newaxis], axis=0) + dtau[ind, :] * res
        )
        trans = jnp.exp(-factor * tau_opaque)

        return trans * incoming_flux


class ArtReflectPure(ArtCommon):
    """Atmospheric RT for Pure Reflected light (no source term)

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="fluxadding_toon_hemispheric_mean",
    ):
        """initialization of ArtReflectPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str): Radiative Transfer Solver, fluxadding_toon_hemispheric_mean


        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.rtsolver = rtsolver
        self.method = "reflection_using_" + self.rtsolver

    def run(
        self,
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        reflectivity_surface,
        incoming_flux,
    ):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            single_scattering_albedo: single scattering albedo (Nlayer, N_nus)
            asymmetric_parameter: assymetric parameter (Nlayer, N_nus)
            reflectivity_surface: reflectivity from the surface (N_nus)
            incoming flux: incoming flux F_0^- (N_nus)


        Returns:
            1D array: spectrum
        """

        if self.rtsolver == "fluxadding_toon_hemispheric_mean":
            _, Nnus = dtau.shape
            sourcef = jnp.zeros_like(dtau)
            source_surface = jnp.zeros(Nnus)
            return rtrun_reflect_fluxadding_toonhm(
                dtau,
                single_scattering_albedo,
                asymmetric_parameter,
                sourcef,
                source_surface,
                reflectivity_surface,
                incoming_flux,
            )
        else:
            print("rtsolver=", self.rtsolver)
            raise ValueError("Unknown radiative transfer solver (rtsolver).")


class ArtReflectEmis(ArtCommon):
    """Atmospheric RT for Reflected light with Source Term

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="fluxadding_toon_hemispheric_mean",
    ):
        """initialization of ArtReflectionPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str): Radiative Transfer Solver, fluxadding_toon_hemispheric_mean


        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.rtsolver = rtsolver
        self.method = "reflection_using_" + self.rtsolver

    def run(
        self,
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        source_surface,
        reflectivity_surface,
        incoming_flux,
        nu_grid=None,
    ):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            single_scattering_albedo: single scattering albedo (Nlayer, N_nus)
            asymmetric_parameter: assymetric parameter (Nlayer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            source_surface: source from the surface (N_nus)
            reflectivity_surface: reflectivity from the surface (N_nus)
            incoming flux: incoming flux F_0^- (N_nus)
            nu_grid (1D array): if nu_grid is not initialized, provide it.


        Returns:
            1D array: spectrum
        """
        if self.nu_grid is not None:
            sourcef = piBarr(temperature, self.nu_grid)
        elif nu_grid is not None:
            sourcef = piBarr(temperature, nu_grid)
        else:
            raise ValueError("the wavenumber grid is not given.")

        if self.rtsolver == "fluxadding_toon_hemispheric_mean":
            return rtrun_reflect_fluxadding_toonhm(
                dtau,
                single_scattering_albedo,
                asymmetric_parameter,
                sourcef,
                source_surface,
                reflectivity_surface,
                incoming_flux,
            )
        else:
            print("rtsolver=", self.rtsolver)
            raise ValueError("Unknown radiative transfer solver (rtsolver).")


class ArtEmisScat(ArtCommon):
    """Atmospheric RT for emission w/ scattering

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="fluxadding_toon_hemispheric_mean",
    ):
        """initialization of ArtEmisScat

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str): Radiative Transfer Solver,
                "fluxadding_toon_hemispheric_mean" (default),
                "lart_toon_hemispheric_mean"

        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.rtsolver = rtsolver
        self.method = "emission_with_scattering_using_" + self.rtsolver

    def run(
        self,
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        nu_grid=None,
        show=False,
    ):
        """run radiative transfer

        Args:
            dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
            temperature (1D array): temperature profile (Nlayer)
            nu_grid (1D array): if nu_grid is not initialized, provide it.
            show: plot intermediate results

        Returns:
            1D array: spectrum
        """
        if self.nu_grid is not None:
            sourcef = piBarr(temperature, self.nu_grid)
        elif nu_grid is not None:
            sourcef = piBarr(temperature, nu_grid)
        else:
            raise ValueError("the wavenumber grid is not given.")

        if self.rtsolver == "lart_toon_hemispheric_mean":
            (
                spectrum,
                cumTtilde,
                Qtilde,
                trans_coeff,
                scat_coeff,
                piB,
            ) = rtrun_emis_scat_lart_toonhm(
                dtau, single_scattering_albedo, asymmetric_parameter, sourcef
            )
            if show:
                from exojax.plot.rtplot import comparison_with_pure_absorption

                spec, spec_pure = comparison_with_pure_absorption(
                    cumTtilde, Qtilde, spectrum, trans_coeff, scat_coeff, piB
                )
                return spectrum, spec, spec_pure

        elif self.rtsolver == "fluxadding_toon_hemispheric_mean":
            spectrum = rtrun_emis_scat_fluxadding_toonhm(
                dtau, single_scattering_albedo, asymmetric_parameter, sourcef
            )

        else:
            print("rtsolver=", self.rtsolver)
            raise ValueError("Unknown radiative transfer solver (rtsolver).")

        return spectrum


class ArtEmisPure(ArtCommon):
    """Atmospheric RT for emission w/ pure absorption

    Notes:
        The default radiative transfer scheme has been the intensity-based transfer since version 1.5

    Attributes:
        pressure_layer: pressure profile in bar

    """

    def __init__(
        self,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nu_grid=None,
        rtsolver="ibased",
        nstream=8,
    ):
        """
        initialization of ArtEmisPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            rtsolver (str, optional): radiative transfer solver (ibased, fbased2st, ibased_linsap). Defaults to "ibased".
            nstream (int, optional): the number of stream. Defaults to 8. Should be 2 for rtsolver = fbased2st
        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid)
        self.method = "emission_with_pure_absorption"
        self.set_capable_rtsolvers()
        self.validate_rtsolver(rtsolver, nstream)

    def set_capable_rtsolvers(self):
        self.rtsolver_dict = {
            "fbased2st": rtrun_emis_pureabs_fbased2st,
            "ibased": rtrun_emis_pureabs_ibased,
            "ibased_linsap": rtrun_emis_pureabs_ibased_linsap,
        }

        self.valid_rtsolvers = list(self.rtsolver_dict.keys())

        # source function to be used in rtsolver
        self.source_position_dict = {
            "fbased2st": "representative",
            "ibased": "representative",
            "ibased_linsap": "upper_boundary",
        }

        self.rtsolver_explanation = {
            "fbased2st": "Flux-based two-stream solver, isothermal layer (ExoJAX1, HELIOS-R1 like)",
            "ibased": "Intensity-based n-stream solver, isothermal layer (e.g. NEMESIS, pRT like)",
            "ibased_linsap": "Intensity-based n-stream solver w/ linear source approximation (linsap), see Olson and Kunasz (e.g. HELIOS-R2 like)",
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
            str_valid_rtsolvers = (
                ", ".join(self.valid_rtsolvers[:-1])
                + f", or {self.valid_rtsolvers[-1]}"
            )
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
            1D array: emission spectrum
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
    """Atmospheric Radiative Transfer for transmission spectroscopy

    Args:
        ArtCommon: ArtCommon class
    """

    def __init__(
        self, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, integration="simpson"
    ):
        """initialization of ArtTransPure

        Args:
            pressure_top (float, optional): layer top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): layer bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): The number of the layers Defaults to 100.
            integration (str, optional): Integration scheme ("simpson", "trapezoid"). Defaults to "simpson".

        Note:
            The users can choose the integration scheme of the chord integration from Trapezoid method or Simpson method.

        """
        super().__init__(pressure_top, pressure_btm, nlayer, nu_grid=None)
        self.method = "transmission_with_pure_absorption"
        self.set_capable_integration()
        self.set_integration_scheme(integration)

    def set_capable_integration(self):
        """sets integration scheme directory"""
        self.integration_dict = {
            "trapezoid": rtrun_trans_pureabs_trapezoid,
            "simpson": rtrun_trans_pureabs_simpson,
        }

        self.valid_integration = list(self.integration_dict.keys())

        self.integration_explanation = {
            "trapezoid": "Trapezoid integration, uses the chord optical depth at the lower boundary of the layers only",
            "simpson": "Simpson integration, uses the chord optical depth at the lower boundary and midppoint of the layers.",
        }

    def set_integration_scheme(self, integration):
        """sets and validates integration

        Args:
            integration (str): integration scheme, i.e. "trapezoid" or "simpson"

        """

        if integration in self.valid_integration:
            self.integration = integration
            print("integration: ", self.integration)
            print(self.integration_explanation[self.integration])
        else:
            str_valid_integration = (
                ", ".join(self.valid_integration[:-1])
                + f", or {self.valid_integration[-1]}"
            )
            raise ValueError(
                "Unknown integration (scheme). Use " + str_valid_integration
            )

    def run(self, dtau, temperature, mean_molecular_weight, radius_btm, gravity_btm):
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

        normalized_height, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm
        )
        normalized_radius_top = normalized_radius_lower[0] + normalized_height[0]
        cgm = chord_geometric_matrix_lower(normalized_height, normalized_radius_lower)
        dtau_chord_lower = chord_optical_depth(cgm, dtau)
        func = self.integration_dict[self.integration]

        if self.integration == "trapezoid":
            return func(
                dtau_chord_lower, normalized_radius_lower, normalized_radius_top
            )
        elif self.integration == "simpson":
            cgm_midpoint = chord_geometric_matrix(
                normalized_height, normalized_radius_lower
            )
            dtau_chord_midpoint = chord_optical_depth(cgm_midpoint, dtau)
            return func(
                dtau_chord_midpoint,
                dtau_chord_lower,
                normalized_radius_lower,
                normalized_height,
            )
