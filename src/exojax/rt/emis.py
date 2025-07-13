from exojax.rt.common import ArtCommon
from exojax.rt.planck import piB, piBarr
from exojax.rt.rtransfer import (
    rtrun_emis_pureabs_fbased2st,
    rtrun_emis_pureabs_ibased,
    rtrun_emis_pureabs_ibased_linsap,
    rtrun_emis_scat_fluxadding_toonhm,
    rtrun_emis_scat_lart_toonhm,
    initialize_gaussian_quadrature,
    setrt_toonhm,
)
from exojax.rt.rtlayer import fluxsum_scan
from exojax.rt.common import ArtCommon

import jax.numpy as jnp
from jax.lax import scan


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
        self.mus, self.weights = initialize_gaussian_quadrature(self.nstream)

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
            return rtfunc(dtau, sourcef, self.mus, self.weights)

    def run_ckd(self, dtau_ckd, temperature, weights, nu_bands):
        """run radiative transfer for CKD

        Args:
            dtau_ckd (3D array): optical depth matrix, dtau  (N_layer, Ng, Nbands)
            temperature (1D array): temperature profile (Nlayer,)
            weights (1D array): weights for the Gaussian quadrature (Ng,)
            nu_bands (1D array): wavenumber grid for the CKD, (Nbands)

        Returns:
            1D array: emission spectrum (Nbands,)
        """

        nlayer, Ng, Nbands = dtau_ckd.shape
        #sourcef = piBarr(temperature, jnp.tile(nu_bands, Ng))
        sourcef = jnp.tile(piBarr(temperature, nu_bands), Ng)


        flux_ckd = rtrun_emis_pureabs_ibased(
            dtau_ckd.reshape((nlayer, Ng * Nbands)), sourcef, self.mus, self.weights
        )
        flux_ckd = flux_ckd.reshape((Ng, Nbands))
        # integrate over the Gaussian quadrature
        return jnp.einsum("n,nm->m", weights, flux_ckd)


class OpartEmisPure(ArtCommon):

    def __init__(
        self,
        opalayer,
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
        nstream=8,
    ):
        """Initialization of OpartEmisPure

        Args:
            opalayer (class): user defined class, needs to define self.nu_grid
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nstream (int, optional): the number of the gaussian quadrature. Defaults to 8.
        """
        super().__init__(pressure_top, pressure_btm, nlayer, opalayer.nu_grid)
        self.nstream = nstream
        self.mus, self.weights = initialize_gaussian_quadrature(self.nstream)
        self.opalayer = opalayer
        self.nu_grid = self.opalayer.nu_grid

    def update_layer(self, carry_tauflux, params):
        """updates the layer opacity and flux

        Args:
            carry_tauflux (list): carry for the tau and flux
            params (list): layer parameters for this layer, params[0] should be temperature

        Returns:
            list: updated carry_tauflux
        """
        tauup, flux = carry_tauflux
        taulow = self.update_layeropacity(tauup, params)
        flux = self.update_layerflux(params[0], tauup, taulow, flux)
        return (taulow, flux)

    def update_layeropacity(self, tauup, params):
        """updates the optical depth of the layer

        Notes:
            up = n, low = n+1 in (44) of Paper II

        Args:
            tauup (array): optical depth at the upper layer [Nnus]
            params : layer parameters for this layer, params[0] should be temperature

        Returns:
            array: taulow (optical depth at the lower layer, [Nnus])
        """
        return tauup + self.opalayer(params)

    def update_layerflux(self, temperature, tauup, taulow, flux):
        """updates the flux of the layer

        Args:
            temperature (float): temperature of the layer, usually params[0] is used
            tauup (array): optical depth at the upper layer [Nnus]
            taulow (array): optical depth at the lower layer [Nnus]
            flux (array): flux array to be updated

        Returns:
            array: updated flux  [Nnus]
        """
        sourcef = piB(temperature, self.opalayer.nu_grid)
        flux = flux + 2.0 * sourcef * fluxsum_scan(
            tauup, taulow, self.mus, self.weights
        )
        return flux

    # --------------------------------------------------------
    # Developer Note (Hajime Kawahara Dec.7 2024):
    # If you wanna refactor this method, read Issue 542 on github.
    # In particular, we do not understand yet how layer_update_function can be included in the class witout the overhead of XLA compilation for each loop.
    # Use forward_time_opart.py and ensure the computation time is not changed (or use a profiler to check if no overhead is added for i>0 loops).
    # --------------------------------------------------------
    def __call__(self, layer_params, layer_update_function):
        """computes outgoing flux

        Args:
            layer_params (list): user defined layer parameters, layer_params[0] should be temperature array
            layer_update_function (method):

        Returns:
            array: flux [Nnus]
        """
        Nnus = len(self.opalayer.nu_grid)
        init_tauintensity = (jnp.zeros(Nnus), jnp.zeros(Nnus))
        tauflux, _ = scan(
            # for the reason not putting unroll option see #546
            # layer_update_function, init_tauintensity, layer_params, unroll=False
            layer_update_function,
            init_tauintensity,
            layer_params,
        )
        return tauflux[1]

    def run(self, opalayer, layer_params, flbl):
        return self(opalayer, layer_params, flbl)


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

    def run_ckd(self, dtau_ckd, single_scattering_albedo, asymmetric_parameter, 
                temperature, weights, nu_bands):
        """run radiative transfer for CKD with scattering
        
        Args:
            dtau_ckd (3D array): optical depth tensor, dtau (N_layer, Ng, Nbands)
            single_scattering_albedo (2D array): single scattering albedo (N_layer, Nbands)
            asymmetric_parameter (2D array): asymmetric parameter (N_layer, Nbands)
            temperature (1D array): temperature profile (Nlayer,)
            weights (1D array): weights for the Gaussian quadrature (Ng,)
            nu_bands (1D array): wavenumber grid for the CKD, (Nbands)
            
        Returns:
            1D array: emission spectrum (Nbands,)
        """
        nlayer, Ng, Nbands = dtau_ckd.shape
        
        # Create source function and tile scattering parameters to match dtau_ckd shape
        sourcef = jnp.tile(piBarr(temperature, nu_bands), Ng)
        
        # Reshape 3D dtau to 2D and tile 2D scattering parameters to match
        dtau_2d = dtau_ckd.reshape((nlayer, Ng * Nbands))
        ssa_2d = jnp.tile(single_scattering_albedo, Ng)
        g_2d = jnp.tile(asymmetric_parameter, Ng)
        
        if self.rtsolver == "lart_toon_hemispheric_mean":
            (spectrum, _, _, _, _, _) = rtrun_emis_scat_lart_toonhm(
                dtau_2d, ssa_2d, g_2d, sourcef
            )
        elif self.rtsolver == "fluxadding_toon_hemispheric_mean":
            spectrum = rtrun_emis_scat_fluxadding_toonhm(
                dtau_2d, ssa_2d, g_2d, sourcef
            )
        else:
            raise ValueError(f"Unknown rtsolver for CKD: {self.rtsolver}")
            
        # Reshape back to (Ng, Nbands) and integrate over g-ordinates
        flux_ckd = spectrum.reshape((Ng, Nbands))
        return jnp.einsum("n,nm->m", weights, flux_ckd)


class OpartEmisScat(ArtCommon):
    """Opart verision of ArtEmisScat.

    This class computes the outgoing emission flux of the atmosphere with scattering in the atmospheric layers.
    Radiative transfer scheme: flux-based two-stream method, using flux-adding treatment, Toon-type hemispheric mean approximation

    """

    def __init__(self, opalayer, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100):
        """Initialization of OpartReflectPure

        Args:
            opalayer (class): user defined class, needs to define self.nu_grid
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
        """
        super().__init__(pressure_top, pressure_btm, nlayer, opalayer.nu_grid)
        self.opalayer = opalayer
        self.nu_grid = self.opalayer.nu_grid

    def update_layer(self, carry_rs, params):
        """updates the layer opacity and effective reflectivity (Rphat) and source (Sphat)

        Args:
            carry_rs (list): carry for the Rphat and Sphat
            params (list): layer parameters for this layer

        Returns:
            list: updated carry_rs
        """
        Rphat_prev, Sphat_prev = carry_rs

        # blackbody source term in the layers
        temparature = params[0]
        source_vector = piB(temparature, self.nu_grid)
        # -------------------------------------------------
        dtau, single_scattering_albedo, asymmetric_parameter = self.opalayer(params)
        trans_coeff_i, scat_coeff_i, pihatB_i, _, _, _ = setrt_toonhm(
            dtau, single_scattering_albedo, asymmetric_parameter, source_vector
        )
        denom = 1.0 - scat_coeff_i * Rphat_prev
        Sphat_each = (
            pihatB_i + trans_coeff_i * (Sphat_prev + pihatB_i * Rphat_prev) / denom
        )
        Rphat_each = scat_coeff_i + trans_coeff_i**2 * Rphat_prev / denom
        carry_rs = [Rphat_each, Sphat_each]
        return carry_rs

    def __call__(
        self,
        layer_params,
        layer_update_function,
    ):
        """computes outgoing flux

        Args:
            layer_params (list): user defined layer parameters, layer_params[0] should be temperature array
            layer_update_function (method):

        Returns:
            array: flux [Nnus]
        """
        # no reflection at the bottom
        reflectivity_bottom = jnp.zeros_like(self.nu_grid)
        # no source term at the bottom
        source_bottom = jnp.zeros_like(self.nu_grid)
        rs_bottom = [reflectivity_bottom, source_bottom]
        rs, _ = scan(layer_update_function, rs_bottom, layer_params)
        return rs[1]

    def run(self, opalayer, layer_params, flbl):
        return self(opalayer, layer_params, flbl)
