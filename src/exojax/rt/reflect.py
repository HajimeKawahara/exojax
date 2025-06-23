import jax.numpy as jnp
from exojax.rt.common import ArtCommon
from exojax.rt.planck import piB, piBarr
from exojax.rt.rtransfer import rtrun_reflect_fluxadding_toonhm, setrt_toonhm
from exojax.utils.indexing import get_smooth_index
from exojax.rt.common import ArtCommon

import jax.numpy as jnp
from jax.lax import scan


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


class OpartReflectPure(ArtCommon):
    """Opart verision of ArtReflectPure.

    This class computes the outgoing flux of the atmosphere with reflection, no emission from atmospheric layers nor surface.
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

        # no source term
        source_vector = jnp.zeros_like(self.nu_grid)
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
        reflectivity_bottom,
        incoming_flux,
    ):
        """computes outgoing flux

        Args:
            layer_params (list): user defined layer parameters, layer_params[0] should be temperature array
            layer_update_function (method):
            relfectivity_bottom (array): reflectivity at the bottom (Nnus)
            incoming_flux (array): incoming flux [Nnus]

        Returns:
            array: flux [Nnus]
        """
        # rs_bottom = (refectivity_bottom, source_bottom)
        source_bottom = jnp.zeros_like(self.nu_grid)
        rs_bottom = [reflectivity_bottom, source_bottom]
        rs, _ = scan(layer_update_function, rs_bottom, layer_params)
        return rs[0] * incoming_flux + rs[1]

    def run(self, opalayer, layer_params, flbl):
        return self(opalayer, layer_params, flbl)


class OpartReflectEmis(ArtCommon):
    """Opart verision of ArtReflectEmis.

    This class computes the outgoing flux of the atmosphere with reflection, with emission from atmospheric layers.
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
        source_bottom,
        reflectivity_bottom,
        incoming_flux,
    ):
        """computes outgoing flux

        Args:
            layer_params (list): user defined layer parameters, layer_params[0] should be temperature array
            layer_update_function (method):
            source_bottom (array): source at the bottom [Nnus]
            reflectivity_bottom (array): reflectivity at the bottom [Nnus]
            incoming_flux (array): incoming flux [Nnus]

        Returns:
            array: flux [Nnus]
        """
        rs_bottom = [reflectivity_bottom, source_bottom]
        rs, _ = scan(layer_update_function, rs_bottom, layer_params)
        return rs[0] * incoming_flux + rs[1]

    def run(self, opalayer, layer_params, flbl):
        return self(opalayer, layer_params, flbl)
