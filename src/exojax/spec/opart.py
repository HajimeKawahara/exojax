from exojax.spec.atmrt import ArtCommon
from exojax.spec.rtlayer import fluxsum_scan

# from exojax.spec.rtlayer import fluxsum_vector  # same cost as fluxsum_scan
from exojax.spec.planck import piB
from exojax.spec.rtransfer import initialize_gaussian_quadrature
from exojax.spec.rtransfer import setrt_toonhm
from jax.lax import scan
import jax.numpy as jnp


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

