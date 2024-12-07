from exojax.spec.atmrt import ArtCommon
from exojax.utils.constants import opfac
from exojax.spec.rtlayer import fluxsum_scan
from exojax.spec.rtlayer import fluxsum_vector  # same cost as fluxsum_scan
from exojax.spec.planck import piB
from exojax.spec.rtransfer import initialize_gaussian_quadrature
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
        """
        Initialization of OpartEmisPure

        Args:
            pressure_top (float, optional): top pressure in bar. Defaults to 1.0e-8.
            pressure_btm (float, optional): bottom pressure in bar. Defaults to 1.0e2.
            nlayer (int, optional): the number of the atmospheric layers. Defaults to 100.
            nu_grid (float, array, optional): the wavenumber grid. Defaults to None.
            nstream (int, optional): the number of stream. Defaults to 8. Should be 2 for rtsolver = fbased2st
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
            _type_: _description_
        """
        Nnus = len(self.opalayer.nu_grid)
        init_tauintensity = (jnp.zeros(Nnus), jnp.zeros(Nnus))
        tauflux, _ = scan(
            layer_update_function, init_tauintensity, layer_params, unroll=False
        )
        return tauflux[1]

    def run(self, opalayer, layer_params, flbl):
        return self(opalayer, layer_params, flbl)


if __name__ == "__main__":

    from exojax.spec.opacalc import OpaPremodit
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol

    from jax import config

    config.update("jax_enable_x64", True)

    class OpaLayer:
        # user defined class
        def __init__(self, opart):
            self.opart = opart
            self.mdb_co = mock_mdbExomol()
            Nnu = 100000
            self.nu_grid, _, _ = wavenumber_grid(
                1900.0, 2300.0, Nnu, unit="cm-1", xsmode="premodit"
            )

            self.opa_co = OpaPremodit(
                self.mdb_co, self.nu_grid, auto_trange=[400.0, 1500.0]
            )

        def __call__(self, parameters):
            temperature, pressure, dP, mixing_ratio = parameters
            xsv_co = self.opa_co.xsvector(temperature, pressure)
            gravity = 2478.57
            dtau_co = opart.opacity_layer_xs(
                xsv_co, dP, mixing_ratio, self.mdb_co.molmass, gravity
            )
            return dtau_co

    opart = OpartEmisPure(
        pressure_top=1.0e-5, pressure_btm=1.0e1, nlayer=200, nstream=8
    )
    opalayer = OpaLayer(opart)

    temperature = opart.powerlaw_temperature(1300.0, 0.1)
    mixing_ratio = opart.constant_mmr_profile(0.01)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
    flux = opart(opalayer, layer_params)

    import matplotlib.pyplot as plt

    plt.plot(opalayer.nu_grid, flux)
    plt.show()
