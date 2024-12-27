"""checks the forward model of the opart spectrum
"""

from exojax.spec.opart import OpartReflectPure
from exojax.spec.layeropacity import single_layer_optical_depth
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def test_forward_opart():
    from exojax.spec.opacalc import OpaPremodit
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    import pandas as pd
    import numpy as np
    import pkg_resources
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF

    from jax import config

    config.update("jax_enable_x64", True)

    class OpaLayer:
    # user defined class, needs to define self.nugrid
        def __init__(self, Nnus=150000):
            self.nu_grid, self.wav, self.resolution = mock_wavenumber_grid()
            self.gravity = 2478.57
            self.mdb_co = mock_mdbExomol()

            self.opa_co = OpaPremodit(
                self.mdb_co, self.nu_grid, auto_trange=[400.0, 1500.0]
            )



        def __call__(self, params):
            temperature, pressure, dP, mixing_ratio = params
            xsv_co = self.opa_co.xsvector(temperature, pressure)
            dtau_co = single_layer_optical_depth(
                xsv_co, dP, mixing_ratio, self.mdb_co.molmass, self.gravity
            )
            single_scattering_albedo = jnp.ones_like(dtau_co) * 0.0001
            asymmetric_parameter = jnp.ones_like(dtau_co) * 0.0001

            return dtau_co, single_scattering_albedo, asymmetric_parameter

    opalayer = OpaLayer()
    opart = OpartReflectPure(
        opalayer, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nstream=8
    )

    def layer_update_function(carry_ip1, params):
        carry_ip1 = opart.update_layer(carry_ip1, params)
        return carry_ip1, None

    temperature = opart.powerlaw_temperature(1300.0, 0.1)
    mixing_ratio = opart.constant_mmr_profile(0.01)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]

    albedo = 0.5
    incoming_flux = jnp.ones_like(opalayer.nu_grid)
    reflectivity_surface = albedo * jnp.ones_like(opalayer.nu_grid)
    source_bottom = jnp.zeros_like(opalayer.nu_grid)

    flux = opart(
        layer_params,
        layer_update_function,
        reflectivity_surface,
        source_bottom,
        incoming_flux,
    )

    opart.change_temperature_range(400.0, 1500.0)
    
    def layer_update_function(carry_tauflux, params):
        carry_tauflux = opart.update_layer(carry_tauflux, params)
        return carry_tauflux, None

    temperature = opart.powerlaw_temperature(1300.0, 0.1)
    temperature = opart.clip_temperature(temperature)
    mixing_ratio = opart.constant_mmr_profile(0.0001)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
    flux = opart(layer_params, layer_update_function)

    filename = pkg_resources.resource_filename(
        "exojax", "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
    )
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(flux / dat["flux"].values - 1.0)
    print(np.max(residual))

    assert np.max(residual) < 0.0056  #  0.005548556139982397 2024/12/07
    plot = False
    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(211)
        plt.plot(opalayer.nu_grid, flux)
        plt.plot(dat["nus"].values, dat["flux"].values, ls="--")
        ax = fig.add_subplot(212)
        plt.plot(opalayer.nu_grid, residual)

        plt.savefig("forward_opart.png")


if __name__ == "__main__":
    test_forward_opart()
