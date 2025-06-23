"""checks the forward model of the opart spectrum
"""
from importlib.resources import files
import pandas as pd
import numpy as np
import jax.numpy as jnp
from exojax.test.data import TESTDATA_CO_EXOMOL_PREMODIT_REFLECTION_REF
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.opacity import OpaPremodit
from exojax.rt import OpartReflectPure
from exojax.rt.layeropacity import single_layer_optical_depth

from jax import config
config.update("jax_enable_x64", True)

def test_forward_reflection_opart():
    class OpaLayer:
        # user defined class, needs to define self.nugrid
        def __init__(self):
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
                dP, xsv_co, mixing_ratio, self.mdb_co.molmass, self.gravity
            )
            single_scattering_albedo = jnp.ones_like(dtau_co) * 0.3
            asymmetric_parameter = jnp.ones_like(dtau_co) * 0.01

            return dtau_co, single_scattering_albedo, asymmetric_parameter

    opalayer = OpaLayer()
    opart = OpartReflectPure(
        opalayer, pressure_top=1.0e-6, pressure_btm=1.0e0, nlayer=200
    )

    def layer_update_function(carry, params):
        carry = opart.update_layer(carry, params)
        return carry, None

    temperature = opart.powerlaw_temperature(1300.0, 0.1)
    mixing_ratio = opart.constant_profile(0.0003)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]

    albedo = 1.0
    incoming_flux = jnp.ones_like(opalayer.nu_grid)
    reflectivity_surface = albedo * jnp.ones_like(opalayer.nu_grid)
    
    flux = opart(
        layer_params, layer_update_function, reflectivity_surface, incoming_flux
    )

    filename = files('exojax').joinpath('data/testdata/' + TESTDATA_CO_EXOMOL_PREMODIT_REFLECTION_REF)
  
    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))

    residual = np.abs(flux / dat["flux"].values - 1.0)
    print(np.max(residual))

    assert np.max(residual) < 1.0e-7 #3.176679863603482e-08 6/23 2025
    plot = False
    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(211)
        plt.plot(opalayer.nu_grid, flux)
        plt.plot(dat["nus"].values, dat["flux"].values, ls="--")
        ax = fig.add_subplot(212)
        plt.plot(opalayer.nu_grid, residual)

        plt.savefig("forward_opart_reflect.png")


if __name__ == "__main__":
    test_forward_reflection_opart()
