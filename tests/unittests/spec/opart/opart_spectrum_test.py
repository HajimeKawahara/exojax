"""checks the forward model of the opart spectrum"""

from exojax.rt import OpartEmisPure
from exojax.rt.layeropacity import single_layer_optical_depth
from jax import config

config.update("jax_enable_x64", True)


def test_forward_opart():
    from exojax.opacity import OpaPremodit
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    import pandas as pd
    import numpy as np
    from importlib.resources import files
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF

    from jax import config

    config.update("jax_enable_x64", True)

    class OpaLayer:
        # user defined class
        def __init__(self):
            self.mdb_co = mock_mdbExomol()
            self.nu_grid, _, _ = mock_wavenumber_grid()
            self.opa_co = OpaPremodit(
                self.mdb_co,
                self.nu_grid,
                auto_trange=[400.0, 1500.0],
                dit_grid_resolution=1.0,
            )
            self.gravity = 2478.57

        def __call__(self, params):
            temperature, pressure, dP, mixing_ratio = params
            xsv_co = self.opa_co.xsvector(temperature, pressure)
            dtau_co = single_layer_optical_depth(
                dP, xsv_co, mixing_ratio, self.mdb_co.molmass, self.gravity
            )
            return dtau_co

    opalayer = OpaLayer()
    opart = OpartEmisPure(
        opalayer, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nstream=8
    )
    opart.change_temperature_range(400.0, 1500.0)

    def layer_update_function(carry_tauflux, params):
        carry_tauflux = opart.update_layer(carry_tauflux, params)
        return carry_tauflux, None

    temperature = opart.powerlaw_temperature(1300.0, 0.1)
    temperature = opart.clip_temperature(temperature)
    mixing_ratio = opart.constant_profile(0.1)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
    flux = opart(layer_params, layer_update_function)
    filename = files("exojax").joinpath(
        "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
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
