"""This example checks if the derivative of transmission spectrum is valid (not nan), see #464.

"""

import jax
import jax.numpy as jnp
from jax import config
import pandas as pd
import numpy as np
from importlib.resources import files

from exojax.opacity import OpaPremodit
from exojax.rt import ArtTransPure
from exojax.database.api  import MdbHitran
from exojax.utils.grids import wavenumber_grid
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ
from exojax.test.data import COMPDATA_TRANSMISSION_CO
from exojax.utils.grids import wav2nu
from exojax.postproc.specop import SopInstProfile

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


def test_transmission_is_differentiable():
    filename = files("exojax").joinpath("data/testdata/" + COMPDATA_TRANSMISSION_CO)
    dat = pd.read_feather(filename)
    wav = (dat["Wavelength[um]"],)
    rprs = dat["Rp/Rs"]
    inst_nus = wav2nu(np.array(wav), "um")

    # Model
    Nx = 3000
    nu_grid, wav, res = wavenumber_grid(
        22900.0, 26000.0, Nx, unit="AA", xsmode="premodit"
    )

    art = ArtTransPure(pressure_top=1.0e-15, pressure_btm=1.0e1, nlayer=100)
    art.change_temperature_range(490.0, 510.0)

    mdb = MdbHitran("CO", nu_grid, gpu_transfer=True, isotope=1)
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        auto_trange=[490.0, 510.0],
        dit_grid_resolution=1.0,
    )

    sop_inst = SopInstProfile(nu_grid, vrmax=100.0)

    def model(params):
        mmr_CO, mu_fid, T_fid, gravity_btm, radius_btm, RV = params
        Tarr = T_fid * np.ones_like(art.pressure)
        mmr_arr = art.constant_profile(mmr_CO)
        mmw = mu_fid * np.ones_like(art.pressure)
        gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)
        Rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
        Rp2_sample = sop_inst.sampling(Rp2, RV, inst_nus)
        return jnp.sqrt(Rp2_sample)

    def objective(params):
        return jnp.sum((np.array(rprs[::-1]) - model(params)) ** 2)

    # Gradient
    grad = jax.grad(objective)
    params = np.array([1, 28.00863, 500, gravity_jupiter(1.0, 1.0), RJ, 0])
    gradient = grad(params)

    print()
    print("Parameters: mmr_CO, mu_fid, T_fid, gravity_btm, radius_btm, RV")
    print("Gradient", gradient)

    assert np.all(gradient == gradient)


if __name__ == "__main__":
    test_transmission_is_differentiable()
