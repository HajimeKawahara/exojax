from exojax.database.hitemp.api import MdbHitemp
from exojax.opacity import OpaPremodit
from exojax.utils.grids import wavenumber_grid
from jax import config
config.update("jax_enable_x64", True)


def plot_broadgrid_and_broadpar(broadening_resolution):
    nu_grid, wav, res = wavenumber_grid(4200.0,
                                        4400.0,
                                        10000,
                                        xsmode="premodit")
    mdb = MdbHitemp(".database/H2O/01_HITEMP2020/", nurange=nu_grid)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      auto_trange=[1000.0, 1200.0],
                      broadening_resolution=broadening_resolution)
    opa.plot_broadening_parameters()


if __name__ == "__main__":
    #bpr = {"mode":"manual", "value": 1.0}
    bpr = {"mode": "single", "value": None}
    #bpr = {"mode":"minmax", "value": None}
    plot_broadgrid_and_broadpar(bpr)
