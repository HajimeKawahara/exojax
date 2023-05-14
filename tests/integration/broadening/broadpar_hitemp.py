from exojax.spec.api import MdbHitemp
from exojax.spec.opacalc import OpaPremodit
from exojax.utils.grids import wavenumber_grid


def plot_broadgrid_and_broadpar(broadening_parameter_resolution):
    nu_grid, wav, res = wavenumber_grid(4200.0,
                                        4400.0,
                                        10000,
                                        xsmode="premodit")
    mdb = MdbHitemp(".database/H2O/01_HITEMP2020/", nurange=nu_grid)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      auto_trange=[1000.0, 1200.0],
                      broadening_parameter_resolution=broadening_parameter_resolution)
    opa.plot_broadening_parameters()


if __name__ == "__main__":
    #bpr = {"mode":"manual", "value": 0.3}
    #bpr = {"mode": "single", "value": None}
    bpr = {"mode":"minmax", "value": None}
    plot_broadgrid_and_broadpar(bpr)
