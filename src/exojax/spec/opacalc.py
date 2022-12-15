"""opacity calculator class
"""

from exojax.spec import initspec
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import nx_from_resolution_eslog
import numpy as np


class OpaCalc():
    """Common Opacity Calculator Class
    """
    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.opainfo = None


class OpaPremodit(OpaCalc):
    def __init__(self, nu0, nu1, resolution=700000, Nx=None, mdb=None):
        super().__init__()

        if Nx is None:
            Nx = nx_from_resolution_eslog(nu0, nu1, resolution)
        if np.mod(Nx, 2) == 1:
            Nx = Nx + 1

        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            nu0, nu1, Nx, unit='AA', xsmode="premodit")

        #default setting
        self.dit_grid_resolution = 0.2
        self.diffmode = 1
        self.warning = True
        #need to refine
        self.Twt = 650.0
        self.Tref = 800.0
        self.dE = 1200.0

        if mdb is not None:
            self.setmdb(mdb)

    def set_gamma_and_n_Texp(self, mdb):
        if mdb.dbtype == "hitran":
            print("gamma_air and temperature exponent are used.")
            self.gamma_ref = mdb.gamma_air
            self.n_Texp = mdb.n_air
        elif mdb.dbtype == "exomol":
            self.gamma_ref = mdb.alpha_ref
            self.n_Texp = mdb.n_Texp

    def setmdb(self, mdb):
        print("Set mdb. opainfo is now available.")
        self.dbtype = mdb.dbtype
        self.set_gamma_and_n_Texp(mdb)
        self.opainfo = initspec.init_premodit(
            mdb.nu_lines,
            self.nu_grid,
            mdb.elower,
            self.gamma_ref,
            self.n_Texp,
            mdb.line_strength_ref,
            self.Twt,
            Tref=self.Tref,
            dE=self.dE,
            dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            warning=self.warning)
